from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import headinfer_reference_benchmark as base
from headinfer.mp import mp_headinfer


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def parse_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list):
        raise TypeError(
            f"Invalid generated_token_ids type: {type(value)}"
        )

    return tuple(int(token) for token in value)


def get_rows(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = report.get("rows")

    if not isinstance(rows, list) or not rows:
        raise RuntimeError(
            "Report does not contain a non-empty rows list"
        )

    return rows


def get_sequences(
    report: dict[str, Any],
) -> list[tuple[int, ...]]:
    rows = get_rows(report)

    return [
        parse_token_ids(row["generated_token_ids"])
        for row in rows
    ]


def mean_column(
    rows: list[dict[str, Any]],
    key: str,
) -> float:
    values = [
        float(row[key])
        for row in rows
    ]

    return sum(values) / len(values)


def build_resident_indices(
    total_head_blocks: int,
    resident_head_blocks: int,
) -> set[int]:
    if not 0 <= resident_head_blocks <= total_head_blocks:
        raise ValueError(
            "resident_head_blocks must be within "
            "total_head_blocks"
        )

    if resident_head_blocks == 0:
        return set()

    return {
        (index * total_head_blocks)
        // resident_head_blocks
        for index in range(resident_head_blocks)
    }


class AsyncResidentHeadInferCache(
    base.InstrumentedHeadInferCache
):
    """
    Asynchronous resident KV cache.

    Resident blocks remain on GPU.
    Non-resident blocks use explicit H2D/D2H CUDA streams.
    CUDA events connect D2H completion to later H2D use.
    """

    def __init__(
        self,
        total_head_blocks: int,
        resident_head_blocks: int,
    ) -> None:
        super().__init__()

        self.resident_indices = build_resident_indices(
            total_head_blocks=total_head_blocks,
            resident_head_blocks=resident_head_blocks,
        )

        self.pending_d2h_events: dict[
            int,
            torch.cuda.Event,
        ] = {}

    def _check_resident_gpu(
        self,
        layer_idx: int,
    ) -> None:
        key_tensor = self.key_cache[layer_idx]
        value_tensor = self.value_cache[layer_idx]

        if key_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Resident key block {layer_idx} "
                f"is not on GPU: {key_tensor.device}"
            )

        if value_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Resident value block {layer_idx} "
                f"is not on GPU: {value_tensor.device}"
            )

    def prefetch_layer(
        self,
        layer_idx: int,
    ):
        if layer_idx >= len(self):
            return None

        if layer_idx in self.resident_indices:
            self._check_resident_gpu(layer_idx)
            return None

        d2h_event = self.pending_d2h_events.pop(
            layer_idx,
            None,
        )

        with torch.cuda.stream(self.prefetch_stream):
            if d2h_event is not None:
                self.prefetch_stream.wait_event(
                    d2h_event
                )

            target_device = self.original_device[
                layer_idx
            ]

            for cache in (
                self.key_cache,
                self.value_cache,
            ):
                tensor = cache[layer_idx]

                if tensor.device.type == "cpu":
                    self.h2d_tensor_copies += 1
                    self.h2d_bytes += base.tensor_bytes(
                        tensor
                    )

                    cache[layer_idx] = tensor.to(
                        target_device,
                        non_blocking=True,
                    )

        return None

    def evict_previous_layer(
        self,
        layer_idx: int,
    ):
        if len(self) <= 2:
            return None

        previous_idx = (
            layer_idx - 1
        ) % len(self)

        if previous_idx in self.resident_indices:
            self._check_resident_gpu(previous_idx)
            return None

        # The current compute stream must finish using
        # the previous block before D2H begins.
        torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(self.evit_stream):
            for cache in (
                self.key_cache,
                self.value_cache,
            ):
                tensor = cache[previous_idx]

                if tensor.device.type == "cuda":
                    self.d2h_tensor_copies += 1
                    self.d2h_bytes += base.tensor_bytes(
                        tensor
                    )

                    cache[previous_idx] = tensor.to(
                        "cpu",
                        non_blocking=True,
                    )

            event = torch.cuda.Event(
                enable_timing=False
            )
            event.record(self.evit_stream)

        self.pending_d2h_events[previous_idx] = event

        return None

    def __getitem__(
        self,
        layer_idx: int,
    ):
        if not 0 <= layer_idx < len(self):
            raise KeyError(
                f"Cache has {len(self)} blocks; "
                f"requested {layer_idx}"
            )

        self.evict_previous_layer(layer_idx)

        # Normally the previous __getitem__ already
        # prefetched this block. This fallback handles
        # the first access after a D2H operation.
        if (
            self.key_cache[layer_idx].device.type
            == "cpu"
        ):
            self.prefetch_layer(layer_idx)

        # Make all queued H2D operations visible to
        # the current compute stream.
        self.prefetch_stream.synchronize()

        key_tensor = self.key_cache[layer_idx]
        value_tensor = self.value_cache[layer_idx]

        if key_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Current key block {layer_idx} "
                "is not on GPU"
            )

        if value_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Current value block {layer_idx} "
                "is not on GPU"
            )

        if self.beam_idx is not None:
            beam_idx = self.beam_idx.to(
                self.original_device[layer_idx]
            )

            key_tensor = key_tensor.index_select(
                0,
                beam_idx,
            )
            value_tensor = value_tensor.index_select(
                0,
                beam_idx,
            )

        self.prefetch_layer(
            (layer_idx + 1) % len(self)
        )

        return key_tensor, value_tensor


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument(
        "--standard-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("correctness", "performance"),
        required=True,
    )
    parser.add_argument(
        "--resident-head-blocks",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=9216,
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    blocking_enabled = (
        os.environ.get("CUDA_LAUNCH_BLOCKING")
        == "1"
    )

    if args.mode == "correctness":
        if not blocking_enabled:
            raise RuntimeError(
                "Correctness mode requires "
                "CUDA_LAUNCH_BLOCKING=1"
            )
    else:
        if blocking_enabled:
            raise RuntimeError(
                "Performance mode requires "
                "CUDA_LAUNCH_BLOCKING to be unset"
            )

    if args.results_dir.exists() and any(
        args.results_dir.iterdir()
    ):
        raise RuntimeError(
            f"Results directory is not empty: "
            f"{args.results_dir}"
        )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    query_heads = int(config.num_attention_heads)

    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // query_heads
    )

    total_head_blocks = layers * kv_heads

    if args.resident_head_blocks > total_head_blocks:
        raise ValueError(
            "resident-head-blocks exceeds "
            "total-head-blocks"
        )

    standard_report = load_json(
        args.standard_report
    )
    standard_sequences = get_sequences(
        standard_report
    )

    if len(set(standard_sequences)) != 1:
        raise RuntimeError(
            "Standard reference sequence is unstable"
        )

    standard_sequence = standard_sequences[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()

    mp_headinfer(model)

    def build_cache(method: str):
        if method != "resident":
            raise ValueError(
                f"Unexpected method: {method}"
            )

        return AsyncResidentHeadInferCache(
            total_head_blocks=total_head_blocks,
            resident_head_blocks=(
                args.resident_head_blocks
            ),
        )

    base.build_cache = build_cache

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(args.seed)

    input_ids = torch.randint(
        low=10,
        high=int(config.vocab_size) - 10,
        size=(1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to("cuda")

    for index in range(args.warmup_repeats):
        print(
            f"warmup={index + 1}/"
            f"{args.warmup_repeats}",
            flush=True,
        )

        base.run_once(
            model,
            method="resident",
            input_ids=input_ids,
            decode_tokens=args.decode_tokens,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    rows = []

    for trial in range(1, args.repeats + 1):
        print(
            f"trial={trial}/{args.repeats}",
            flush=True,
        )

        row = base.run_once(
            model,
            method="resident",
            input_ids=input_ids,
            decode_tokens=args.decode_tokens,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

        row["trial"] = trial
        row["resident_head_blocks"] = (
            args.resident_head_blocks
        )

        rows.append(row)

    raw_path = args.results_dir / (
        "async_resident_"
        f"{args.sequence_length}_raw.csv"
    )

    report_path = args.results_dir / (
        "async_resident_"
        f"{args.sequence_length}_report.json"
    )

    base.write_csv(
        raw_path,
        rows,
    )

    generated_sequences = [
        parse_token_ids(
            row["generated_token_ids"]
        )
        for row in rows
    ]

    stable = len(
        set(generated_sequences)
    ) == 1

    matches_standard = all(
        sequence == standard_sequence
        for sequence in generated_sequences
    )

    correct = stable and matches_standard

    h2d_mean = mean_column(
        rows,
        "h2d_gib",
    )
    d2h_mean = mean_column(
        rows,
        "d2h_gib",
    )
    transfer_mean = h2d_mean + d2h_mean

    report = {
        "method": "async_resident",
        "mode": args.mode,
        "model_path": args.model_path,
        "sequence_length": args.sequence_length,
        "decode_tokens": args.decode_tokens,
        "warmup_repeats": args.warmup_repeats,
        "measured_repeats": args.repeats,
        "seed": args.seed,
        "resident_head_blocks": (
            args.resident_head_blocks
        ),
        "total_head_blocks": total_head_blocks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_launch_blocking": os.environ.get(
                "CUDA_LAUNCH_BLOCKING",
                "",
            ),
        },
        "results": {
            "stable_across_trials": stable,
            "matches_standard": matches_standard,
            "correct": correct,
            "decode_end_to_end_ms_per_token_mean": (
                mean_column(
                    rows,
                    "decode_end_to_end_ms_per_token",
                )
            ),
            "cache_gpu_gib_mean": mean_column(
                rows,
                "cache_gpu_gib",
            ),
            "h2d_gib_mean": h2d_mean,
            "d2h_gib_mean": d2h_mean,
            "total_transfer_gib_mean": transfer_mean,
        },
        "rows": rows,
        "note": (
            "Async resident benchmark. "
            "Correctness and performance must be "
            "reported separately."
        ),
    }

    if args.mode == "correctness":
        report["status"] = (
            "PASS_CORRECTNESS"
            if correct
            else "FAIL_CORRECTNESS"
        )
    else:
        report["status"] = (
            "VALID_PERFORMANCE_MEASUREMENT"
            if correct
            else "INVALID_PERFORMANCE_CORRECTNESS"
        )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report["results"],
            indent=2,
        )
    )
    print(f"raw={raw_path}")
    print(f"report={report_path}")
    print(f"status={report['status']}")

    if args.mode == "correctness":
        return 0 if correct else 1

    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())