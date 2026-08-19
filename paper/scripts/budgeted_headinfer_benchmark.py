from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import headinfer_reference_benchmark as base
from headinfer.mp import mp_headinfer


class BudgetedResidentHeadInferCache(
    base.InstrumentedHeadInferCache
):
    """
    Synchronous resident-head cache.

    A selected set of KV head blocks remains on GPU permanently.
    Other KV head blocks are synchronously moved between CPU and GPU.
    This version is intended for correctness validation, not performance.
    """

    def __init__(
        self,
        total_head_blocks: int,
        resident_head_blocks: int,
    ) -> None:
        super().__init__()

        if not 0 <= resident_head_blocks <= total_head_blocks:
            raise ValueError(
                "resident_head_blocks must be within total_head_blocks"
            )

        if resident_head_blocks == 0:
            self.resident_indices = set()
        else:
            self.resident_indices = {
                (index * total_head_blocks) // resident_head_blocks
                for index in range(resident_head_blocks)
            }

    def _check_gpu_resident(
        self,
        layer_idx: int,
    ) -> None:
        key_tensor = self.key_cache[layer_idx]
        value_tensor = self.value_cache[layer_idx]

        if key_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Resident key block {layer_idx} is not on GPU: "
                f"{key_tensor.device}"
            )

        if value_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Resident value block {layer_idx} is not on GPU: "
                f"{value_tensor.device}"
            )

    def prefetch_layer(
        self,
        layer_idx: int,
    ):
        if layer_idx >= len(self):
            return None

        if layer_idx in self.resident_indices:
            self._check_gpu_resident(layer_idx)
            return None

        target_device = self.original_device[layer_idx]

        for cache in (
            self.key_cache,
            self.value_cache,
        ):
            tensor = cache[layer_idx]

            if tensor.device.type == "cpu":
                self.h2d_tensor_copies += 1
                self.h2d_bytes += base.tensor_bytes(tensor)

                cache[layer_idx] = tensor.to(
                    target_device,
                    non_blocking=False,
                )

        torch.cuda.synchronize()
        return None

    def evict_previous_layer(
        self,
        layer_idx: int,
    ):
        if len(self) <= 2:
            return None

        previous_idx = (layer_idx - 1) % len(self)

        if previous_idx in self.resident_indices:
            self._check_gpu_resident(previous_idx)
            return None

        torch.cuda.synchronize()

        for cache in (
            self.key_cache,
            self.value_cache,
        ):
            tensor = cache[previous_idx]

            if tensor.device.type == "cuda":
                self.d2h_tensor_copies += 1
                self.d2h_bytes += base.tensor_bytes(tensor)

                cache[previous_idx] = tensor.to(
                    "cpu",
                    non_blocking=False,
                )

        return None

    def __getitem__(
        self,
        layer_idx: int,
    ):
        if not 0 <= layer_idx < len(self):
            raise KeyError(
                f"Cache has {len(self)} blocks, "
                f"requested block {layer_idx}"
            )

        self.evict_previous_layer(layer_idx)
        self.prefetch_layer(layer_idx)

        key_tensor = self.key_cache[layer_idx]
        value_tensor = self.value_cache[layer_idx]

        if key_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Current key block {layer_idx} is not on GPU"
            )

        if value_tensor.device.type != "cuda":
            raise RuntimeError(
                f"Current value block {layer_idx} is not on GPU"
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

        return key_tensor, value_tensor


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=("resident",),
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
        default=8192,
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

    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        print(
            "WARNING: CUDA_LAUNCH_BLOCKING=1 is not enabled",
            flush=True,
        )

    if args.resident_head_blocks < 0:
        raise ValueError(
            "resident-head-blocks must be non-negative"
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
            f"resident-head-blocks={args.resident_head_blocks} "
            f"exceeds total-head-blocks={total_head_blocks}"
        )

    print(
        f"method=resident "
        f"model={args.model_path} "
        f"layers={layers} "
        f"query_heads={query_heads} "
        f"kv_heads={kv_heads} "
        f"head_dim={head_dim} "
        f"total_head_blocks={total_head_blocks} "
        f"resident_head_blocks={args.resident_head_blocks}",
        flush=True,
    )

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

        return BudgetedResidentHeadInferCache(
            total_head_blocks=total_head_blocks,
            resident_head_blocks=args.resident_head_blocks,
        )

    base.build_cache = build_cache

    generator = torch.Generator(
        device="cpu",
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
            f"warmup={index + 1}/{args.warmup_repeats}",
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
        "headinfer_reference_resident_"
        f"{args.sequence_length}_raw.csv"
    )

    report_path = args.results_dir / (
        "headinfer_reference_resident_"
        f"{args.sequence_length}_report.json"
    )

    base.write_csv(
        raw_path,
        rows,
    )

    generated_sequences = {
        tuple(
            json.loads(row["generated_token_ids"])
        )
        for row in rows
    }

    sequence_stable = (
        len(generated_sequences) == 1
    )

    transfer_values = [
        float(row["h2d_gib"])
        + float(row["d2h_gib"])
        for row in rows
    ]

    gpu_values = [
        float(row["cache_gpu_gib"])
        for row in rows
    ]

    latency_values = [
        float(row["decode_end_to_end_ms_per_token"])
        for row in rows
    ]

    report = {
        "method": "resident",
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
        },
        "results": {
            "decode_end_to_end_ms_per_token_mean": (
                sum(latency_values)
                / len(latency_values)
            ),
            "cache_gpu_gib_mean": (
                sum(gpu_values)
                / len(gpu_values)
            ),
            "total_transfer_gib_mean": (
                sum(transfer_values)
                / len(transfer_values)
            ),
            "generated_sequence_stable_across_trials": (
                sequence_stable
            ),
        },
        "rows": rows,
        "note": (
            "Synchronous resident KV-head prototype. "
            "This run validates cache placement and "
            "sequence stability before performance testing."
        ),
    }

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

    if sequence_stable:
        print("resident_benchmark=PASS")
        return 0

    print(
        "resident_benchmark=FAIL "
        "(generated sequence is unstable)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())