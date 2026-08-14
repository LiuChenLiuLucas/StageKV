"""Day-5 StageKV residency sweep: r in {0, 1, 2, 4}, fixed g=2.

This runner reuses the Day-4 attention patch and replaces its all-CPU cache
with a hybrid cache. The first ``r`` KV heads of every layer stay on GPU;
the remaining heads stay on CPU and are synchronously copied only when their
attention group is evaluated. The script is for correctness and placement,
not for performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache

from stagekv_cpu_g2_correctness import (
    CPUHeadOffloadPatch,
    compare_logit_distributions,
    compare_tensors,
    dynamic_cache_bytes,
    null_context,
    run_autoregressive,
    validate_structure,
)


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day5_residency"


class ResidentCPUHeadCache(Cache):
    """Hybrid cache with ``r`` persistent GPU KV heads and CPU overflow heads."""

    def __init__(self, layers: int, kv_heads: int, resident_heads: int) -> None:
        super().__init__()
        if not 0 <= resident_heads <= kv_heads:
            raise ValueError("resident_heads must be between 0 and kv_heads")
        self.layers = layers
        self.kv_heads = kv_heads
        self.resident_heads = resident_heads
        self.gpu_key_cache: list[torch.Tensor | None] = [None] * layers
        self.gpu_value_cache: list[torch.Tensor | None] = [None] * layers
        self.cpu_key_cache: list[torch.Tensor | None] = [None] * layers
        self.cpu_value_cache: list[torch.Tensor | None] = [None] * layers
        self.gpu_to_cpu_append_calls = 0
        self.gpu_to_cpu_bytes = 0
        self.cpu_to_gpu_group_transfers = 0
        self.cpu_to_gpu_bytes = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        key = self.gpu_key_cache[layer_idx]
        if key is None:
            key = self.cpu_key_cache[layer_idx]
        return 0 if key is None else int(key.shape[-2])

    def get_max_cache_shape(self) -> None:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self.append(layer_idx, key_states, value_states)
        # This method is present for the Transformers Cache protocol. The
        # patched attention path reads groups through load_group instead.
        return key_states, value_states

    @staticmethod
    def _bytes(tensor: torch.Tensor | None) -> int:
        return 0 if tensor is None else tensor.numel() * tensor.element_size()

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.device.type != "cuda" or value.device.type != "cuda":
            raise RuntimeError("new K/V states must be on CUDA")
        gpu_key = key[:, : self.resident_heads].detach().contiguous()
        gpu_value = value[:, : self.resident_heads].detach().contiguous()
        cpu_key = key[:, self.resident_heads :].detach().to("cpu", copy=True)
        cpu_value = value[:, self.resident_heads :].detach().to("cpu", copy=True)
        if self.gpu_key_cache[layer_idx] is None:
            self.gpu_key_cache[layer_idx] = gpu_key
            self.gpu_value_cache[layer_idx] = gpu_value
            self.cpu_key_cache[layer_idx] = cpu_key
            self.cpu_value_cache[layer_idx] = cpu_value
        else:
            self.gpu_key_cache[layer_idx] = torch.cat(
                (self.gpu_key_cache[layer_idx], gpu_key), dim=-2
            )
            self.gpu_value_cache[layer_idx] = torch.cat(
                (self.gpu_value_cache[layer_idx], gpu_value), dim=-2
            )
            self.cpu_key_cache[layer_idx] = torch.cat(
                (self.cpu_key_cache[layer_idx], cpu_key), dim=-2
            )
            self.cpu_value_cache[layer_idx] = torch.cat(
                (self.cpu_value_cache[layer_idx], cpu_value), dim=-2
            )
        self.gpu_to_cpu_append_calls += 1
        self.gpu_to_cpu_bytes += self._bytes(cpu_key) + self._bytes(cpu_value)

    def load_group(
        self,
        layer_idx: int,
        start_head: int,
        end_head: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gpu_key = self.gpu_key_cache[layer_idx]
        gpu_value = self.gpu_value_cache[layer_idx]
        cpu_key = self.cpu_key_cache[layer_idx]
        cpu_value = self.cpu_value_cache[layer_idx]
        if gpu_key is None or gpu_value is None or cpu_key is None or cpu_value is None:
            raise RuntimeError(f"layer {layer_idx} cache is not initialized")

        parts_key: list[torch.Tensor] = []
        parts_value: list[torch.Tensor] = []
        resident_end = min(end_head, self.resident_heads)
        if start_head < resident_end:
            parts_key.append(gpu_key[:, start_head:resident_end].to(dtype=dtype))
            parts_value.append(gpu_value[:, start_head:resident_end].to(dtype=dtype))

        cpu_start = max(start_head, self.resident_heads)
        if cpu_start < end_head:
            cpu_offset_start = cpu_start - self.resident_heads
            cpu_offset_end = end_head - self.resident_heads
            key_group = cpu_key[:, cpu_offset_start:cpu_offset_end].to(
                device=device, dtype=dtype
            )
            value_group = cpu_value[:, cpu_offset_start:cpu_offset_end].to(
                device=device, dtype=dtype
            )
            parts_key.append(key_group)
            parts_value.append(value_group)
            self.cpu_to_gpu_group_transfers += 1
            self.cpu_to_gpu_bytes += self._bytes(key_group) + self._bytes(value_group)

        if not parts_key:
            raise RuntimeError("empty KV group")
        return torch.cat(parts_key, dim=1), torch.cat(parts_value, dim=1)

    def cpu_cache_bytes(self) -> int:
        return sum(
            self._bytes(tensor)
            for key, value in zip(self.cpu_key_cache, self.cpu_value_cache)
            for tensor in (key, value)
        )

    def gpu_cache_bytes(self) -> int:
        return sum(
            self._bytes(tensor)
            for key, value in zip(self.gpu_key_cache, self.gpu_value_cache)
            for tensor in (key, value)
        )

    def cache_shape(self, layer_idx: int) -> list[int]:
        gpu_key = self.gpu_key_cache[layer_idx]
        cpu_key = self.cpu_key_cache[layer_idx]
        reference = gpu_key if gpu_key is not None else cpu_key
        if reference is None:
            raise RuntimeError(f"layer {layer_idx} cache is not initialized")
        return [
            int(reference.shape[0]),
            self.kv_heads,
            int(reference.shape[-2]),
            int(reference.shape[-1]),
        ]

    def all_cache_tensors_in_expected_place(self) -> bool:
        return all(
            (tensor is None or tensor.device.type == "cuda")
            for key, value in zip(self.gpu_key_cache, self.gpu_value_cache)
            for tensor in (key, value)
        ) and all(
            (tensor is None or tensor.device.type == "cpu")
            for key, value in zip(self.cpu_key_cache, self.cpu_value_cache)
            for tensor in (key, value)
        )


def run_residency_case(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    *,
    layers: int,
    kv_heads: int,
    resident_heads: int,
    decode_tokens: int,
    query_heads: int,
) -> dict[str, Any]:
    cache = ResidentCPUHeadCache(layers, kv_heads, resident_heads)
    patch = CPUHeadOffloadPatch(
        model,
        cache,
        query_heads=query_heads,
        kv_heads=kv_heads,
        kv_group_size=2,
    )
    result = run_autoregressive(
        model, input_ids, decode_tokens, cpu_patch=patch
    )
    result.update({"cache": cache, "patch": patch})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-total-variation", type=float, default=1e-2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.sequence_length < 1 or args.decode_tokens < 1:
        raise ValueError("sequence-length and decode-tokens must be positive")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // query_heads)
    resident_values = (0, 1, 2, 4)
    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} g=2 r={resident_values}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        device="cuda",
    )

    # The standard run is performed once and reused as the reference for all r.
    print("running=standard_gpu_dynamic_cache", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)
    theoretical_total = (
        2 * layers * kv_heads * (args.sequence_length + args.decode_tokens - 1)
        * head_dim * 2
    )
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for resident_heads in resident_values:
        torch.cuda.empty_cache()
        print(f"running=stagekv_r{resident_heads}_g2_sync", flush=True)
        candidate = run_residency_case(
            model,
            input_ids,
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            decode_tokens=args.decode_tokens,
            query_heads=query_heads,
        )
        cache: ResidentCPUHeadCache = candidate["cache"]
        patch: CPUHeadOffloadPatch = candidate["patch"]
        logits_checks = [
            compare_tensors(a, b, rtol=args.rtol, atol=args.atol)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        dist_checks = [
            compare_logit_distributions(a, b)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        max_tv = max(item["total_variation"] for item in dist_checks)
        min_overlap = min(item["top10_overlap"] for item in dist_checks)
        expected_calls = layers * args.decode_tokens
        expected_transfers = expected_calls * sum(
            1
            for group_start in (0, 2)
            if max(group_start, resident_heads) < group_start + 2
        )
        expected_total = theoretical_total
        expected_gpu = (
            2 * layers * resident_heads * (args.sequence_length + args.decode_tokens - 1)
            * head_dim * 2
        )
        expected_cpu = expected_total - expected_gpu
        placement_correct = (
            cache.all_cache_tensors_in_expected_place()
            and cache.gpu_cache_bytes() == expected_gpu
            and cache.cpu_cache_bytes() == expected_cpu
        )
        transfer_correct = (
            patch.attention_calls == expected_calls
            and cache.gpu_to_cpu_append_calls == expected_calls
            and cache.cpu_to_gpu_group_transfers == expected_transfers
        )
        behavioral = (
            standard["generated_ids"] == candidate["generated_ids"]
            and all(item["top1_equal"] for item in dist_checks)
            and max_tv <= args.max_total_variation
            and min_overlap >= 0.9
            and placement_correct
            and transfer_correct
        )
        report = {
            "status": "PASS_BEHAVIORAL_EQUIVALENCE" if behavioral else "FAIL",
            "resident_kv_heads_r": resident_heads,
            "kv_group_size_g": 2,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "same_generated_sequence": standard["generated_ids"] == candidate["generated_ids"],
            "all_step_top1_equal": all(item["top1_equal"] for item in dist_checks),
            "max_probability_total_variation": max_tv,
            "min_top10_overlap": min_overlap,
            "standard_gpu_cache_bytes": standard["cache_bytes"],
            "stagekv_gpu_resident_cache_bytes": cache.gpu_cache_bytes(),
            "stagekv_cpu_cache_bytes": cache.cpu_cache_bytes(),
            "theoretical_total_cache_bytes": expected_total,
            "expected_gpu_resident_cache_bytes": expected_gpu,
            "expected_cpu_cache_bytes": expected_cpu,
            "placement_correct": placement_correct,
            "expected_attention_calls": expected_calls,
            "observed_attention_calls": patch.attention_calls,
            "expected_cpu_to_gpu_group_transfers": expected_transfers,
            "observed_cpu_to_gpu_group_transfers": cache.cpu_to_gpu_group_transfers,
            "observed_gpu_to_cpu_append_calls": cache.gpu_to_cpu_append_calls,
            "gpu_to_cpu_append_bytes": cache.gpu_to_cpu_bytes,
            "cpu_to_gpu_group_bytes": cache.cpu_to_gpu_bytes,
            "transfer_counts_correct": transfer_correct,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "stagekv_sync_elapsed_seconds": candidate["elapsed_seconds"],
            "max_step_logits_relative_l2_error": max(item["relative_l2_error"] for item in logits_checks),
            "min_step_logits_cosine_similarity": min(item["cosine_similarity"] for item in logits_checks),
            "note": "Correctness and placement only; synchronous transfers are not a performance claim.",
        }
        reports.append(report)
        rows.append(report.copy())
        del candidate, cache, patch

    report_path = Path(args.results_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    (report_path / "stagekv_residency.json").write_text(
        json.dumps({"model": args.model_path, "results": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(rows[0].keys())
    with (report_path / "stagekv_residency.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"model": args.model_path, "results": reports}, ensure_ascii=False, indent=2))
    print(f"saved={report_path / 'stagekv_residency.json'}")
    print(f"saved={report_path / 'stagekv_residency.csv'}")
    if not all(item["status"] == "PASS_BEHAVIORAL_EQUIVALENCE" for item in reports):
        raise RuntimeError("At least one StageKV residency case failed")
    print("stagekv_residency=PASS_ALL_R_VALUES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
