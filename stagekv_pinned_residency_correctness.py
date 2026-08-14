"""Day-6 StageKV static pinned-cache sweep for r in {0, 1, 2, 4}.

The first r KV heads stay in a preallocated GPU cache. Remaining KV heads are
stored in a preallocated pinned CPU cache and copied to GPU with
``non_blocking=True`` in groups of at most two heads. This script validates
behavior, placement, pinning, capacity, and transfer accounting. It does not
make a performance claim; transfers are not overlapped with attention yet.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache

from stagekv_cpu_g2_correctness import (
    CPUHeadOffloadPatch,
    compare_logit_distributions,
    compare_tensors,
    run_autoregressive,
    validate_structure,
)


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day6_pinned_residency"
RESIDENT_VALUES = (0, 1, 2, 4)
KV_GROUP_SIZE = 2


class StaticPinnedResidentCache(Cache):
    """Preallocated hybrid KV cache with pinned CPU overflow storage."""

    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        resident_heads: int,
        max_cache_len: int,
    ) -> None:
        super().__init__()
        if not 0 <= resident_heads <= kv_heads:
            raise ValueError("resident_heads must be between 0 and kv_heads")
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        self.layers = layers
        self.kv_heads = kv_heads
        self.resident_heads = resident_heads
        self.cpu_heads = kv_heads - resident_heads
        self.max_cache_len = max_cache_len
        self.lengths = [0] * layers

        self.gpu_key_cache: list[torch.Tensor | None] = [None] * layers
        self.gpu_value_cache: list[torch.Tensor | None] = [None] * layers
        self.cpu_key_cache: list[torch.Tensor | None] = [None] * layers
        self.cpu_value_cache: list[torch.Tensor | None] = [None] * layers

        self.gpu_to_cpu_append_calls = 0
        self.gpu_to_cpu_bytes = 0
        self.cpu_to_gpu_group_transfers = 0
        self.cpu_to_gpu_bytes = 0
        self.non_blocking_h2d_calls = 0
        self.cache_growth_cat_calls = 0
        self.allocations = 0
        self._batch_size: int | None = None
        self._head_dim: int | None = None
        self._dtype: torch.dtype | None = None

    @staticmethod
    def _bytes(tensor: torch.Tensor | None) -> int:
        return 0 if tensor is None else tensor.numel() * tensor.element_size()

    def _allocate_from(self, key: torch.Tensor) -> None:
        if self._dtype is not None:
            return
        batch_size, observed_heads, _, head_dim = key.shape
        if observed_heads != self.kv_heads:
            raise RuntimeError(
                f"expected {self.kv_heads} raw KV heads, got {observed_heads}"
            )
        self._batch_size = int(batch_size)
        self._head_dim = int(head_dim)
        self._dtype = key.dtype
        for layer_idx in range(self.layers):
            if self.resident_heads:
                shape = (
                    batch_size,
                    self.resident_heads,
                    self.max_cache_len,
                    head_dim,
                )
                self.gpu_key_cache[layer_idx] = torch.empty(
                    shape, dtype=key.dtype, device=key.device
                )
                self.gpu_value_cache[layer_idx] = torch.empty(
                    shape, dtype=key.dtype, device=key.device
                )
                self.allocations += 2
            if self.cpu_heads:
                shape = (
                    batch_size,
                    self.cpu_heads,
                    self.max_cache_len,
                    head_dim,
                )
                self.cpu_key_cache[layer_idx] = torch.empty(
                    shape, dtype=key.dtype, device="cpu", pin_memory=True
                )
                self.cpu_value_cache[layer_idx] = torch.empty(
                    shape, dtype=key.dtype, device="cpu", pin_memory=True
                )
                self.allocations += 2

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx]

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self.append(layer_idx, key_states, value_states)
        return key_states, value_states

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.device.type != "cuda" or value.device.type != "cuda":
            raise RuntimeError("new K/V states must be on CUDA")
        if key.shape != value.shape:
            raise RuntimeError("key and value shapes differ")
        self._allocate_from(key)
        start = self.lengths[layer_idx]
        end = start + int(key.shape[-2])
        if end > self.max_cache_len:
            raise RuntimeError(
                f"cache capacity exceeded at layer {layer_idx}: "
                f"end={end}, capacity={self.max_cache_len}"
            )

        if self.resident_heads:
            gpu_key = self.gpu_key_cache[layer_idx]
            gpu_value = self.gpu_value_cache[layer_idx]
            assert gpu_key is not None and gpu_value is not None
            gpu_key[:, :, start:end].copy_(key[:, : self.resident_heads])
            gpu_value[:, :, start:end].copy_(value[:, : self.resident_heads])

        if self.cpu_heads:
            cpu_key = self.cpu_key_cache[layer_idx]
            cpu_value = self.cpu_value_cache[layer_idx]
            assert cpu_key is not None and cpu_value is not None
            source_key = key[:, self.resident_heads :]
            source_value = value[:, self.resident_heads :]
            # Keep D2H blocking in this stage. It guarantees that the pinned
            # buffer is ready before it becomes the source of the next H2D.
            cpu_key[:, :, start:end].copy_(source_key, non_blocking=False)
            cpu_value[:, :, start:end].copy_(source_value, non_blocking=False)
            self.gpu_to_cpu_bytes += self._bytes(source_key) + self._bytes(source_value)

        self.lengths[layer_idx] = end
        self.gpu_to_cpu_append_calls += 1

    def load_group(
        self,
        layer_idx: int,
        start_head: int,
        end_head: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= start_head < end_head <= self.kv_heads:
            raise ValueError("invalid KV-head group")
        length = self.lengths[layer_idx]
        if length == 0:
            raise RuntimeError(f"layer {layer_idx} cache is empty")
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []

        resident_end = min(end_head, self.resident_heads)
        if start_head < resident_end:
            gpu_key = self.gpu_key_cache[layer_idx]
            gpu_value = self.gpu_value_cache[layer_idx]
            assert gpu_key is not None and gpu_value is not None
            key_parts.append(gpu_key[:, start_head:resident_end, :length])
            value_parts.append(gpu_value[:, start_head:resident_end, :length])

        cpu_start = max(start_head, self.resident_heads)
        if cpu_start < end_head:
            cpu_key = self.cpu_key_cache[layer_idx]
            cpu_value = self.cpu_value_cache[layer_idx]
            assert cpu_key is not None and cpu_value is not None
            offset_start = cpu_start - self.resident_heads
            offset_end = end_head - self.resident_heads
            cpu_key_view = cpu_key[:, offset_start:offset_end, :length]
            cpu_value_view = cpu_value[:, offset_start:offset_end, :length]
            key_group = cpu_key_view.to(
                device=device, dtype=dtype, non_blocking=True
            )
            value_group = cpu_value_view.to(
                device=device, dtype=dtype, non_blocking=True
            )
            key_parts.append(key_group)
            value_parts.append(value_group)
            self.cpu_to_gpu_group_transfers += 1
            self.non_blocking_h2d_calls += 2
            self.cpu_to_gpu_bytes += self._bytes(key_group) + self._bytes(value_group)

        if not key_parts:
            raise RuntimeError("empty KV-head group")
        if len(key_parts) == 1:
            return key_parts[0], value_parts[0]
        # This concatenation joins resident and offloaded heads for r=1. It
        # does not grow the persistent cache and is not cache reallocation.
        return torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1)

    def cache_shape(self, layer_idx: int) -> list[int]:
        if self._batch_size is None or self._head_dim is None:
            raise RuntimeError("cache is not initialized")
        return [
            self._batch_size,
            self.kv_heads,
            self.lengths[layer_idx],
            self._head_dim,
        ]

    def used_gpu_bytes(self) -> int:
        if self._dtype is None or self._batch_size is None or self._head_dim is None:
            return 0
        return (
            2
            * self._batch_size
            * self.resident_heads
            * sum(self.lengths)
            * self._head_dim
            * torch.empty((), dtype=self._dtype).element_size()
        )

    def used_cpu_bytes(self) -> int:
        if self._dtype is None or self._batch_size is None or self._head_dim is None:
            return 0
        return (
            2
            * self._batch_size
            * self.cpu_heads
            * sum(self.lengths)
            * self._head_dim
            * torch.empty((), dtype=self._dtype).element_size()
        )

    def cpu_cache_bytes(self) -> int:
        """Compatibility hook used by the shared autoregressive runner."""
        return self.used_cpu_bytes()

    def allocated_gpu_bytes(self) -> int:
        return sum(
            self._bytes(tensor)
            for key, value in zip(self.gpu_key_cache, self.gpu_value_cache)
            for tensor in (key, value)
        )

    def allocated_cpu_bytes(self) -> int:
        return sum(
            self._bytes(tensor)
            for key, value in zip(self.cpu_key_cache, self.cpu_value_cache)
            for tensor in (key, value)
        )

    def all_cpu_buffers_pinned(self) -> bool:
        cpu_tensors = [
            tensor
            for key, value in zip(self.cpu_key_cache, self.cpu_value_cache)
            for tensor in (key, value)
            if tensor is not None
        ]
        return self.cpu_heads == 0 or (
            bool(cpu_tensors) and all(tensor.is_pinned() for tensor in cpu_tensors)
        )

    def placement_correct(self) -> bool:
        return all(
            tensor is None or tensor.device.type == "cuda"
            for key, value in zip(self.gpu_key_cache, self.gpu_value_cache)
            for tensor in (key, value)
        ) and all(
            tensor is None or tensor.device.type == "cpu"
            for key, value in zip(self.cpu_key_cache, self.cpu_value_cache)
            for tensor in (key, value)
        )


def expected_transfer_groups_per_attention(resident_heads: int) -> int:
    return sum(
        1
        for group_start in range(0, 4, KV_GROUP_SIZE)
        if max(group_start, resident_heads) < group_start + KV_GROUP_SIZE
    )


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
    max_cache_len = args.sequence_length + args.decode_tokens - 1
    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} max_cache_len={max_cache_len} "
        f"g={KV_GROUP_SIZE} r={RESIDENT_VALUES}",
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

    print("running=standard_gpu_dynamic_cache", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)
    element_size = 2  # BF16
    total_used_bytes = 2 * layers * kv_heads * max_cache_len * head_dim * element_size
    expected_attention_calls = layers * args.decode_tokens
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for resident_heads in RESIDENT_VALUES:
        torch.cuda.empty_cache()
        cache = StaticPinnedResidentCache(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        patch = CPUHeadOffloadPatch(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=stagekv_static_pinned_r{resident_heads}_g2", flush=True)
        candidate = run_autoregressive(
            model, input_ids, args.decode_tokens, cpu_patch=patch
        )
        torch.cuda.synchronize()

        logits_checks = [
            compare_tensors(a, b, rtol=args.rtol, atol=args.atol)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        distribution_checks = [
            compare_logit_distributions(a, b)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        max_tv = max(item["total_variation"] for item in distribution_checks)
        min_top10_overlap = min(item["top10_overlap"] for item in distribution_checks)
        expected_gpu_bytes = total_used_bytes * resident_heads // kv_heads
        expected_cpu_bytes = total_used_bytes - expected_gpu_bytes
        expected_transfer_calls = (
            expected_attention_calls
            * expected_transfer_groups_per_attention(resident_heads)
        )
        expected_allocations = layers * 2 * int(resident_heads > 0) + layers * 2 * int(
            resident_heads < kv_heads
        )
        placement_and_capacity_correct = (
            cache.placement_correct()
            and cache.all_cpu_buffers_pinned()
            and cache.used_gpu_bytes() == expected_gpu_bytes
            and cache.used_cpu_bytes() == expected_cpu_bytes
            and cache.allocated_gpu_bytes() == expected_gpu_bytes
            and cache.allocated_cpu_bytes() == expected_cpu_bytes
            and all(length == max_cache_len for length in cache.lengths)
            and cache.allocations == expected_allocations
            and cache.cache_growth_cat_calls == 0
        )
        transfer_counts_correct = (
            patch.attention_calls == expected_attention_calls
            and cache.gpu_to_cpu_append_calls == expected_attention_calls
            and cache.cpu_to_gpu_group_transfers == expected_transfer_calls
            and cache.non_blocking_h2d_calls == expected_transfer_calls * 2
        )
        same_sequence = standard["generated_ids"] == candidate["generated_ids"]
        all_top1_equal = all(item["top1_equal"] for item in distribution_checks)
        behavioral_equivalence = (
            same_sequence
            and all_top1_equal
            and max_tv <= args.max_total_variation
            and min_top10_overlap >= 0.9
            and placement_and_capacity_correct
            and transfer_counts_correct
        )
        checks = {
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "probability_total_variation_within_threshold": (
                max_tv <= args.max_total_variation
            ),
            "top10_overlap_within_threshold": min_top10_overlap >= 0.9,
            "placement_pinning_and_capacity_correct": placement_and_capacity_correct,
            "transfer_counts_correct": transfer_counts_correct,
        }
        report = {
            "status": (
                "PASS_BEHAVIORAL_EQUIVALENCE" if behavioral_equivalence else "FAIL"
            ),
            "resident_kv_heads_r": resident_heads,
            "kv_group_size_g": KV_GROUP_SIZE,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "max_cache_len": max_cache_len,
            "cache_implementation": "preallocated_static_pinned_cpu",
            "h2d_non_blocking_enabled": True,
            "async_overlap_enabled": False,
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "max_probability_total_variation": max_tv,
            "min_top10_overlap": min_top10_overlap,
            "max_step_logits_relative_l2_error": max(
                item["relative_l2_error"] for item in logits_checks
            ),
            "min_step_logits_cosine_similarity": min(
                item["cosine_similarity"] for item in logits_checks
            ),
            "cpu_buffers_pinned": cache.all_cpu_buffers_pinned(),
            "cache_growth_cat_calls": cache.cache_growth_cat_calls,
            "cache_allocations": cache.allocations,
            "expected_cache_allocations": expected_allocations,
            "stagekv_gpu_used_cache_bytes": cache.used_gpu_bytes(),
            "stagekv_cpu_used_cache_bytes": cache.used_cpu_bytes(),
            "stagekv_gpu_allocated_cache_bytes": cache.allocated_gpu_bytes(),
            "stagekv_cpu_allocated_cache_bytes": cache.allocated_cpu_bytes(),
            "expected_gpu_cache_bytes": expected_gpu_bytes,
            "expected_cpu_cache_bytes": expected_cpu_bytes,
            "placement_pinning_and_capacity_correct": placement_and_capacity_correct,
            "expected_attention_calls": expected_attention_calls,
            "observed_attention_calls": patch.attention_calls,
            "expected_cpu_to_gpu_group_transfers": expected_transfer_calls,
            "observed_cpu_to_gpu_group_transfers": cache.cpu_to_gpu_group_transfers,
            "non_blocking_h2d_tensor_copies": cache.non_blocking_h2d_calls,
            "gpu_to_cpu_append_bytes": cache.gpu_to_cpu_bytes,
            "cpu_to_gpu_group_bytes": cache.cpu_to_gpu_bytes,
            "transfer_counts_correct": transfer_counts_correct,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "stagekv_static_pinned_elapsed_seconds": candidate["elapsed_seconds"],
            "behavioral_checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "note": (
                "Correctness and placement only. CPU buffers are pinned and H2D "
                "copies use non_blocking=True, but there is no separate CUDA "
                "transfer stream or compute/transfer overlap in this prototype."
            ),
        }
        reports.append(report)
        rows.append(report.copy())
        del candidate, cache, patch

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "stagekv_pinned_residency.json"
    csv_path = results_dir / "stagekv_pinned_residency.csv"
    document = {
        "model": args.model_path,
        "performance_claim_enabled": False,
        "results": reports,
    }
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"saved={json_path}")
    print(f"saved={csv_path}")
    if not all(item["status"] == "PASS_BEHAVIORAL_EQUIVALENCE" for item in reports):
        raise RuntimeError("At least one static pinned residency case failed")
    print("stagekv_pinned_residency=PASS_ALL_R_VALUES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
