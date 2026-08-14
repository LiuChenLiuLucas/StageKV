"""Day-8 StageKV asynchronous phase-aware correctness sweep.

The initial prefill uses fresh GPU K/V directly. During decode, offloaded KV
groups are prefetched from pinned CPU memory on a dedicated CUDA stream into
two reusable GPU staging slots. CUDA events protect slot reuse and make the
compute stream wait only for the group it is about to consume. The next group
is scheduled before the current group's SDPA call, creating an explicit
transfer/compute overlap opportunity.

This script validates behavior and the asynchronous execution path. Event
timings are diagnostic; robust performance claims require warmup and repeated
measurements in a later benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from stagekv_cpu_g2_correctness import (
    compare_logit_distributions,
    compare_tensors,
    run_autoregressive,
    validate_structure,
)
from stagekv_pinned_residency_correctness import (
    KV_GROUP_SIZE,
    RESIDENT_VALUES,
    StaticPinnedResidentCache,
    expected_transfer_groups_per_attention,
)
from stagekv_phase_aware_correctness import (
    PhaseAwareOffloadPatch,
    expected_h2d_bytes,
)


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day8_async_phase_aware"


@dataclass
class PrefetchHandle:
    slot_index: int
    group_index: int
    key: torch.Tensor
    value: torch.Tensor
    ready_event: torch.cuda.Event


class AsyncDoubleBufferPatch(PhaseAwareOffloadPatch):
    """Phase-aware attention with a transfer stream and two staging slots."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        cache: StaticPinnedResidentCache,
        *,
        query_heads: int,
        kv_heads: int,
        kv_group_size: int,
    ) -> None:
        super().__init__(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=kv_group_size,
        )
        self.cache: StaticPinnedResidentCache
        self.transfer_stream: torch.cuda.Stream | None = None
        self.compute_stream_handle: int | None = None
        self.transfer_stream_handle: int | None = None
        self.staging_key: list[torch.Tensor] = []
        self.staging_value: list[torch.Tensor] = []
        self.slot_free_events: list[torch.cuda.Event | None] = [None, None]
        self.staging_allocations = 0
        self.staging_allocated_bytes = 0

        self.async_prefetch_group_calls = 0
        self.compute_wait_for_ready_calls = 0
        self.transfer_wait_for_source_calls = 0
        self.transfer_wait_for_slot_reuse_calls = 0
        self.consumed_event_records = 0
        self.overlap_opportunities = 0
        self.direct_resident_decode_group_calls = 0
        self.staged_resident_d2d_tensor_copies = 0
        self.staged_resident_d2d_bytes = 0
        self.cuda_event_timing_pairs = 0
        self._transfer_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _ensure_async_resources(self, reference: torch.Tensor) -> None:
        if self.transfer_stream is None:
            self.transfer_stream = torch.cuda.Stream(device=reference.device)
            self.compute_stream_handle = int(
                torch.cuda.current_stream(reference.device).cuda_stream
            )
            self.transfer_stream_handle = int(self.transfer_stream.cuda_stream)
        if self.staging_key:
            return
        batch_size, _, _, head_dim = reference.shape
        shape = (
            batch_size,
            self.kv_group_size,
            self.cache.max_cache_len,
            head_dim,
        )
        for _ in range(2):
            key_slot = torch.empty(shape, dtype=reference.dtype, device=reference.device)
            value_slot = torch.empty(
                shape, dtype=reference.dtype, device=reference.device
            )
            self.staging_key.append(key_slot)
            self.staging_value.append(value_slot)
            self.staging_allocations += 2
            self.staging_allocated_bytes += self._tensor_bytes(key_slot)
            self.staging_allocated_bytes += self._tensor_bytes(value_slot)

    def _group_needs_cpu(self, start_head: int, end_head: int) -> bool:
        return max(start_head, self.cache.resident_heads) < end_head

    def _prefetch_group(
        self,
        *,
        layer_idx: int,
        group_index: int,
        start_head: int,
        end_head: int,
        length: int,
        source_ready_event: torch.cuda.Event,
        reference: torch.Tensor,
    ) -> PrefetchHandle:
        if not self._group_needs_cpu(start_head, end_head):
            raise RuntimeError("resident-only groups must not enter async prefetch")
        self._ensure_async_resources(reference)
        assert self.transfer_stream is not None
        slot_index = group_index % 2
        free_event = self.slot_free_events[slot_index]
        if free_event is not None:
            self.transfer_stream.wait_event(free_event)
            self.transfer_wait_for_slot_reuse_calls += 1
        self.transfer_stream.wait_event(source_ready_event)
        self.transfer_wait_for_source_calls += 1

        group_heads = end_head - start_head
        key_destination = self.staging_key[slot_index][:, :group_heads, :length]
        value_destination = self.staging_value[slot_index][:, :group_heads, :length]
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.transfer_stream):
            transfer_start.record(self.transfer_stream)
            resident_end = min(end_head, self.cache.resident_heads)
            destination_offset = 0
            if start_head < resident_end:
                gpu_key = self.cache.gpu_key_cache[layer_idx]
                gpu_value = self.cache.gpu_value_cache[layer_idx]
                assert gpu_key is not None and gpu_value is not None
                resident_count = resident_end - start_head
                key_destination[:, :resident_count].copy_(
                    gpu_key[:, start_head:resident_end, :length], non_blocking=True
                )
                value_destination[:, :resident_count].copy_(
                    gpu_value[:, start_head:resident_end, :length], non_blocking=True
                )
                self.staged_resident_d2d_tensor_copies += 2
                self.staged_resident_d2d_bytes += (
                    self._tensor_bytes(key_destination[:, :resident_count])
                    + self._tensor_bytes(value_destination[:, :resident_count])
                )
                destination_offset = resident_count

            cpu_start = max(start_head, self.cache.resident_heads)
            if cpu_start < end_head:
                cpu_key = self.cache.cpu_key_cache[layer_idx]
                cpu_value = self.cache.cpu_value_cache[layer_idx]
                assert cpu_key is not None and cpu_value is not None
                cpu_offset_start = cpu_start - self.cache.resident_heads
                cpu_offset_end = end_head - self.cache.resident_heads
                cpu_count = cpu_offset_end - cpu_offset_start
                cpu_key_view = cpu_key[
                    :, cpu_offset_start:cpu_offset_end, :length
                ]
                cpu_value_view = cpu_value[
                    :, cpu_offset_start:cpu_offset_end, :length
                ]
                key_destination[
                    :, destination_offset : destination_offset + cpu_count
                ].copy_(cpu_key_view, non_blocking=True)
                value_destination[
                    :, destination_offset : destination_offset + cpu_count
                ].copy_(cpu_value_view, non_blocking=True)
                self.cache.cpu_to_gpu_group_transfers += 1
                self.cache.non_blocking_h2d_calls += 2
                self.cache.cpu_to_gpu_bytes += (
                    self._tensor_bytes(cpu_key_view) + self._tensor_bytes(cpu_value_view)
                )
            transfer_end.record(self.transfer_stream)

        self.async_prefetch_group_calls += 1
        self.cuda_event_timing_pairs += 1
        self._transfer_timing_events.append((transfer_start, transfer_end))
        return PrefetchHandle(
            slot_index=slot_index,
            group_index=group_index,
            key=key_destination,
            value=value_destination,
            ready_event=transfer_end,
        )

    def _mark_consumed(
        self,
        handle: PrefetchHandle,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        consumed = torch.cuda.Event()
        consumed.record(compute_stream)
        self.slot_free_events[handle.slot_index] = consumed
        self.consumed_event_records += 1

    def transfer_event_total_ms(self) -> float:
        return float(
            sum(start.elapsed_time(end) for start, end in self._transfer_timing_events)
        )

    def _attention_forward(
        self,
        module: Any,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, StaticPinnedResidentCache]:
        del cache_position, kwargs
        if output_attentions:
            raise RuntimeError("This prototype supports output_attentions=False only")
        if not use_cache or past_key_value is not self.cache:
            raise RuntimeError(
                "AsyncDoubleBufferPatch requires its cache and use_cache=True"
            )

        batch_size, query_length, _ = hidden_states.size()
        previous_cache_length = self.cache.get_seq_length(layer_idx)
        is_prefill = previous_cache_length == 0
        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)
        query_states = query_states.view(
            batch_size, query_length, module.num_heads, module.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, query_length, module.num_key_value_heads, module.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, query_length, module.num_key_value_heads, module.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            if not hasattr(module, "rotary_emb"):
                raise RuntimeError(
                    "This Transformers version must pass position_embeddings"
                )
            cos, sin = module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        self.cache.append(layer_idx, key_states, value_states)

        outputs: list[torch.Tensor] = []
        if is_prefill:
            for group_index, kv_start in enumerate(
                range(0, self.kv_heads, self.kv_group_size)
            ):
                kv_end = kv_start + self.kv_group_size
                query_start = kv_start * self.query_heads_per_kv
                query_end = kv_end * self.query_heads_per_kv
                key_group = key_states[:, kv_start:kv_end].repeat_interleave(
                    self.query_heads_per_kv, dim=1
                )
                value_group = value_states[:, kv_start:kv_end].repeat_interleave(
                    self.query_heads_per_kv, dim=1
                )
                group_mask = self._group_mask(attention_mask, query_start, query_end)
                outputs.append(
                    functional.scaled_dot_product_attention(
                        query_states[:, query_start:query_end],
                        key_group,
                        value_group,
                        attn_mask=group_mask,
                        dropout_p=0.0,
                        is_causal=group_mask is None and query_length > 1,
                    )
                )
                self.group_calls += 1
                self.prefill_direct_group_calls += 1
            self.prefill_attention_calls += 1
        else:
            compute_stream = torch.cuda.current_stream(query_states.device)
            has_offloaded_heads = self.cache.resident_heads < self.kv_heads
            source_ready_event: torch.cuda.Event | None = None
            if has_offloaded_heads:
                self._ensure_async_resources(query_states)
                assert self.transfer_stream is not None
                source_ready_event = torch.cuda.Event()
                source_ready_event.record(compute_stream)
            length = self.cache.get_seq_length(layer_idx)
            groups = [
                (index, start, start + self.kv_group_size)
                for index, start in enumerate(
                    range(0, self.kv_heads, self.kv_group_size)
                )
            ]
            handles: dict[int, PrefetchHandle] = {}

            def schedule(group_index: int) -> None:
                if source_ready_event is None:
                    raise RuntimeError("async prefetch scheduled without offloaded heads")
                _, start_head, end_head = groups[group_index]
                handles[group_index] = self._prefetch_group(
                    layer_idx=layer_idx,
                    group_index=group_index,
                    start_head=start_head,
                    end_head=end_head,
                    length=length,
                    source_ready_event=source_ready_event,
                    reference=query_states,
                )

            if self._group_needs_cpu(groups[0][1], groups[0][2]):
                schedule(0)
            elif len(groups) > 1 and self._group_needs_cpu(groups[1][1], groups[1][2]):
                schedule(1)
                self.overlap_opportunities += 1

            for group_index, kv_start, kv_end in groups:
                handle = handles.get(group_index)
                if handle is not None:
                    compute_stream.wait_event(handle.ready_event)
                    self.compute_wait_for_ready_calls += 1

                next_index = group_index + 1
                if (
                    next_index < len(groups)
                    and next_index not in handles
                    and self._group_needs_cpu(
                        groups[next_index][1], groups[next_index][2]
                    )
                ):
                    schedule(next_index)
                    self.overlap_opportunities += 1

                if handle is None:
                    gpu_key = self.cache.gpu_key_cache[layer_idx]
                    gpu_value = self.cache.gpu_value_cache[layer_idx]
                    assert gpu_key is not None and gpu_value is not None
                    key_group = gpu_key[:, kv_start:kv_end, :length]
                    value_group = gpu_value[:, kv_start:kv_end, :length]
                    self.direct_resident_decode_group_calls += 1
                else:
                    key_group = handle.key
                    value_group = handle.value

                query_start = kv_start * self.query_heads_per_kv
                query_end = kv_end * self.query_heads_per_kv
                key_group = key_group.repeat_interleave(
                    self.query_heads_per_kv, dim=1
                )
                value_group = value_group.repeat_interleave(
                    self.query_heads_per_kv, dim=1
                )
                group_mask = self._group_mask(attention_mask, query_start, query_end)
                outputs.append(
                    functional.scaled_dot_product_attention(
                        query_states[:, query_start:query_end],
                        key_group,
                        value_group,
                        attn_mask=group_mask,
                        dropout_p=0.0,
                        is_causal=group_mask is None and query_length > 1,
                    )
                )
                self.group_calls += 1
                self.decode_cached_group_calls += 1
                if handle is not None:
                    self._mark_consumed(handle, compute_stream)
            self.decode_attention_calls += 1
            self.decode_cpu_to_gpu_group_transfers += sum(
                1 for handle in handles.values() if handle is not None
            )

        attn_output = torch.cat(outputs, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, module.hidden_size
        )
        attn_output = module.o_proj(attn_output)
        self.attention_calls += 1
        if len(self.call_shapes) < 4:
            self.call_shapes.append(
                {
                    "query_shape": list(query_states.shape),
                    "raw_kv_cache_shape": self.cache.cache_shape(layer_idx),
                    "previous_cache_length": previous_cache_length,
                    "phase": "prefill" if is_prefill else "decode",
                    "kv_source": (
                        "fresh_gpu" if is_prefill else "async_double_buffer"
                    ),
                }
            )
        return attn_output, None, self.cache


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
    element_size = 2
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
    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} g={KV_GROUP_SIZE} r={RESIDENT_VALUES}",
        flush=True,
    )
    print("running=standard_gpu_dynamic_cache", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)

    total_cache_bytes = 2 * layers * kv_heads * max_cache_len * head_dim * element_size
    expected_prefill_calls = layers
    expected_decode_calls = layers * (args.decode_tokens - 1)
    expected_prefill_groups = expected_prefill_calls * (kv_heads // KV_GROUP_SIZE)
    expected_decode_groups = expected_decode_calls * (kv_heads // KV_GROUP_SIZE)
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
        patch = AsyncDoubleBufferPatch(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=stagekv_async_r{resident_heads}_g2", flush=True)
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
        expected_gpu_cache = total_cache_bytes * resident_heads // kv_heads
        expected_cpu_cache = total_cache_bytes - expected_gpu_cache
        expected_async_prefetches = (
            expected_decode_calls
            * expected_transfer_groups_per_attention(resident_heads)
        )
        expected_direct_decode_groups = expected_decode_groups - expected_async_prefetches
        expected_overlap_opportunities = (
            expected_decode_calls if resident_heads < kv_heads else 0
        )
        slots_used = (
            0
            if expected_async_prefetches == 0
            else (1 if resident_heads == 2 else 2)
        )
        expected_slot_reuse_waits = max(expected_async_prefetches - slots_used, 0)
        expected_h2d_byte_count = expected_h2d_bytes(
            layers=layers,
            offloaded_heads=kv_heads - resident_heads,
            sequence_length=args.sequence_length,
            decode_tokens=args.decode_tokens,
            head_dim=head_dim,
            element_size=element_size,
        )
        expected_staging_allocations = 0 if resident_heads == kv_heads else 4
        expected_staging_bytes = (
            0
            if resident_heads == kv_heads
            else 2
            * 2
            * 1
            * KV_GROUP_SIZE
            * max_cache_len
            * head_dim
            * element_size
        )
        expected_resident_d2d_copies = (
            expected_decode_calls * 2 if resident_heads == 1 else 0
        )
        expected_cache_allocations = (
            layers * 2 * int(resident_heads > 0)
            + layers * 2 * int(resident_heads < kv_heads)
        )

        same_sequence = standard["generated_ids"] == candidate["generated_ids"]
        all_top1_equal = all(item["top1_equal"] for item in distribution_checks)
        cache_correct = (
            cache.placement_correct()
            and cache.all_cpu_buffers_pinned()
            and cache.used_gpu_bytes() == expected_gpu_cache
            and cache.used_cpu_bytes() == expected_cpu_cache
            and cache.allocated_gpu_bytes() == expected_gpu_cache
            and cache.allocated_cpu_bytes() == expected_cpu_cache
            and all(length == max_cache_len for length in cache.lengths)
            and cache.allocations == expected_cache_allocations
            and cache.cache_growth_cat_calls == 0
        )
        async_path_correct = (
            patch.prefill_attention_calls == expected_prefill_calls
            and patch.decode_attention_calls == expected_decode_calls
            and patch.prefill_direct_group_calls == expected_prefill_groups
            and patch.decode_cached_group_calls == expected_decode_groups
            and patch.prefill_cpu_to_gpu_group_transfers == 0
            and patch.async_prefetch_group_calls == expected_async_prefetches
            and patch.compute_wait_for_ready_calls == expected_async_prefetches
            and patch.transfer_wait_for_source_calls == expected_async_prefetches
            and patch.transfer_wait_for_slot_reuse_calls == expected_slot_reuse_waits
            and patch.consumed_event_records == expected_async_prefetches
            and patch.overlap_opportunities == expected_overlap_opportunities
            and patch.direct_resident_decode_group_calls == expected_direct_decode_groups
            and patch.cuda_event_timing_pairs == expected_async_prefetches
            and patch.staging_allocations == expected_staging_allocations
            and patch.staging_allocated_bytes == expected_staging_bytes
            and patch.staged_resident_d2d_tensor_copies == expected_resident_d2d_copies
            and (
                expected_async_prefetches == 0
                or patch.compute_stream_handle != patch.transfer_stream_handle
            )
        )
        transfer_correct = (
            cache.cpu_to_gpu_group_transfers == expected_async_prefetches
            and cache.non_blocking_h2d_calls == expected_async_prefetches * 2
            and cache.cpu_to_gpu_bytes == expected_h2d_byte_count
        )
        checks = {
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "probability_total_variation_within_threshold": (
                max_tv <= args.max_total_variation
            ),
            "top10_overlap_within_threshold": min_top10_overlap >= 0.9,
            "cache_correct": cache_correct,
            "async_path_correct": async_path_correct,
            "transfer_counts_and_bytes_correct": transfer_correct,
        }
        status = "PASS_BEHAVIORAL_EQUIVALENCE" if all(checks.values()) else "FAIL"
        transfer_event_total_ms = (
            patch.transfer_event_total_ms() if expected_async_prefetches else 0.0
        )
        report = {
            "status": status,
            "resident_kv_heads_r": resident_heads,
            "kv_group_size_g": KV_GROUP_SIZE,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "phase_aware_prefill_enabled": True,
            "dedicated_transfer_stream_enabled": expected_async_prefetches > 0,
            "double_buffer_enabled": expected_async_prefetches > 0,
            "compute_stream_handle": patch.compute_stream_handle,
            "transfer_stream_handle": patch.transfer_stream_handle,
            "streams_distinct": (
                expected_async_prefetches == 0
                or patch.compute_stream_handle != patch.transfer_stream_handle
            ),
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
            "stagekv_gpu_cache_bytes": cache.used_gpu_bytes(),
            "stagekv_cpu_cache_bytes": cache.used_cpu_bytes(),
            "staging_allocated_bytes": patch.staging_allocated_bytes,
            "cache_correct": cache_correct,
            "prefill_cpu_to_gpu_group_transfers": (
                patch.prefill_cpu_to_gpu_group_transfers
            ),
            "expected_async_prefetch_group_calls": expected_async_prefetches,
            "observed_async_prefetch_group_calls": patch.async_prefetch_group_calls,
            "compute_wait_for_ready_calls": patch.compute_wait_for_ready_calls,
            "transfer_wait_for_source_calls": patch.transfer_wait_for_source_calls,
            "expected_transfer_wait_for_slot_reuse_calls": expected_slot_reuse_waits,
            "observed_transfer_wait_for_slot_reuse_calls": (
                patch.transfer_wait_for_slot_reuse_calls
            ),
            "consumed_event_records": patch.consumed_event_records,
            "expected_overlap_opportunities": expected_overlap_opportunities,
            "observed_overlap_opportunities": patch.overlap_opportunities,
            "expected_direct_resident_decode_group_calls": expected_direct_decode_groups,
            "observed_direct_resident_decode_group_calls": (
                patch.direct_resident_decode_group_calls
            ),
            "cuda_event_timing_pairs": patch.cuda_event_timing_pairs,
            "transfer_event_total_ms": transfer_event_total_ms,
            "async_path_correct": async_path_correct,
            "expected_cpu_to_gpu_bytes": expected_h2d_byte_count,
            "observed_cpu_to_gpu_bytes": cache.cpu_to_gpu_bytes,
            "transfer_counts_and_bytes_correct": transfer_correct,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "stagekv_async_elapsed_seconds": candidate["elapsed_seconds"],
            "behavioral_checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "note": (
                "A dedicated transfer stream, two persistent staging slots, ready "
                "events, and consumed events are enabled. Counters prove the async "
                "path was scheduled. Event and wall-clock timings remain diagnostic "
                "until a warmed, repeated benchmark is run."
            ),
        }
        reports.append(report)
        rows.append(report.copy())
        del candidate, cache, patch

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "stagekv_async_phase_aware.json"
    csv_path = results_dir / "stagekv_async_phase_aware.csv"
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
        raise RuntimeError("At least one async phase-aware residency case failed")
    print("stagekv_async_phase_aware=PASS_ALL_R_VALUES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
