"""Day-12 correctness test for one-layer-ahead StageKV prefetch.

During decode, layer L consumes a persistent staging slot that was filled
while layer L-1 was computing.  Before layer L launches its grouped SDPA,
the H2D stream starts copying the historical CPU KV for layer L+1.  Two
layer-sized slots are reused in ping-pong order and protected by CUDA events.

This is a correctness and scheduling test.  Its single-run timings are only
diagnostic; a warmed repeated benchmark is required for a performance claim.
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

from stagekv_bidirectional_async_correctness import (
    BidirectionalAsyncPatch,
    DeferredAsyncResidentCache,
    expected_historical_h2d_bytes,
)
from stagekv_cpu_g2_correctness import (
    compare_logit_distributions,
    compare_tensors,
    run_autoregressive,
    validate_structure,
)
from stagekv_pinned_residency_correctness import KV_GROUP_SIZE


MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR = "/root/stagekv/results/day12_cross_layer_prefetch"
DEFAULT_RESIDENT_HEADS = (1, 2)


@dataclass
class LayerPrefetchHandle:
    layer_idx: int
    slot_idx: int
    historical_length: int
    key: torch.Tensor
    value: torch.Tensor
    ready_event: torch.cuda.Event


class CrossLayerPrefetchPatch(BidirectionalAsyncPatch):
    """Prefetch an entire layer's offloaded KV one transformer layer ahead."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        cache: DeferredAsyncResidentCache,
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
        # Replace Day-10 group-sized resources with two whole-offloaded-layer slots.
        self.staging_key = []
        self.staging_value = []
        self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.consumed_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.slot_in_use = [False, False]
        self.staging_allocations = 0
        self.staging_allocated_bytes = 0
        self.pending: dict[int, LayerPrefetchHandle] = {}

        self.layer_prefetch_calls = 0
        self.lookahead_prefetch_calls = 0
        self.layer0_fallback_prefetch_calls = 0
        self.compute_wait_for_layer_ready_calls = 0
        self.layer_ready_event_records = 0
        self.layer_consumed_event_records = 0
        self.layer_slot_reuse_waits = 0
        self.layer_h2d_tensor_copies = 0
        self.layer_h2d_bytes = 0
        self.layer_prefetch_hits = 0
        self.layer_prefetch_misses = 0
        self.reusable_h2d_event_count = 4

    @staticmethod
    def _bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _ensure_layer_resources(self, reference: torch.Tensor) -> None:
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
            self.cache.cpu_heads,
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
            self.staging_allocated_bytes += self._bytes(key_slot)
            self.staging_allocated_bytes += self._bytes(value_slot)

    def _schedule_layer_prefetch(
        self,
        *,
        layer_idx: int,
        historical_length: int,
        reference: torch.Tensor,
        source_gate: torch.cuda.Event,
        lookahead: bool,
    ) -> None:
        if historical_length <= 0 or self.cache.cpu_heads == 0:
            return
        if layer_idx in self.pending:
            raise RuntimeError(f"layer {layer_idx} already has a pending prefetch")
        self._ensure_layer_resources(reference)
        assert self.transfer_stream is not None
        slot_idx = layer_idx % 2
        if self.slot_in_use[slot_idx]:
            self.transfer_stream.wait_event(self.consumed_events[slot_idx])
            self.layer_slot_reuse_waits += 1

        # source_gate is recorded before layer-L SDPA.  Waiting on it protects
        # earlier resident-cache writes without serializing against layer-L SDPA.
        self.transfer_stream.wait_event(source_gate)
        self.cache.enqueue_cpu_ready_wait(layer_idx, self.transfer_stream)

        cpu_key = self.cache.cpu_key_cache[layer_idx]
        cpu_value = self.cache.cpu_value_cache[layer_idx]
        assert cpu_key is not None and cpu_value is not None
        source_key = cpu_key[:, :, :historical_length]
        source_value = cpu_value[:, :, :historical_length]
        destination_key = self.staging_key[slot_idx][:, :, :historical_length]
        destination_value = self.staging_value[slot_idx][:, :, :historical_length]
        with torch.cuda.stream(self.transfer_stream):
            timing_start = timing_end = None
            if self.cache.transfer_event_timing:
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record(self.transfer_stream)
            destination_key.copy_(source_key, non_blocking=True)
            destination_value.copy_(source_value, non_blocking=True)
            if timing_end is not None:
                timing_end.record(self.transfer_stream)
            self.ready_events[slot_idx].record(self.transfer_stream)

        if timing_start is not None and timing_end is not None:
            self.h2d_transfer_timing_events.append((timing_start, timing_end))

        copied_bytes = self._bytes(destination_key) + self._bytes(destination_value)
        self.cache.cpu_to_gpu_group_transfers += 1
        self.cache.non_blocking_h2d_calls += 2
        self.cache.cpu_to_gpu_bytes += copied_bytes
        self.layer_prefetch_calls += 1
        self.layer_h2d_tensor_copies += 2
        self.layer_h2d_bytes += copied_bytes
        self.layer_ready_event_records += 1
        if lookahead:
            self.lookahead_prefetch_calls += 1
        else:
            self.layer0_fallback_prefetch_calls += 1
        self.slot_in_use[slot_idx] = True
        self.pending[layer_idx] = LayerPrefetchHandle(
            layer_idx=layer_idx,
            slot_idx=slot_idx,
            historical_length=historical_length,
            key=destination_key,
            value=destination_value,
            ready_event=self.ready_events[slot_idx],
        )

    def _take_layer_prefetch(
        self,
        *,
        layer_idx: int,
        historical_length: int,
        reference: torch.Tensor,
        source_gate: torch.cuda.Event,
    ) -> LayerPrefetchHandle:
        handle = self.pending.pop(layer_idx, None)
        if handle is None:
            self.layer_prefetch_misses += 1
            self._schedule_layer_prefetch(
                layer_idx=layer_idx,
                historical_length=historical_length,
                reference=reference,
                source_gate=source_gate,
                lookahead=False,
            )
            handle = self.pending.pop(layer_idx)
        else:
            self.layer_prefetch_hits += 1
        if handle.historical_length != historical_length:
            raise RuntimeError(
                f"prefetch length mismatch for layer {layer_idx}: "
                f"{handle.historical_length} != {historical_length}"
            )
        torch.cuda.current_stream(reference.device).wait_event(handle.ready_event)
        self.compute_wait_for_layer_ready_calls += 1
        return handle

    def _mark_layer_consumed(
        self, handle: LayerPrefetchHandle, compute_stream: torch.cuda.Stream
    ) -> None:
        self.consumed_events[handle.slot_idx].record(compute_stream)
        self.layer_consumed_event_records += 1

    def _historical_group(
        self,
        *,
        handle: LayerPrefetchHandle,
        layer_idx: int,
        start_head: int,
        end_head: int,
        historical_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        resident_end = min(end_head, self.cache.resident_heads)
        if start_head < resident_end:
            gpu_key = self.cache.gpu_key_cache[layer_idx]
            gpu_value = self.cache.gpu_value_cache[layer_idx]
            assert gpu_key is not None and gpu_value is not None
            key_parts.append(gpu_key[:, start_head:resident_end, :historical_length])
            value_parts.append(gpu_value[:, start_head:resident_end, :historical_length])
            self.direct_resident_decode_group_calls += 1

        cpu_start = max(start_head, self.cache.resident_heads)
        if cpu_start < end_head:
            offset_start = cpu_start - self.cache.resident_heads
            offset_end = end_head - self.cache.resident_heads
            key_parts.append(handle.key[:, offset_start:offset_end])
            value_parts.append(handle.value[:, offset_start:offset_end])
        if len(key_parts) == 1:
            return key_parts[0], value_parts[0]
        return torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1)

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
    ) -> tuple[torch.Tensor, None, DeferredAsyncResidentCache]:
        del cache_position, kwargs
        if output_attentions:
            raise RuntimeError("Cross-layer prototype supports output_attentions=False")
        if not use_cache or past_key_value is not self.cache:
            raise RuntimeError("CrossLayerPrefetchPatch requires its own cache")

        batch_size, query_length, _ = hidden_states.size()
        historical_length = self.cache.get_seq_length(layer_idx)
        is_prefill = historical_length == 0
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
            cos, sin = module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        compute_stream = torch.cuda.current_stream(query_states.device)
        source_gate, event_slot = self.cache.record_source_ready(
            layer_idx, compute_stream, query_states.device
        )
        handle: LayerPrefetchHandle | None = None
        if not is_prefill:
            handle = self._take_layer_prefetch(
                layer_idx=layer_idx,
                historical_length=historical_length,
                reference=query_states,
                source_gate=source_gate,
            )
            next_layer = layer_idx + 1
            if next_layer < len(self.modules):
                next_length = self.cache.get_seq_length(next_layer)
                self._schedule_layer_prefetch(
                    layer_idx=next_layer,
                    historical_length=next_length,
                    reference=query_states,
                    source_gate=source_gate,
                    lookahead=True,
                )

        outputs: list[torch.Tensor] = []
        for kv_start in range(0, self.kv_heads, self.kv_group_size):
            kv_end = kv_start + self.kv_group_size
            query_start = kv_start * self.query_heads_per_kv
            query_end = kv_end * self.query_heads_per_kv
            if is_prefill:
                key_group = key_states[:, kv_start:kv_end]
                value_group = value_states[:, kv_start:kv_end]
                self.prefill_direct_group_calls += 1
            else:
                assert handle is not None
                historical_key, historical_value = self._historical_group(
                    handle=handle,
                    layer_idx=layer_idx,
                    start_head=kv_start,
                    end_head=kv_end,
                    historical_length=historical_length,
                )
                key_group = torch.cat(
                    (historical_key, key_states[:, kv_start:kv_end]), dim=-2
                )
                value_group = torch.cat(
                    (historical_value, value_states[:, kv_start:kv_end]), dim=-2
                )
                self.decode_cached_group_calls += 1

            self.fresh_gpu_kv_group_uses += 1
            key_group = key_group.repeat_interleave(self.query_heads_per_kv, dim=1)
            value_group = value_group.repeat_interleave(self.query_heads_per_kv, dim=1)
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

        if handle is not None:
            self._mark_layer_consumed(handle, compute_stream)
        attn_output = torch.cat(outputs, dim=1)
        self.cache.append_deferred(
            layer_idx,
            key_states,
            value_states,
            source_ready_event=source_gate,
            event_slot=event_slot,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, module.hidden_size
        )
        attn_output = module.o_proj(attn_output)
        self.attention_calls += 1
        if is_prefill:
            self.prefill_attention_calls += 1
        else:
            self.decode_attention_calls += 1
        return attn_output, None, self.cache


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--resident-heads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-total-variation", type=float, default=1e-2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.sequence_length < 1 or args.decode_tokens < 2:
        raise ValueError("sequence-length must be positive and decode-tokens >= 2")
    if any(value not in DEFAULT_RESIDENT_HEADS for value in args.resident_heads):
        raise ValueError("this first cross-layer test supports resident heads 1 and 2")

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // query_heads)
    max_cache_len = args.sequence_length + args.decode_tokens - 1
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()

    print(f"model={args.model_path}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"layers={layers} kv_heads={kv_heads} sequence_length={args.sequence_length} "
        f"decode_tokens={args.decode_tokens} resident_heads={args.resident_heads}",
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

    print("running=standard", flush=True)
    standard = run_autoregressive(model, input_ids, args.decode_tokens)
    total_cache_bytes = 2 * layers * kv_heads * max_cache_len * head_dim * element_size
    expected_attention_calls = layers * args.decode_tokens
    expected_decode_calls = layers * (args.decode_tokens - 1)
    expected_prefill_calls = layers
    expected_group_calls = expected_attention_calls * (kv_heads // KV_GROUP_SIZE)
    rows: list[dict[str, Any]] = []

    for resident_heads in args.resident_heads:
        torch.cuda.empty_cache()
        old_cache = DeferredAsyncResidentCache(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        old_patch = BidirectionalAsyncPatch(
            model,
            old_cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=day10_bidirectional_r{resident_heads}", flush=True)
        old = run_autoregressive(
            model, input_ids, args.decode_tokens, cpu_patch=old_patch
        )
        torch.cuda.synchronize()
        old_elapsed = old["elapsed_seconds"]
        del old, old_patch, old_cache
        torch.cuda.empty_cache()

        cache = DeferredAsyncResidentCache(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        patch = CrossLayerPrefetchPatch(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=cross_layer_prefetch_r{resident_heads}", flush=True)
        candidate = run_autoregressive(
            model, input_ids, args.decode_tokens, cpu_patch=patch
        )
        torch.cuda.synchronize()

        logits_checks = [
            compare_tensors(a, b, rtol=args.rtol, atol=args.atol)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        distributions = [
            compare_logit_distributions(a, b)
            for a, b in zip(standard["step_logits"], candidate["step_logits"])
        ]
        same_sequence = standard["generated_ids"] == candidate["generated_ids"]
        all_top1 = all(item["top1_equal"] for item in distributions)
        max_tv = max(item["total_variation"] for item in distributions)
        min_top10 = min(item["top10_overlap"] for item in distributions)

        offloaded_heads = kv_heads - resident_heads
        expected_gpu_bytes = total_cache_bytes * resident_heads // kv_heads
        expected_cpu_bytes = total_cache_bytes - expected_gpu_bytes
        expected_h2d_bytes = expected_historical_h2d_bytes(
            layers=layers,
            offloaded_heads=offloaded_heads,
            sequence_length=args.sequence_length,
            decode_tokens=args.decode_tokens,
            head_dim=head_dim,
            element_size=element_size,
        )
        expected_prefetches = expected_decode_calls
        expected_lookahead = (layers - 1) * (args.decode_tokens - 1)
        expected_fallback = args.decode_tokens - 1
        expected_slot_waits = max(expected_prefetches - 2, 0)
        expected_d2h_calls = expected_attention_calls
        expected_staging_bytes = (
            2
            * 2
            * offloaded_heads
            * max_cache_len
            * head_dim
            * element_size
        )

        cache_correct = (
            cache.placement_correct()
            and cache.all_cpu_buffers_pinned()
            and cache.used_gpu_bytes() == expected_gpu_bytes
            and cache.used_cpu_bytes() == expected_cpu_bytes
            and all(length == max_cache_len for length in cache.lengths)
        )
        d2h_correct = (
            cache.async_d2h_append_calls == expected_d2h_calls
            and cache.non_blocking_d2h_tensor_copies == expected_d2h_calls * 2
            and cache.blocking_d2h_tensor_copies == 0
            and cache.source_ready_event_records == expected_d2h_calls
            and cache.d2h_source_wait_calls == expected_d2h_calls
            and cache.d2h_ready_event_records == expected_d2h_calls
            and cache.gpu_to_cpu_bytes == expected_cpu_bytes
        )
        prefetch_correct = (
            patch.layer_prefetch_calls == expected_prefetches
            and patch.lookahead_prefetch_calls == expected_lookahead
            and patch.layer0_fallback_prefetch_calls == expected_fallback
            and patch.layer_prefetch_hits == expected_lookahead
            and patch.layer_prefetch_misses == expected_fallback
            and patch.compute_wait_for_layer_ready_calls == expected_prefetches
            and patch.layer_ready_event_records == expected_prefetches
            and patch.layer_consumed_event_records == expected_prefetches
            and patch.layer_slot_reuse_waits == expected_slot_waits
            and patch.layer_h2d_tensor_copies == expected_prefetches * 2
            and patch.layer_h2d_bytes == expected_h2d_bytes
            and cache.cpu_to_gpu_group_transfers == expected_prefetches
            and cache.cpu_to_gpu_bytes == expected_h2d_bytes
            and cache.non_blocking_h2d_calls == expected_prefetches * 2
            and cache.h2d_wait_for_cpu_ready_calls == expected_prefetches
            and not patch.pending
            and patch.staging_allocations == 4
            and patch.staging_allocated_bytes == expected_staging_bytes
        )
        attention_correct = (
            patch.attention_calls == expected_attention_calls
            and patch.prefill_attention_calls == expected_prefill_calls
            and patch.decode_attention_calls == expected_decode_calls
            and patch.group_calls == expected_group_calls
        )
        stream_correct = (
            cache.d2h_stream_handle is not None
            and patch.transfer_stream_handle is not None
            and patch.compute_stream_handle is not None
            and len(
                {
                    cache.d2h_stream_handle,
                    patch.transfer_stream_handle,
                    patch.compute_stream_handle,
                }
            )
            == 3
        )
        checks = {
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1,
            "probability_total_variation_within_threshold": max_tv
            <= args.max_total_variation,
            "top10_overlap_within_threshold": min_top10 >= 0.9,
            "cache_correct": cache_correct,
            "d2h_correct": d2h_correct,
            "cross_layer_prefetch_correct": prefetch_correct,
            "attention_correct": attention_correct,
            "streams_distinct": stream_correct,
        }
        status = "PASS_BEHAVIORAL_EQUIVALENCE" if all(checks.values()) else "FAIL"
        row = {
            "status": status,
            "resident_kv_heads_r": resident_heads,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1,
            "max_probability_total_variation": max_tv,
            "min_top10_overlap": min_top10,
            "max_step_logits_relative_l2_error": max(
                item["relative_l2_error"] for item in logits_checks
            ),
            "stagekv_gpu_cache_bytes": cache.used_gpu_bytes(),
            "stagekv_cpu_cache_bytes": cache.used_cpu_bytes(),
            "expected_layer_prefetches": expected_prefetches,
            "observed_layer_prefetches": patch.layer_prefetch_calls,
            "expected_lookahead_prefetches": expected_lookahead,
            "observed_lookahead_prefetches": patch.lookahead_prefetch_calls,
            "expected_layer0_fallbacks": expected_fallback,
            "observed_layer0_fallbacks": patch.layer0_fallback_prefetch_calls,
            "expected_prefetch_hits": expected_lookahead,
            "observed_prefetch_hits": patch.layer_prefetch_hits,
            "expected_prefetch_misses": expected_fallback,
            "observed_prefetch_misses": patch.layer_prefetch_misses,
            "expected_slot_reuse_waits": expected_slot_waits,
            "observed_slot_reuse_waits": patch.layer_slot_reuse_waits,
            "expected_h2d_bytes": expected_h2d_bytes,
            "observed_h2d_bytes": patch.layer_h2d_bytes,
            "h2d_tensor_copies": patch.layer_h2d_tensor_copies,
            "async_d2h_append_calls": cache.async_d2h_append_calls,
            "non_blocking_d2h_tensor_copies": cache.non_blocking_d2h_tensor_copies,
            "blocking_d2h_tensor_copies": cache.blocking_d2h_tensor_copies,
            "staging_gpu_bytes": patch.staging_allocated_bytes,
            "compute_stream_handle": patch.compute_stream_handle,
            "h2d_stream_handle": patch.transfer_stream_handle,
            "d2h_stream_handle": cache.d2h_stream_handle,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "day10_elapsed_seconds": old_elapsed,
            "cross_layer_elapsed_seconds": candidate["elapsed_seconds"],
            "diagnostic_speedup_over_day10": old_elapsed
            / candidate["elapsed_seconds"],
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }
        rows.append(row)
        print(
            f"r={resident_heads} status={status} day10={old_elapsed:.3f}s "
            f"cross_layer={candidate['elapsed_seconds']:.3f}s "
            f"speedup={row['diagnostic_speedup_over_day10']:.3f}x "
            f"failed={row['failed_checks']}",
            flush=True,
        )
        del candidate, patch, cache

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "stagekv_cross_layer_prefetch.json"
    csv_path = results_dir / "stagekv_cross_layer_prefetch.csv"
    document = {
        "model": args.model_path,
        "performance_claim_enabled": False,
        "optimization": "prefetch layer L+1 while layer L grouped SDPA executes",
        "results": rows,
    }
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(csv_path, rows)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"saved={json_path}")
    print(f"saved={csv_path}")
    if not all(row["status"] == "PASS_BEHAVIORAL_EQUIVALENCE" for row in rows):
        raise RuntimeError("At least one cross-layer prefetch case failed")
    print("stagekv_cross_layer_prefetch=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
