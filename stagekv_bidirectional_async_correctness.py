"""Day-10 StageKV bidirectional asynchronous correctness sweep.

This version removes the most expensive redundant path found after Day 9:

    fresh GPU K/V -> blocking D2H -> immediate H2D -> attention

During decode, attention now reads only the historical prefix from the hybrid
cache and concatenates the current token's fresh GPU K/V directly.  After the
attention work is queued, the fresh offloaded heads are written to pinned CPU
memory on a dedicated D2H stream.  The next decode step waits on a reusable
per-layer event before reading that CPU prefix.

The H2D double buffer also reuses ready/consumed CUDA events.  When all KV
heads are resident (r=4), the patch uses one full-head SDPA call instead of the
two grouped calls used by offloaded configurations.

This remains a correctness-first Python prototype.  Run this test before
adding the new method to the calibrated Day-9 benchmark.
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
    CPUHeadOffloadPatch,
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


DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/stagekv/results/day10_bidirectional_async"


@dataclass
class ReusablePrefetchHandle:
    slot_index: int
    key: torch.Tensor
    value: torch.Tensor
    ready_event: torch.cuda.Event


class DeferredAsyncResidentCache(StaticPinnedResidentCache):
    """Static hybrid cache with non-blocking deferred D2H appends."""

    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        resident_heads: int,
        max_cache_len: int,
        transfer_event_timing: bool = False,
    ) -> None:
        super().__init__(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        self.d2h_stream: torch.cuda.Stream | None = None
        self.d2h_stream_handle: int | None = None
        self.transfer_event_timing = transfer_event_timing
        self.d2h_transfer_timing_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self.source_ready_events: list[list[torch.cuda.Event] | None] = [
            None for _ in range(layers)
        ]
        self.cpu_ready_event_pool: list[list[torch.cuda.Event] | None] = [
            None for _ in range(layers)
        ]
        self.latest_cpu_ready_event: list[torch.cuda.Event | None] = [
            None for _ in range(layers)
        ]
        self.event_generations = [0] * layers

        self.async_d2h_append_calls = 0
        self.non_blocking_d2h_tensor_copies = 0
        self.d2h_source_wait_calls = 0
        self.d2h_ready_event_records = 0
        self.source_ready_event_records = 0
        self.h2d_wait_for_cpu_ready_calls = 0
        self.reusable_d2h_event_count = 0
        self.blocking_d2h_tensor_copies = 0

    def d2h_event_total_ms(self) -> float:
        """Return aggregate D2H stream time after a full device sync."""
        if not self.transfer_event_timing:
            return 0.0
        return sum(
            float(start.elapsed_time(end))
            for start, end in self.d2h_transfer_timing_events
        )

    def _ensure_d2h_resources(self, layer_idx: int, device: torch.device) -> None:
        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream(device=device)
            self.d2h_stream_handle = int(self.d2h_stream.cuda_stream)
        if self.source_ready_events[layer_idx] is None:
            self.source_ready_events[layer_idx] = [
                torch.cuda.Event(),
                torch.cuda.Event(),
            ]
            self.cpu_ready_event_pool[layer_idx] = [
                torch.cuda.Event(),
                torch.cuda.Event(),
            ]
            self.reusable_d2h_event_count += 4

    def record_source_ready(
        self,
        layer_idx: int,
        compute_stream: torch.cuda.Stream,
        device: torch.device,
    ) -> tuple[torch.cuda.Event, int]:
        """Record that fresh K/V and prior resident writes are GPU-ready."""
        self._ensure_d2h_resources(layer_idx, device)
        events = self.source_ready_events[layer_idx]
        assert events is not None
        slot = self.event_generations[layer_idx] % 2
        event = events[slot]
        event.record(compute_stream)
        self.source_ready_event_records += 1
        return event, slot

    def enqueue_cpu_ready_wait(
        self,
        layer_idx: int,
        transfer_stream: torch.cuda.Stream,
    ) -> None:
        """Protect the historical CPU prefix before H2D reads it."""
        event = self.latest_cpu_ready_event[layer_idx]
        if event is None:
            raise RuntimeError(
                f"Layer {layer_idx} CPU cache has no completed/queued D2H append"
            )
        transfer_stream.wait_event(event)
        self.h2d_wait_for_cpu_ready_calls += 1

    def append_deferred(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        source_ready_event: torch.cuda.Event | None,
        event_slot: int | None,
    ) -> None:
        """Persist fresh K/V without blocking the Python/compute stream."""
        if key.device.type != "cuda" or value.device.type != "cuda":
            raise RuntimeError("fresh K/V states must be on CUDA")
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
            if source_ready_event is None or event_slot is None:
                raise RuntimeError("offloaded heads require a source-ready event")
            self._ensure_d2h_resources(layer_idx, key.device)
            assert self.d2h_stream is not None
            ready_pool = self.cpu_ready_event_pool[layer_idx]
            assert ready_pool is not None
            ready_event = ready_pool[event_slot]
            cpu_key = self.cpu_key_cache[layer_idx]
            cpu_value = self.cpu_value_cache[layer_idx]
            assert cpu_key is not None and cpu_value is not None
            source_key = key[:, self.resident_heads :]
            source_value = value[:, self.resident_heads :]
            destination_key = cpu_key[:, :, start:end]
            destination_value = cpu_value[:, :, start:end]

            self.d2h_stream.wait_event(source_ready_event)
            self.d2h_source_wait_calls += 1
            with torch.cuda.stream(self.d2h_stream):
                timing_start = timing_end = None
                if self.transfer_event_timing:
                    timing_start = torch.cuda.Event(enable_timing=True)
                    timing_end = torch.cuda.Event(enable_timing=True)
                    timing_start.record(self.d2h_stream)
                destination_key.copy_(source_key, non_blocking=True)
                destination_value.copy_(source_value, non_blocking=True)
                if timing_end is not None:
                    timing_end.record(self.d2h_stream)
                ready_event.record(self.d2h_stream)
            if timing_start is not None and timing_end is not None:
                self.d2h_transfer_timing_events.append((timing_start, timing_end))
            source_key.record_stream(self.d2h_stream)
            source_value.record_stream(self.d2h_stream)
            self.latest_cpu_ready_event[layer_idx] = ready_event
            self.async_d2h_append_calls += 1
            self.non_blocking_d2h_tensor_copies += 2
            self.d2h_ready_event_records += 1
            self.gpu_to_cpu_bytes += self._bytes(source_key) + self._bytes(source_value)

        self.lengths[layer_idx] = end
        self.event_generations[layer_idx] += 1
        self.gpu_to_cpu_append_calls += 1

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        raise RuntimeError(
            "DeferredAsyncResidentCache requires append_deferred(); "
            "blocking append is disabled"
        )


class BidirectionalAsyncPatch(CPUHeadOffloadPatch):
    """Fresh-current-KV attention with asynchronous D2H and H2D streams."""

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
        self.cache: DeferredAsyncResidentCache
        self.transfer_stream: torch.cuda.Stream | None = None
        self.compute_stream_handle: int | None = None
        self.transfer_stream_handle: int | None = None
        self.staging_key: list[torch.Tensor] = []
        self.staging_value: list[torch.Tensor] = []
        self.ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.consumed_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.slot_in_use = [False, False]
        self.staging_allocations = 0
        self.staging_allocated_bytes = 0

        self.prefill_attention_calls = 0
        self.decode_attention_calls = 0
        self.prefill_direct_group_calls = 0
        self.decode_cached_group_calls = 0
        self.full_resident_fast_path_calls = 0
        self.async_prefetch_group_calls = 0
        self.compute_wait_for_ready_calls = 0
        self.transfer_wait_for_slot_reuse_calls = 0
        self.h2d_source_wait_calls = 0
        self.ready_event_records = 0
        self.consumed_event_records = 0
        self.overlap_opportunities = 0
        self.direct_resident_decode_group_calls = 0
        self.staged_resident_d2d_tensor_copies = 0
        self.staged_resident_d2d_bytes = 0
        self.fresh_gpu_kv_group_uses = 0
        self.decode_d2h_to_h2d_roundtrips = 0
        self.reusable_h2d_event_count = 4
        self.h2d_transfer_timing_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []

    def h2d_event_total_ms(self) -> float:
        """Return aggregate H2D stream time after a full device sync."""
        if not self.cache.transfer_event_timing:
            return 0.0
        return sum(
            float(start.elapsed_time(end))
            for start, end in self.h2d_transfer_timing_events
        )

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _ensure_h2d_resources(self, reference: torch.Tensor) -> None:
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

    def _prefetch_historical_group(
        self,
        *,
        layer_idx: int,
        group_index: int,
        start_head: int,
        end_head: int,
        historical_length: int,
        reference: torch.Tensor,
    ) -> ReusablePrefetchHandle:
        if not self._group_needs_cpu(start_head, end_head):
            raise RuntimeError("resident-only groups must not enter H2D prefetch")
        self._ensure_h2d_resources(reference)
        assert self.transfer_stream is not None
        slot_index = group_index % 2
        if self.slot_in_use[slot_index]:
            self.transfer_stream.wait_event(self.consumed_events[slot_index])
            self.transfer_wait_for_slot_reuse_calls += 1

        group_heads = end_head - start_head
        key_destination = self.staging_key[slot_index][
            :, :group_heads, :historical_length
        ]
        value_destination = self.staging_value[slot_index][
            :, :group_heads, :historical_length
        ]
        resident_end = min(end_head, self.cache.resident_heads)
        destination_offset = 0
        with torch.cuda.stream(self.transfer_stream):
            if start_head < resident_end:
                gpu_key = self.cache.gpu_key_cache[layer_idx]
                gpu_value = self.cache.gpu_value_cache[layer_idx]
                assert gpu_key is not None and gpu_value is not None
                resident_count = resident_end - start_head
                key_destination[:, :resident_count].copy_(
                    gpu_key[:, start_head:resident_end, :historical_length],
                    non_blocking=True,
                )
                value_destination[:, :resident_count].copy_(
                    gpu_value[:, start_head:resident_end, :historical_length],
                    non_blocking=True,
                )
                self.staged_resident_d2d_tensor_copies += 2
                self.staged_resident_d2d_bytes += (
                    self._tensor_bytes(key_destination[:, :resident_count])
                    + self._tensor_bytes(value_destination[:, :resident_count])
                )
                destination_offset = resident_count

            cpu_start = max(start_head, self.cache.resident_heads)
            cpu_key = self.cache.cpu_key_cache[layer_idx]
            cpu_value = self.cache.cpu_value_cache[layer_idx]
            assert cpu_key is not None and cpu_value is not None
            cpu_offset_start = cpu_start - self.cache.resident_heads
            cpu_offset_end = end_head - self.cache.resident_heads
            cpu_count = cpu_offset_end - cpu_offset_start
            cpu_key_view = cpu_key[
                :, cpu_offset_start:cpu_offset_end, :historical_length
            ]
            cpu_value_view = cpu_value[
                :, cpu_offset_start:cpu_offset_end, :historical_length
            ]
            timing_start = timing_end = None
            if self.cache.transfer_event_timing:
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record(self.transfer_stream)
            key_destination[
                :, destination_offset : destination_offset + cpu_count
            ].copy_(cpu_key_view, non_blocking=True)
            value_destination[
                :, destination_offset : destination_offset + cpu_count
            ].copy_(cpu_value_view, non_blocking=True)
            if timing_end is not None:
                timing_end.record(self.transfer_stream)
            self.ready_events[slot_index].record(self.transfer_stream)

        if timing_start is not None and timing_end is not None:
            self.h2d_transfer_timing_events.append((timing_start, timing_end))

        self.cache.cpu_to_gpu_group_transfers += 1
        self.cache.non_blocking_h2d_calls += 2
        self.cache.cpu_to_gpu_bytes += (
            self._tensor_bytes(cpu_key_view) + self._tensor_bytes(cpu_value_view)
        )
        self.async_prefetch_group_calls += 1
        self.ready_event_records += 1
        return ReusablePrefetchHandle(
            slot_index=slot_index,
            key=key_destination,
            value=value_destination,
            ready_event=self.ready_events[slot_index],
        )

    def _mark_consumed(
        self,
        handle: ReusablePrefetchHandle,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        event = self.consumed_events[handle.slot_index]
        event.record(compute_stream)
        self.slot_in_use[handle.slot_index] = True
        self.consumed_event_records += 1

    def _full_resident_attention(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        historical_length: int,
        attention_mask: torch.Tensor | None,
        query_length: int,
    ) -> torch.Tensor:
        if historical_length:
            gpu_key = self.cache.gpu_key_cache[layer_idx]
            gpu_value = self.cache.gpu_value_cache[layer_idx]
            assert gpu_key is not None and gpu_value is not None
            key_all = torch.cat(
                (gpu_key[:, :, :historical_length], key_states), dim=-2
            )
            value_all = torch.cat(
                (gpu_value[:, :, :historical_length], value_states), dim=-2
            )
        else:
            key_all = key_states
            value_all = value_states
        key_all = key_all.repeat_interleave(self.query_heads_per_kv, dim=1)
        value_all = value_all.repeat_interleave(self.query_heads_per_kv, dim=1)
        output = functional.scaled_dot_product_attention(
            query_states,
            key_all,
            value_all,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and query_length > 1,
        )
        self.full_resident_fast_path_calls += 1
        self.fresh_gpu_kv_group_uses += 1
        return output

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
            raise RuntimeError("This prototype supports output_attentions=False only")
        if not use_cache or past_key_value is not self.cache:
            raise RuntimeError(
                "BidirectionalAsyncPatch requires its cache and use_cache=True"
            )

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

        compute_stream = torch.cuda.current_stream(query_states.device)
        source_ready_event: torch.cuda.Event | None = None
        event_slot: int | None = None
        if self.cache.cpu_heads:
            source_ready_event, event_slot = self.cache.record_source_ready(
                layer_idx,
                compute_stream,
                query_states.device,
            )

        if self.cache.resident_heads == self.kv_heads:
            attn_output = self._full_resident_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                layer_idx=layer_idx,
                historical_length=historical_length,
                attention_mask=attention_mask,
                query_length=query_length,
            )
            if is_prefill:
                self.prefill_attention_calls += 1
            else:
                self.decode_attention_calls += 1
        else:
            outputs: list[torch.Tensor] = []
            groups = [
                (index, start, start + self.kv_group_size)
                for index, start in enumerate(
                    range(0, self.kv_heads, self.kv_group_size)
                )
            ]
            handles: dict[int, ReusablePrefetchHandle] = {}

            if not is_prefill:
                self._ensure_h2d_resources(query_states)
                assert self.transfer_stream is not None
                assert source_ready_event is not None
                self.cache.enqueue_cpu_ready_wait(layer_idx, self.transfer_stream)
                self.transfer_stream.wait_event(source_ready_event)
                self.h2d_source_wait_calls += 1

                def schedule(group_index: int) -> None:
                    _, start_head, end_head = groups[group_index]
                    handles[group_index] = self._prefetch_historical_group(
                        layer_idx=layer_idx,
                        group_index=group_index,
                        start_head=start_head,
                        end_head=end_head,
                        historical_length=historical_length,
                        reference=query_states,
                    )

                if self._group_needs_cpu(groups[0][1], groups[0][2]):
                    schedule(0)
                elif len(groups) > 1 and self._group_needs_cpu(
                    groups[1][1], groups[1][2]
                ):
                    schedule(1)
                    self.overlap_opportunities += 1

            for group_index, kv_start, kv_end in groups:
                query_start = kv_start * self.query_heads_per_kv
                query_end = kv_end * self.query_heads_per_kv
                if is_prefill:
                    key_group = key_states[:, kv_start:kv_end]
                    value_group = value_states[:, kv_start:kv_end]
                    self.prefill_direct_group_calls += 1
                else:
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
                        historical_key = gpu_key[
                            :, kv_start:kv_end, :historical_length
                        ]
                        historical_value = gpu_value[
                            :, kv_start:kv_end, :historical_length
                        ]
                        self.direct_resident_decode_group_calls += 1
                    else:
                        historical_key = handle.key
                        historical_value = handle.value
                    key_group = torch.cat(
                        (historical_key, key_states[:, kv_start:kv_end]), dim=-2
                    )
                    value_group = torch.cat(
                        (historical_value, value_states[:, kv_start:kv_end]), dim=-2
                    )
                    self.decode_cached_group_calls += 1
                    if handle is not None:
                        self._mark_consumed(handle, compute_stream)

                self.fresh_gpu_kv_group_uses += 1
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

            attn_output = torch.cat(outputs, dim=1)
            if is_prefill:
                self.prefill_attention_calls += 1
            else:
                self.decode_attention_calls += 1

        self.cache.append_deferred(
            layer_idx,
            key_states,
            value_states,
            source_ready_event=source_ready_event,
            event_slot=event_slot,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, module.hidden_size
        )
        attn_output = module.o_proj(attn_output)
        self.attention_calls += 1
        if len(self.call_shapes) < 6:
            self.call_shapes.append(
                {
                    "query_shape": list(query_states.shape),
                    "historical_cache_length": historical_length,
                    "fresh_kv_length": int(key_states.shape[-2]),
                    "final_cache_length": self.cache.get_seq_length(layer_idx),
                    "phase": "prefill" if is_prefill else "decode",
                    "fresh_kv_source": "gpu_direct",
                    "historical_kv_source": (
                        "none"
                        if is_prefill
                        else (
                            "gpu_resident_fast_path"
                            if self.cache.resident_heads == self.kv_heads
                            else "async_hybrid_cache"
                        )
                    ),
                }
            )
        return attn_output, None, self.cache


def expected_historical_h2d_bytes(
    *,
    layers: int,
    offloaded_heads: int,
    sequence_length: int,
    decode_tokens: int,
    head_dim: int,
    element_size: int,
) -> int:
    """Only historical K/V is read; fresh current-token K/V stays on GPU."""
    decode_steps = decode_tokens - 1
    summed_history_lengths = (
        decode_steps * sequence_length
        + decode_steps * (decode_steps - 1) // 2
    )
    return (
        2
        * layers
        * offloaded_heads
        * summed_history_lengths
        * head_dim
        * element_size
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=8)
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
        f"layers={layers} query_heads={query_heads} kv_heads={kv_heads} "
        f"head_dim={head_dim} sequence_length={args.sequence_length} "
        f"decode_tokens={args.decode_tokens} r={RESIDENT_VALUES}",
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
    total_cache_bytes = (
        2 * layers * kv_heads * max_cache_len * head_dim * element_size
    )
    expected_prefill_calls = layers
    expected_decode_calls = layers * (args.decode_tokens - 1)
    expected_attention_calls = expected_prefill_calls + expected_decode_calls
    groups_per_attention = kv_heads // KV_GROUP_SIZE
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for resident_heads in RESIDENT_VALUES:
        torch.cuda.empty_cache()
        cache = DeferredAsyncResidentCache(
            layers=layers,
            kv_heads=kv_heads,
            resident_heads=resident_heads,
            max_cache_len=max_cache_len,
        )
        patch = BidirectionalAsyncPatch(
            model,
            cache,
            query_heads=query_heads,
            kv_heads=kv_heads,
            kv_group_size=KV_GROUP_SIZE,
        )
        print(f"running=stagekv_bidirectional_async_r{resident_heads}", flush=True)
        candidate = run_autoregressive(
            model,
            input_ids,
            args.decode_tokens,
            cpu_patch=patch,
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
        same_sequence = standard["generated_ids"] == candidate["generated_ids"]
        all_top1_equal = all(item["top1_equal"] for item in distribution_checks)
        max_tv = max(item["total_variation"] for item in distribution_checks)
        min_top10_overlap = min(item["top10_overlap"] for item in distribution_checks)

        offloaded_heads = kv_heads - resident_heads
        expected_gpu_cache = total_cache_bytes * resident_heads // kv_heads
        expected_cpu_cache = total_cache_bytes - expected_gpu_cache
        expected_h2d_groups = (
            expected_decode_calls
            * expected_transfer_groups_per_attention(resident_heads)
        )
        expected_h2d_bytes = expected_historical_h2d_bytes(
            layers=layers,
            offloaded_heads=offloaded_heads,
            sequence_length=args.sequence_length,
            decode_tokens=args.decode_tokens,
            head_dim=head_dim,
            element_size=element_size,
        )
        expected_d2h_calls = expected_attention_calls if offloaded_heads else 0
        expected_d2h_bytes = expected_cpu_cache
        expected_cpu_ready_waits = expected_decode_calls if offloaded_heads else 0
        expected_group_calls = (
            0 if resident_heads == kv_heads else expected_attention_calls * groups_per_attention
        )
        expected_fast_path_calls = (
            expected_attention_calls if resident_heads == kv_heads else 0
        )
        expected_staging_allocations = 0 if resident_heads == kv_heads else 4
        expected_staging_bytes = (
            0
            if resident_heads == kv_heads
            else 2
            * 2
            * KV_GROUP_SIZE
            * max_cache_len
            * head_dim
            * element_size
        )
        expected_cache_allocations = (
            layers * 2 * int(resident_heads > 0)
            + layers * 2 * int(resident_heads < kv_heads)
        )
        expected_decode_group_calls = (
            0 if resident_heads == kv_heads else expected_decode_calls * groups_per_attention
        )
        expected_prefill_group_calls = (
            0 if resident_heads == kv_heads else expected_prefill_calls * groups_per_attention
        )
        expected_direct_resident_groups = (
            expected_decode_group_calls - expected_h2d_groups
        )
        slots_used = (
            0
            if expected_h2d_groups == 0
            else (1 if resident_heads == 2 else 2)
        )
        expected_slot_reuse_waits = max(expected_h2d_groups - slots_used, 0)
        expected_fresh_gpu_uses = (
            expected_attention_calls
            if resident_heads == kv_heads
            else expected_attention_calls * groups_per_attention
        )

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
        d2h_path_correct = (
            cache.async_d2h_append_calls == expected_d2h_calls
            and cache.non_blocking_d2h_tensor_copies == expected_d2h_calls * 2
            and cache.blocking_d2h_tensor_copies == 0
            and cache.source_ready_event_records == expected_d2h_calls
            and cache.d2h_source_wait_calls == expected_d2h_calls
            and cache.d2h_ready_event_records == expected_d2h_calls
            and cache.gpu_to_cpu_bytes == expected_d2h_bytes
        )
        h2d_path_correct = (
            cache.cpu_to_gpu_group_transfers == expected_h2d_groups
            and cache.non_blocking_h2d_calls == expected_h2d_groups * 2
            and cache.cpu_to_gpu_bytes == expected_h2d_bytes
            and cache.h2d_wait_for_cpu_ready_calls == expected_cpu_ready_waits
            and patch.async_prefetch_group_calls == expected_h2d_groups
            and patch.compute_wait_for_ready_calls == expected_h2d_groups
            and patch.ready_event_records == expected_h2d_groups
            and patch.consumed_event_records == expected_h2d_groups
            and patch.h2d_source_wait_calls == expected_cpu_ready_waits
            and patch.transfer_wait_for_slot_reuse_calls
            == expected_slot_reuse_waits
            and patch.direct_resident_decode_group_calls
            == expected_direct_resident_groups
            and patch.staging_allocations == expected_staging_allocations
            and patch.staging_allocated_bytes == expected_staging_bytes
        )
        attention_path_correct = (
            patch.attention_calls == expected_attention_calls
            and patch.prefill_attention_calls == expected_prefill_calls
            and patch.decode_attention_calls == expected_decode_calls
            and patch.group_calls == expected_group_calls
            and patch.prefill_direct_group_calls == expected_prefill_group_calls
            and patch.decode_cached_group_calls == expected_decode_group_calls
            and patch.full_resident_fast_path_calls == expected_fast_path_calls
            and patch.fresh_gpu_kv_group_uses == expected_fresh_gpu_uses
            and patch.overlap_opportunities
            == (expected_decode_calls if offloaded_heads else 0)
            and patch.decode_d2h_to_h2d_roundtrips == 0
        )
        stream_path_correct = (
            offloaded_heads == 0
            or (
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
                and cache.reusable_d2h_event_count == layers * 4
                and patch.reusable_h2d_event_count == 4
            )
        )
        checks = {
            "same_generated_sequence": same_sequence,
            "all_step_top1_equal": all_top1_equal,
            "probability_total_variation_within_threshold": (
                max_tv <= args.max_total_variation
            ),
            "top10_overlap_within_threshold": min_top10_overlap >= 0.9,
            "cache_correct": cache_correct,
            "d2h_path_correct": d2h_path_correct,
            "h2d_path_correct": h2d_path_correct,
            "attention_path_correct": attention_path_correct,
            "stream_path_correct": stream_path_correct,
        }
        status = "PASS_BEHAVIORAL_EQUIVALENCE" if all(checks.values()) else "FAIL"
        report = {
            "status": status,
            "resident_kv_heads_r": resident_heads,
            "kv_group_size_g": KV_GROUP_SIZE,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "fresh_decode_kv_used_directly_on_gpu": True,
            "blocking_d2h_removed": cache.blocking_d2h_tensor_copies == 0,
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
            "staging_gpu_bytes": patch.staging_allocated_bytes,
            "expected_async_d2h_append_calls": expected_d2h_calls,
            "observed_async_d2h_append_calls": cache.async_d2h_append_calls,
            "non_blocking_d2h_tensor_copies": cache.non_blocking_d2h_tensor_copies,
            "blocking_d2h_tensor_copies": cache.blocking_d2h_tensor_copies,
            "expected_d2h_bytes": expected_d2h_bytes,
            "observed_d2h_bytes": cache.gpu_to_cpu_bytes,
            "expected_h2d_group_transfers": expected_h2d_groups,
            "observed_h2d_group_transfers": cache.cpu_to_gpu_group_transfers,
            "expected_h2d_bytes": expected_h2d_bytes,
            "observed_h2d_bytes": cache.cpu_to_gpu_bytes,
            "cpu_ready_waits": cache.h2d_wait_for_cpu_ready_calls,
            "expected_slot_reuse_waits": expected_slot_reuse_waits,
            "observed_slot_reuse_waits": patch.transfer_wait_for_slot_reuse_calls,
            "reusable_d2h_event_count": cache.reusable_d2h_event_count,
            "reusable_h2d_event_count": patch.reusable_h2d_event_count,
            "staging_allocated_bytes": patch.staging_allocated_bytes,
            "full_resident_fast_path_calls": patch.full_resident_fast_path_calls,
            "group_attention_calls": patch.group_calls,
            "compute_stream_handle": patch.compute_stream_handle,
            "h2d_stream_handle": patch.transfer_stream_handle,
            "d2h_stream_handle": cache.d2h_stream_handle,
            "streams_distinct": stream_path_correct,
            "standard_elapsed_seconds": standard["elapsed_seconds"],
            "stagekv_elapsed_seconds": candidate["elapsed_seconds"],
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "note": (
                "Historical K/V is prefetched from the hybrid cache. Fresh current-token "
                "K/V remains on GPU for the current attention and is appended to pinned "
                "CPU asynchronously for the next token. Timing remains diagnostic until "
                "the calibrated benchmark is updated."
            ),
        }
        reports.append(report)
        rows.append(report.copy())
        print(
            f"r={resident_heads} status={status} "
            f"standard={standard['elapsed_seconds']:.3f}s "
            f"stagekv={candidate['elapsed_seconds']:.3f}s "
            f"failed={report['failed_checks']}",
            flush=True,
        )
        del candidate, patch, cache

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "stagekv_bidirectional_async.json"
    csv_path = results_dir / "stagekv_bidirectional_async.csv"
    document = {
        "model": args.model_path,
        "performance_claim_enabled": False,
        "optimization": (
            "fresh decode K/V direct on GPU plus deferred asynchronous D2H append"
        ),
        "results": reports,
    }
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(csv_path, rows)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"saved={json_path}")
    print(f"saved={csv_path}")
    if not all(item["status"] == "PASS_BEHAVIORAL_EQUIVALENCE" for item in reports):
        raise RuntimeError("At least one bidirectional async residency case failed")
    print("stagekv_bidirectional_async=PASS_ALL_R_VALUES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
