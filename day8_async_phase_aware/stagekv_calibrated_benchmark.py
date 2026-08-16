"""Day-9 calibrated benchmark for Standard, synchronous, and async StageKV.

This is a measurement script, not a new algorithm.  It compares a standard
GPU DynamicCache against the Day-7 phase-aware synchronous path and the Day-8
phase-aware asynchronous double-buffer path under one controlled protocol:

* identical deterministic input tokens for all methods;
* complete same-length warm-up rounds that are excluded from the results;
* at least five measured repetitions per method by default;
* 32 generated tokens by default, with prefill, TTFT, first decode step, and
  steady decode reported separately; and
* rotated execution order to avoid assigning CUDA/JIT warm-up to one method.

The script launches one isolated worker process per context length.  It writes
all individual trials before aggregation, so paper tables can always be traced
back to raw measurements.  The StageKV variants retain the behavioral checks
from Days 7 and 8: every measured trial must match the Standard greedy token
sequence from the same round.

Required companion files in the same directory:
  stagekv_cpu_g2_correctness.py
  stagekv_pinned_residency_correctness.py
  stagekv_phase_aware_correctness.py
  stagekv_async_phase_aware_correctness.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from stagekv_async_phase_aware_correctness import AsyncDoubleBufferPatch
from stagekv_cpu_g2_correctness import dynamic_cache_bytes, validate_structure
from stagekv_phase_aware_correctness import PhaseAwareOffloadPatch
from stagekv_pinned_residency_correctness import KV_GROUP_SIZE, StaticPinnedResidentCache


MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR = "/root/stagekv/results/day9_calibrated_benchmark"
DEFAULT_LENGTHS = (4096,)
DEFAULT_RESIDENT_HEADS = (0, 1, 2, 4)
MARKER = "__STAGEKV_DAY9_RESULT__="


@dataclass(frozen=True)
class MethodSpec:
    """One reproducible benchmark configuration."""

    family: str
    resident_heads: int | None = None

    @property
    def method(self) -> str:
        if self.family == "standard":
            return "standard"
        assert self.resident_heads is not None
        return f"stagekv_{self.family}_r{self.resident_heads}"

    @property
    def async_enabled(self) -> bool:
        return self.family == "async"


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def percentile(values: list[float], quantile: float) -> float:
    """Linearly interpolated percentile, including for short trial lists."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def method_specs(
    resident_heads: list[int], stagekv_modes: list[str]
) -> list[MethodSpec]:
    specs = [MethodSpec("standard")]
    for family in stagekv_modes:
        specs.extend(MethodSpec(family, resident) for resident in resident_heads)
    return specs


def rotated(items: list[MethodSpec], round_index: int) -> list[MethodSpec]:
    """Rotate by two places so positions change even with five repetitions."""
    shift = (round_index * 2) % len(items)
    return items[shift:] + items[:shift]


def new_runtime(
    model: AutoModelForCausalLM,
    spec: MethodSpec,
    *,
    layers: int,
    query_heads: int,
    kv_heads: int,
    max_cache_len: int,
) -> tuple[
    DynamicCache | StaticPinnedResidentCache,
    PhaseAwareOffloadPatch | AsyncDoubleBufferPatch | None,
]:
    if spec.family == "standard":
        return DynamicCache(), None

    assert spec.resident_heads is not None
    cache = StaticPinnedResidentCache(
        layers=layers,
        kv_heads=kv_heads,
        resident_heads=spec.resident_heads,
        max_cache_len=max_cache_len,
    )
    patch_type = (
        AsyncDoubleBufferPatch if spec.family == "async" else PhaseAwareOffloadPatch
    )
    patch = patch_type(
        model,
        cache,
        query_heads=query_heads,
        kv_heads=kv_heads,
        kv_group_size=KV_GROUP_SIZE,
    )
    return cache, patch


def elapsed_event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Events must be synchronized before this function is called."""
    return float(start.elapsed_time(end))


def timed_model_forward(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | StaticPinnedResidentCache,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Measure one forward plus greedy selection with wall and CUDA clocks."""
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = elapsed_event_ms(start_event, end_event)
    del output
    return next_token, logits, wall_ms, cuda_ms


def cache_statistics(
    cache: DynamicCache | StaticPinnedResidentCache,
    patch: PhaseAwareOffloadPatch | AsyncDoubleBufferPatch | None,
) -> dict[str, Any]:
    if isinstance(cache, DynamicCache):
        total = dynamic_cache_bytes(cache)
        return {
            "cache_gpu_bytes": total,
            "cache_cpu_bytes": 0,
            "cache_total_bytes": total,
            "staging_gpu_bytes": 0,
            "h2d_group_transfers": 0,
            "h2d_bytes": 0,
            "async_prefetch_groups": 0,
            "async_transfer_event_total_ms": 0.0,
            "dedicated_transfer_stream_enabled": False,
        }

    async_patch = patch if isinstance(patch, AsyncDoubleBufferPatch) else None
    return {
        "cache_gpu_bytes": cache.used_gpu_bytes(),
        "cache_cpu_bytes": cache.used_cpu_bytes(),
        "cache_total_bytes": cache.used_gpu_bytes() + cache.used_cpu_bytes(),
        "staging_gpu_bytes": 0 if async_patch is None else async_patch.staging_allocated_bytes,
        "h2d_group_transfers": cache.cpu_to_gpu_group_transfers,
        "h2d_bytes": cache.cpu_to_gpu_bytes,
        "async_prefetch_groups": (
            0 if async_patch is None else async_patch.async_prefetch_group_calls
        ),
        "async_transfer_event_total_ms": (
            0.0 if async_patch is None else async_patch.transfer_event_total_ms()
        ),
        "dedicated_transfer_stream_enabled": (
            False if async_patch is None else async_patch.transfer_stream is not None
        ),
    }


def run_once(
    model: AutoModelForCausalLM,
    spec: MethodSpec,
    input_ids: torch.Tensor,
    *,
    layers: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    decode_tokens: int,
) -> dict[str, Any]:
    """Run one complete prompt plus greedy decode with fresh cache state."""
    if decode_tokens < 3:
        raise ValueError("decode_tokens must be at least 3")

    max_cache_len = int(input_ids.shape[1]) + decode_tokens - 1
    cache, patch = new_runtime(
        model,
        spec,
        layers=layers,
        query_heads=query_heads,
        kv_heads=kv_heads,
        max_cache_len=max_cache_len,
    )
    context: Iterator[Any] = patch.install() if patch is not None else nullcontext()
    process = psutil.Process()
    cpu_before = process.memory_info().rss
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    generated_ids: list[int] = []
    decode_wall_ms: list[float] = []
    decode_cuda_ms: list[float] = []
    decode_h2d_event_ms: list[float] = []
    try:
        attention_mask = torch.ones_like(input_ids)
        with context:
            next_token, logits, prefill_wall_ms, prefill_cuda_ms = timed_model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache=cache,
            )
            generated_ids.append(int(next_token.item()))
            del logits
            prefill_peak_allocated = torch.cuda.max_memory_allocated()
            prefill_peak_reserved = torch.cuda.max_memory_reserved()

            torch.cuda.reset_peak_memory_stats()
            for decode_step in range(1, decode_tokens):
                attention_mask = torch.ones(
                    (1, input_ids.shape[1] + decode_step),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                transfer_before = (
                    patch.transfer_event_total_ms()
                    if isinstance(patch, AsyncDoubleBufferPatch)
                    else 0.0
                )
                next_token, logits, wall_ms, cuda_ms = timed_model_forward(
                    model,
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    cache=cache,
                )
                transfer_after = (
                    patch.transfer_event_total_ms()
                    if isinstance(patch, AsyncDoubleBufferPatch)
                    else 0.0
                )
                generated_ids.append(int(next_token.item()))
                decode_wall_ms.append(wall_ms)
                decode_cuda_ms.append(cuda_ms)
                decode_h2d_event_ms.append(transfer_after - transfer_before)
                del logits

            decode_peak_allocated = torch.cuda.max_memory_allocated()
            decode_peak_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.synchronize()
            stats = cache_statistics(cache, patch)
            cpu_after = process.memory_info().rss

        theoretical_cache_bytes = (
            2
            * layers
            * kv_heads
            * max_cache_len
            * head_dim
            # input_ids is int64, whereas Qwen's K/V cache is BF16.
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )
        steady_cuda_ms = decode_cuda_ms[1:]
        steady_wall_ms = decode_wall_ms[1:]
        return {
            "status": "ok",
            "generated_token_ids": json.dumps(generated_ids),
            "step_top1_ids": json.dumps(generated_ids),
            "prefill_wall_ms": prefill_wall_ms,
            "prefill_cuda_ms": prefill_cuda_ms,
            "time_to_first_token_wall_ms": prefill_wall_ms,
            "time_to_first_token_cuda_ms": prefill_cuda_ms,
            "prefill_tokens_per_second": input_ids.shape[1] / (prefill_cuda_ms / 1000.0),
            "decode_step_wall_ms": json.dumps(decode_wall_ms),
            "decode_step_cuda_ms": json.dumps(decode_cuda_ms),
            "decode_step_h2d_event_ms": json.dumps(decode_h2d_event_ms),
            "decode_first_step_wall_ms": decode_wall_ms[0],
            "decode_first_step_cuda_ms": decode_cuda_ms[0],
            "decode_steady_wall_mean_ms": statistics.mean(steady_wall_ms),
            "decode_steady_cuda_mean_ms": statistics.mean(steady_cuda_ms),
            "decode_steady_cuda_p50_ms": percentile(steady_cuda_ms, 0.50),
            "decode_steady_cuda_p95_ms": percentile(steady_cuda_ms, 0.95),
            "decode_steady_tokens_per_second": 1000.0 / statistics.mean(steady_cuda_ms),
            "prefill_peak_gpu_allocated_gib": gib(prefill_peak_allocated),
            "prefill_peak_gpu_reserved_gib": gib(prefill_peak_reserved),
            "decode_peak_gpu_allocated_gib": gib(decode_peak_allocated),
            "decode_peak_gpu_reserved_gib": gib(decode_peak_reserved),
            "cpu_rss_before_gib": gib(cpu_before),
            "cpu_rss_after_gib": gib(cpu_after),
            "cpu_rss_delta_gib": gib(cpu_after - cpu_before),
            "cache_gpu_gib": gib(stats["cache_gpu_bytes"]),
            "cache_cpu_gib": gib(stats["cache_cpu_bytes"]),
            "cache_total_gib": gib(stats["cache_total_bytes"]),
            "cache_ratio": stats["cache_total_bytes"] / theoretical_cache_bytes,
            "theoretical_cache_gib": gib(theoretical_cache_bytes),
            "staging_gpu_gib": gib(stats["staging_gpu_bytes"]),
            "h2d_group_transfers": stats["h2d_group_transfers"],
            "h2d_gib": gib(stats["h2d_bytes"]),
            "async_prefetch_groups": stats["async_prefetch_groups"],
            "async_h2d_event_total_ms": stats["async_transfer_event_total_ms"],
            "dedicated_transfer_stream_enabled": stats[
                "dedicated_transfer_stream_enabled"
            ],
            "generated_tokens": decode_tokens,
            "decode_forward_steps": decode_tokens - 1,
        }
    finally:
        del cache, patch
        gc.collect()
        torch.cuda.empty_cache()


def add_trial_metadata(
    row: dict[str, Any],
    *,
    spec: MethodSpec,
    trial: int,
    sequence_length: int,
    decode_tokens: int,
    round_order_position: int,
    is_warmup: bool,
) -> None:
    row.update(
        {
            "method": spec.method,
            "family": spec.family,
            "resident_kv_heads_r": "" if spec.resident_heads is None else spec.resident_heads,
            "kv_group_size_g": "" if spec.family == "standard" else KV_GROUP_SIZE,
            "sequence_length": sequence_length,
            "trial": trial,
            "is_warmup": is_warmup,
            "round_order_position": round_order_position,
            "decode_tokens": decode_tokens,
        }
    )


def set_behavioral_checks(rows: list[dict[str, Any]]) -> None:
    """Pair all StageKV variants with Standard from the same measured round."""
    standard = next(
        (row for row in rows if row["method"] == "standard" and row["status"] == "ok"),
        None,
    )
    for row in rows:
        if row["status"] != "ok":
            row["same_generated_sequence_as_standard"] = False
            row["all_step_top1_equal_as_standard"] = False
            continue
        if standard is None:
            row["same_generated_sequence_as_standard"] = False
            row["all_step_top1_equal_as_standard"] = False
            continue
        row["same_generated_sequence_as_standard"] = (
            row["generated_token_ids"] == standard["generated_token_ids"]
        )
        row["all_step_top1_equal_as_standard"] = (
            row["step_top1_ids"] == standard["step_top1_ids"]
        )


def worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    validate_structure(config)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // query_heads)
    specs = method_specs(args.resident_heads, args.stagekv_modes)

    print(f"loading_model={args.model_path}", flush=True)
    print(
        f"sequence_length={args.sequence_length} decode_tokens={args.decode_tokens} "
        f"warmup_repeats={args.warmup_repeats} repeats={args.repeats}",
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
    generator.manual_seed(args.seed + args.sequence_length)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        device="cuda",
    )

    def execute_round(trial: int, is_warmup: bool) -> list[dict[str, Any]]:
        offset = trial + (args.warmup_repeats if not is_warmup else 0)
        ordered_specs = rotated(specs, offset)
        rows: list[dict[str, Any]] = []
        for order_position, spec in enumerate(ordered_specs):
            print(
                f"running={'warmup' if is_warmup else 'trial'}={trial} "
                f"position={order_position} method={spec.method}",
                flush=True,
            )
            try:
                row = run_once(
                    model,
                    spec,
                    input_ids,
                    layers=layers,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    decode_tokens=args.decode_tokens,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                row = {"status": "oom", "error": str(exc).replace("\n", " ")[:1000]}
            add_trial_metadata(
                row,
                spec=spec,
                trial=trial,
                sequence_length=args.sequence_length,
                decode_tokens=args.decode_tokens,
                round_order_position=order_position,
                is_warmup=is_warmup,
            )
            rows.append(row)
        set_behavioral_checks(rows)
        return rows

    for warmup_round in range(1, args.warmup_repeats + 1):
        for row in execute_round(warmup_round, True):
            print(MARKER + json.dumps(row), flush=True)
    for trial in range(1, args.repeats + 1):
        rows = execute_round(trial, False)
        for row in rows:
            print(MARKER + json.dumps(row), flush=True)
            if row["status"] == "ok":
                print(
                    f"trial={trial} method={row['method']} "
                    f"prefill_cuda={row['prefill_cuda_ms']:.2f}ms "
                    f"first_decode_cuda={row['decode_first_step_cuda_ms']:.2f}ms "
                    f"steady_cuda={row['decode_steady_cuda_mean_ms']:.2f}ms/token "
                    f"match={row['same_generated_sequence_as_standard']}",
                    flush=True,
                )
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        generated_ids = json.loads(row["generated_token_ids"])
        output.append(
            {
                "method": row["method"],
                "family": row["family"],
                "resident_kv_heads_r": row["resident_kv_heads_r"],
                "sequence_length": row["sequence_length"],
                "trial": row["trial"],
                "generated_token_position": 1,
                "phase": "prefill_ttft",
                "token_id": generated_ids[0],
                "wall_ms": row["time_to_first_token_wall_ms"],
                "cuda_ms": row["time_to_first_token_cuda_ms"],
                "h2d_event_ms": 0.0,
            }
        )
        wall_steps = json.loads(row["decode_step_wall_ms"])
        cuda_steps = json.loads(row["decode_step_cuda_ms"])
        h2d_steps = json.loads(row["decode_step_h2d_event_ms"])
        for index, (wall_ms, cuda_ms, h2d_ms) in enumerate(
            zip(wall_steps, cuda_steps, h2d_steps), start=1
        ):
            output.append(
                {
                    "method": row["method"],
                    "family": row["family"],
                    "resident_kv_heads_r": row["resident_kv_heads_r"],
                    "sequence_length": row["sequence_length"],
                    "trial": row["trial"],
                    "generated_token_position": index + 1,
                    "phase": "decode_first" if index == 1 else "decode_steady",
                    "token_id": generated_ids[index],
                    "wall_ms": wall_ms,
                    "cuda_ms": cuda_ms,
                    "h2d_event_ms": h2d_ms,
                }
            )
    return output


SUMMARY_METRICS = (
    "prefill_wall_ms",
    "prefill_cuda_ms",
    "time_to_first_token_wall_ms",
    "time_to_first_token_cuda_ms",
    "prefill_tokens_per_second",
    "decode_first_step_wall_ms",
    "decode_first_step_cuda_ms",
    "decode_steady_wall_mean_ms",
    "decode_steady_cuda_mean_ms",
    "decode_steady_tokens_per_second",
    "prefill_peak_gpu_allocated_gib",
    "prefill_peak_gpu_reserved_gib",
    "decode_peak_gpu_allocated_gib",
    "decode_peak_gpu_reserved_gib",
    "cache_gpu_gib",
    "cache_cpu_gib",
    "cache_total_gib",
    "staging_gpu_gib",
    "h2d_gib",
    "async_h2d_event_total_ms",
)


def summarize(
    rows: list[dict[str, Any]], specs: list[MethodSpec]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    lengths = sorted({int(row["sequence_length"]) for row in rows})
    for length in lengths:
        for spec in specs:
            group = [
                row
                for row in rows
                if row["method"] == spec.method
                and int(row["sequence_length"]) == length
                and row["status"] == "ok"
            ]
            summary: dict[str, Any] = {
                "method": spec.method,
                "family": spec.family,
                "resident_kv_heads_r": "" if spec.resident_heads is None else spec.resident_heads,
                "sequence_length": length,
                "status": "ok" if group else "oom_or_missing",
                "successful_trials": len(group),
            }
            if not group:
                summaries.append(summary)
                continue
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            steady_cuda = [
                latency
                for row in group
                for latency in json.loads(row["decode_step_cuda_ms"])[1:]
            ]
            steady_wall = [
                latency
                for row in group
                for latency in json.loads(row["decode_step_wall_ms"])[1:]
            ]
            summary["decode_steady_cuda_p50_ms_pooled"] = percentile(steady_cuda, 0.50)
            summary["decode_steady_cuda_p95_ms_pooled"] = percentile(steady_cuda, 0.95)
            summary["decode_steady_wall_p50_ms_pooled"] = percentile(steady_wall, 0.50)
            summary["decode_steady_wall_p95_ms_pooled"] = percentile(steady_wall, 0.95)
            summary["all_generated_sequences_match_standard"] = all(
                bool(row["same_generated_sequence_as_standard"]) for row in group
            )
            summary["all_step_top1_match_standard"] = all(
                bool(row["all_step_top1_equal_as_standard"]) for row in group
            )
            summaries.append(summary)
    return summaries


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lengths = sorted({int(row["sequence_length"]) for row in summary})
    for length in lengths:
        standard = next(
            (
                row
                for row in summary
                if row["sequence_length"] == length
                and row["method"] == "standard"
                and row["status"] == "ok"
            ),
            None,
        )
        if standard is None:
            output.append({"comparison": "missing_standard", "sequence_length": length})
            continue
        for candidate in (
            row
            for row in summary
            if row["sequence_length"] == length
            and row["method"] != "standard"
            and row["status"] == "ok"
        ):
            output.append(
                {
                    "comparison": "candidate_vs_standard",
                    "sequence_length": length,
                    "candidate_method": candidate["method"],
                    "candidate_family": candidate["family"],
                    "resident_kv_heads_r": candidate["resident_kv_heads_r"],
                    "prefill_cuda_overhead_percent": (
                        candidate["prefill_cuda_ms_mean"]
                        / standard["prefill_cuda_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "ttft_cuda_overhead_percent": (
                        candidate["time_to_first_token_cuda_ms_mean"]
                        / standard["time_to_first_token_cuda_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "decode_first_cuda_overhead_percent": (
                        candidate["decode_first_step_cuda_ms_mean"]
                        / standard["decode_first_step_cuda_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "steady_decode_cuda_overhead_percent": (
                        candidate["decode_steady_cuda_mean_ms_mean"]
                        / standard["decode_steady_cuda_mean_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "steady_decode_cuda_speed_ratio": (
                        standard["decode_steady_cuda_mean_ms_mean"]
                        / candidate["decode_steady_cuda_mean_ms_mean"]
                    ),
                    "peak_decode_gpu_allocated_saving_gib": (
                        standard["decode_peak_gpu_allocated_gib_mean"]
                        - candidate["decode_peak_gpu_allocated_gib_mean"]
                    ),
                    "persistent_kv_gpu_saving_gib": (
                        standard["cache_gpu_gib_mean"] - candidate["cache_gpu_gib_mean"]
                    ),
                    "candidate_cpu_kv_gib": candidate["cache_cpu_gib_mean"],
                    "candidate_h2d_gib": candidate["h2d_gib_mean"],
                    "all_generated_sequences_match_standard": candidate[
                        "all_generated_sequences_match_standard"
                    ],
                    "all_step_top1_match_standard": candidate[
                        "all_step_top1_match_standard"
                    ],
                }
            )
        for resident in {row["resident_kv_heads_r"] for row in summary}:
            sync = next(
                (
                    row
                    for row in summary
                    if row["sequence_length"] == length
                    and row["family"] == "sync"
                    and row["resident_kv_heads_r"] == resident
                    and row["status"] == "ok"
                ),
                None,
            )
            async_row = next(
                (
                    row
                    for row in summary
                    if row["sequence_length"] == length
                    and row["family"] == "async"
                    and row["resident_kv_heads_r"] == resident
                    and row["status"] == "ok"
                ),
                None,
            )
            if sync is None or async_row is None:
                continue
            output.append(
                {
                    "comparison": "async_vs_sync_same_residency",
                    "sequence_length": length,
                    "candidate_method": async_row["method"],
                    "reference_method": sync["method"],
                    "resident_kv_heads_r": resident,
                    "prefill_cuda_change_percent": (
                        async_row["prefill_cuda_ms_mean"]
                        / sync["prefill_cuda_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "steady_decode_cuda_change_percent": (
                        async_row["decode_steady_cuda_mean_ms_mean"]
                        / sync["decode_steady_cuda_mean_ms_mean"]
                        - 1.0
                    )
                    * 100.0,
                    "async_over_sync_steady_speed_ratio": (
                        sync["decode_steady_cuda_mean_ms_mean"]
                        / async_row["decode_steady_cuda_mean_ms_mean"]
                    ),
                    "sync_h2d_gib": sync["h2d_gib_mean"],
                    "async_h2d_gib": async_row["h2d_gib_mean"],
                    "async_h2d_event_total_ms": async_row[
                        "async_h2d_event_total_ms_mean"
                    ],
                    "same_persistent_gpu_kv_gib": math.isclose(
                        sync["cache_gpu_gib_mean"],
                        async_row["cache_gpu_gib_mean"],
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                }
            )
    return output


def validate_results(
    rows: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    *,
    specs: list[MethodSpec],
    lengths: list[int],
    repeats: int,
    warmup_repeats: int,
) -> None:
    for length in lengths:
        for spec in specs:
            measured = [
                row
                for row in rows
                if row["method"] == spec.method
                and int(row["sequence_length"]) == length
                and row["status"] == "ok"
            ]
            warmup = [
                row
                for row in warmups
                if row["method"] == spec.method
                and int(row["sequence_length"]) == length
                and row["status"] == "ok"
            ]
            if len(measured) != repeats:
                raise RuntimeError(
                    f"Expected {repeats} successful trials for {spec.method} at {length}, "
                    f"found {len(measured)}"
                )
            if len(warmup) != warmup_repeats:
                raise RuntimeError(
                    f"Expected {warmup_repeats} warmups for {spec.method} at {length}, "
                    f"found {len(warmup)}"
                )
            if any(abs(float(row["cache_ratio"]) - 1.0) > 1e-6 for row in measured):
                raise RuntimeError(f"KV-cache byte mismatch for {spec.method} at {length}")
            if any(
                not row["same_generated_sequence_as_standard"]
                or not row["all_step_top1_equal_as_standard"]
                for row in measured
            ):
                raise RuntimeError(
                    f"Behavioral mismatch against Standard for {spec.method} at {length}"
                )


def save_results(
    results_dir: Path,
    rows: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    specs: list[MethodSpec],
    args: argparse.Namespace,
) -> None:
    summary = summarize(rows, specs)
    comparison = comparison_rows(summary)
    write_csv(results_dir / "day9_warmup.csv", warmups)
    write_csv(results_dir / "day9_raw.csv", rows)
    write_csv(results_dir / "day9_per_token.csv", per_token_rows(rows))
    write_csv(results_dir / "day9_summary.csv", summary)
    write_csv(results_dir / "day9_comparison.csv", comparison)
    manifest = {
        "benchmark": "StageKV Day-9 calibrated benchmark",
        "performance_claim_protocol": {
            "isolated_worker_per_context_length": True,
            "warmup_rounds_excluded": args.warmup_repeats,
            "measured_repetitions": args.repeats,
            "decode_tokens": args.decode_tokens,
            "rotated_execution_order": True,
            "fixed_seed": args.seed,
            "greedy_behavioral_match_required": True,
            "cuda_event_timing": True,
        },
        "model_path": args.model_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "gpu_total_memory_gib": (
            gib(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "lengths": args.lengths,
        "methods": [spec.method for spec in specs],
        "files": {
            "warmup": "day9_warmup.csv",
            "raw": "day9_raw.csv",
            "per_token": "day9_per_token.csv",
            "summary": "day9_summary.csv",
            "comparison": "day9_comparison.csv",
        },
    }
    (results_dir / "day9_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def orchestrate(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    specs = method_specs(args.resident_heads, args.stagekv_modes)
    rows: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    for length in args.lengths:
        print(f"\n=== Day-9 sequence_length={length} ===", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-path",
            args.model_path,
            "--sequence-length",
            str(length),
            "--decode-tokens",
            str(args.decode_tokens),
            "--repeats",
            str(args.repeats),
            "--warmup-repeats",
            str(args.warmup_repeats),
            "--seed",
            str(args.seed),
            "--resident-heads",
            *[str(value) for value in args.resident_heads],
            "--stagekv-modes",
            *args.stagekv_modes,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if line.startswith(MARKER):
                row = json.loads(line[len(MARKER) :])
                (warmups if row["is_warmup"] else rows).append(row)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Worker failed for sequence length {length}: {return_code}")
        save_results(results_dir, rows, warmups, specs, args)

    validate_results(
        rows,
        warmups,
        specs=specs,
        lengths=args.lengths,
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
    )
    save_results(results_dir, rows, warmups, specs, args)
    print(f"warmup={results_dir / 'day9_warmup.csv'}")
    print(f"raw={results_dir / 'day9_raw.csv'}")
    print(f"per_token={results_dir / 'day9_per_token.csv'}")
    print(f"summary={results_dir / 'day9_summary.csv'}")
    print(f"comparison={results_dir / 'day9_comparison.csv'}")
    print(f"manifest={results_dir / 'day9_manifest.json'}")
    print("stagekv_day9_calibrated_benchmark=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resident-heads",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESIDENT_HEADS),
    )
    parser.add_argument(
        "--stagekv-modes",
        choices=("sync", "async"),
        nargs="+",
        default=("sync", "async"),
    )
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decode_tokens < 3:
        raise ValueError("decode-tokens must be at least 3")
    if args.repeats < 5:
        raise ValueError("Day-9 requires at least five measured repetitions")
    if args.warmup_repeats < 1:
        raise ValueError("warmup-repeats must be at least 1")
    if any(length < 1 for length in args.lengths):
        raise ValueError("all context lengths must be positive")
    if any(not 0 <= resident <= 4 for resident in args.resident_heads):
        raise ValueError("resident-heads must be in [0, 4] for Qwen2.5-7B")
    if len(set(args.resident_heads)) != len(args.resident_heads):
        raise ValueError("resident-heads must not contain duplicates")
    if args.worker:
        if args.sequence_length is None:
            raise ValueError("worker mode requires --sequence-length")
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
