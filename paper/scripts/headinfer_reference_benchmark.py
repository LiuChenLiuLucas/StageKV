from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from headinfer.cache import OffloadedCache as HeadInferOffloadedCache
from headinfer.mp import mp_headinfer


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class InstrumentedHeadInferCache(HeadInferOffloadedCache):
    """HeadInfer cache with transfer-byte and transfer-call counters."""

    def __init__(self) -> None:
        super().__init__()
        self.h2d_tensor_copies = 0
        self.h2d_bytes = 0
        self.d2h_tensor_copies = 0
        self.d2h_bytes = 0

    def prefetch_layer(self, layer_idx: int):
        if layer_idx < len(self):
            for tensor in (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
            ):
                if tensor.device.type == "cpu":
                    self.h2d_tensor_copies += 1
                    self.h2d_bytes += tensor_bytes(tensor)

        return super().prefetch_layer(layer_idx)

    def evict_previous_layer(self, layer_idx: int):
        if len(self) > 2:
            previous_idx = (layer_idx - 1) % len(self)

            for tensor in (
                self.key_cache[previous_idx],
                self.value_cache[previous_idx],
            ):
                if tensor.device.type == "cuda":
                    self.d2h_tensor_copies += 1
                    self.d2h_bytes += tensor_bytes(tensor)

        return super().evict_previous_layer(layer_idx)


def cache_placement(cache: Any) -> dict[str, int]:
    cpu_bytes = 0
    gpu_bytes = 0
    total_bytes = 0
    cpu_tensors = 0
    gpu_tensors = 0

    for key_tensor, value_tensor in zip(
        cache.key_cache,
        cache.value_cache,
    ):
        for tensor in (key_tensor, value_tensor):
            if tensor is None:
                continue

            size = tensor_bytes(tensor)
            total_bytes += size

            if tensor.device.type == "cuda":
                gpu_bytes += size
                gpu_tensors += 1
            elif tensor.device.type == "cpu":
                cpu_bytes += size
                cpu_tensors += 1

    return {
        "cache_cpu_bytes": cpu_bytes,
        "cache_gpu_bytes": gpu_bytes,
        "cache_total_bytes": total_bytes,
        "cache_cpu_tensors": cpu_tensors,
        "cache_gpu_tensors": gpu_tensors,
    }


def timed_forward(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | InstrumentedHeadInferCache,
) -> tuple[torch.Tensor, float, float]:
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
        last_hidden = output.last_hidden_state[:, -1:, :]
        logits = model.lm_head(last_hidden)[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)

    end_event.record()
    end_event.synchronize()

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))

    del output, last_hidden, logits
    return next_token, wall_ms, cuda_ms


def build_cache(
    method: str,
) -> DynamicCache | InstrumentedHeadInferCache:
    if method == "standard":
        return DynamicCache()

    if method == "headinfer":
        return InstrumentedHeadInferCache()

    raise ValueError(f"Unknown method: {method}")


def run_once(
    model: AutoModelForCausalLM,
    *,
    method: str,
    input_ids: torch.Tensor,
    decode_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if decode_tokens < 3:
        raise ValueError("decode_tokens must be at least 3")

    process = psutil.Process()
    cache = build_cache(method)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    cpu_rss_before = process.memory_info().rss

    attention_mask = torch.ones_like(input_ids)

    next_token, prefill_wall_ms, prefill_cuda_ms = timed_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        cache=cache,
    )

    generated_ids = [int(next_token.item())]
    decode_wall_ms: list[float] = []
    decode_cuda_ms: list[float] = []

    prefill_peak_allocated = torch.cuda.max_memory_allocated()
    prefill_peak_reserved = torch.cuda.max_memory_reserved()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    decode_total_start = time.perf_counter()

    for decode_step in range(1, decode_tokens):
        current_length = input_ids.shape[1] + decode_step

        attention_mask = torch.ones(
            (1, current_length),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        next_token, wall_ms, cuda_ms = timed_forward(
            model,
            input_ids=next_token,
            attention_mask=attention_mask,
            cache=cache,
        )

        generated_ids.append(int(next_token.item()))
        decode_wall_ms.append(wall_ms)
        decode_cuda_ms.append(cuda_ms)

    torch.cuda.synchronize()
    decode_total_wall_ms = (
        time.perf_counter() - decode_total_start
    ) * 1000.0

    decode_peak_allocated = torch.cuda.max_memory_allocated()
    decode_peak_reserved = torch.cuda.max_memory_reserved()

    placement = cache_placement(cache)
    cpu_rss_after = process.memory_info().rss

    max_cache_len = input_ids.shape[1] + decode_tokens - 1
    theoretical_cache_bytes = (
        2
        * layers
        * kv_heads
        * max_cache_len
        * head_dim
        * torch.empty((), dtype=torch.bfloat16).element_size()
    )

    steady_wall = decode_wall_ms[1:]
    steady_cuda = decode_cuda_ms[1:]

    if not steady_wall or not steady_cuda:
        raise RuntimeError("Not enough decode steps for steady metrics")

    h2d_bytes = int(getattr(cache, "h2d_bytes", 0))
    d2h_bytes = int(getattr(cache, "d2h_bytes", 0))
    h2d_copies = int(getattr(cache, "h2d_tensor_copies", 0))
    d2h_copies = int(getattr(cache, "d2h_tensor_copies", 0))

    row = {
        "status": "ok",
        "method": method,
        "sequence_length": int(input_ids.shape[1]),
        "decode_tokens": decode_tokens,
        "generated_token_ids": json.dumps(generated_ids),
        "prefill_wall_ms": prefill_wall_ms,
        "prefill_cuda_ms": prefill_cuda_ms,
        "decode_first_step_wall_ms": decode_wall_ms[0],
        "decode_first_step_cuda_ms": decode_cuda_ms[0],
        "decode_steady_wall_mean_ms": statistics.mean(steady_wall),
        "decode_steady_cuda_mean_ms": statistics.mean(steady_cuda),
        "decode_steady_cuda_p50_ms": percentile(steady_cuda, 0.50),
        "decode_steady_cuda_p95_ms": percentile(steady_cuda, 0.95),
        "decode_end_to_end_wall_ms": decode_total_wall_ms,
        "decode_end_to_end_ms_per_token": (
            decode_total_wall_ms / len(decode_wall_ms)
        ),
        "decode_tokens_per_second": (
            1000.0 / statistics.mean(steady_cuda)
        ),
        "prefill_peak_gpu_allocated_gib": gib(
            prefill_peak_allocated
        ),
        "prefill_peak_gpu_reserved_gib": gib(
            prefill_peak_reserved
        ),
        "decode_peak_gpu_allocated_gib": gib(
            decode_peak_allocated
        ),
        "decode_peak_gpu_reserved_gib": gib(
            decode_peak_reserved
        ),
        "cache_gpu_gib": gib(placement["cache_gpu_bytes"]),
        "cache_cpu_gib": gib(placement["cache_cpu_bytes"]),
        "cache_total_gib": gib(placement["cache_total_bytes"]),
        "cache_gpu_tensors": placement["cache_gpu_tensors"],
        "cache_cpu_tensors": placement["cache_cpu_tensors"],
        "theoretical_cache_gib": gib(theoretical_cache_bytes),
        "cache_ratio": (
            placement["cache_total_bytes"] / theoretical_cache_bytes
        ),
        "h2d_tensor_copies": h2d_copies,
        "h2d_gib": gib(h2d_bytes),
        "d2h_tensor_copies": d2h_copies,
        "d2h_gib": gib(d2h_bytes),
        "cpu_rss_before_gib": gib(cpu_rss_before),
        "cpu_rss_after_gib": gib(cpu_rss_after),
        "cpu_rss_delta_gib": gib(
            cpu_rss_after - cpu_rss_before
        ),
    }

    del cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_std(
    rows: list[dict[str, Any]],
    key: str,
) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--method",
        choices=("standard", "headinfer"),
        required=True,
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    if config.model_type != "qwen2":
        raise RuntimeError(
            f"Expected Qwen2/Qwen2.5, got {config.model_type}"
        )

    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    layers = int(config.num_hidden_layers)
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // query_heads
    )

    print(
        f"method={args.method} "
        f"model={args.model_path} "
        f"layers={layers} "
        f"query_heads={query_heads} "
        f"kv_heads={kv_heads} "
        f"head_dim={head_dim}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()

    if args.method == "headinfer":
        mp_headinfer(model)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    input_ids = torch.randint(
        low=10,
        high=int(config.vocab_size) - 10,
        size=(1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to("cuda")

    for warmup_index in range(args.warmup_repeats):
        print(
            f"warmup={warmup_index + 1}/{args.warmup_repeats}",
            flush=True,
        )

        run_once(
            model,
            method=args.method,
            input_ids=input_ids,
            decode_tokens=args.decode_tokens,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    rows: list[dict[str, Any]] = []

    for trial in range(1, args.repeats + 1):
        print(f"trial={trial}/{args.repeats}", flush=True)

        row = run_once(
            model,
            method=args.method,
            input_ids=input_ids,
            decode_tokens=args.decode_tokens,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        row["trial"] = trial
        rows.append(row)

    raw_path = args.results_dir / (
        f"headinfer_reference_{args.method}_"
        f"{args.sequence_length}_raw.csv"
    )
    report_path = args.results_dir / (
        f"headinfer_reference_{args.method}_"
        f"{args.sequence_length}_report.json"
    )

    write_csv(raw_path, rows)

    latency_mean, latency_std = mean_std(
        rows,
        "decode_end_to_end_ms_per_token",
    )
    steady_mean, steady_std = mean_std(
        rows,
        "decode_steady_cuda_mean_ms",
    )
    gpu_kv_mean, gpu_kv_std = mean_std(
        rows,
        "cache_gpu_gib",
    )

    generated_sequences = {
        row["generated_token_ids"] for row in rows
    }

    report = {
        "method": args.method,
        "model_path": args.model_path,
        "sequence_length": args.sequence_length,
        "decode_tokens": args.decode_tokens,
        "warmup_repeats": args.warmup_repeats,
        "measured_repeats": args.repeats,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hostname": platform.node(),
        },
        "model_structure": {
            "layers": layers,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "results": {
            "decode_end_to_end_ms_per_token_mean": latency_mean,
            "decode_end_to_end_ms_per_token_std": latency_std,
            "decode_steady_cuda_mean_ms": steady_mean,
            "decode_steady_cuda_std_ms": steady_std,
            "cache_gpu_gib_mean": gpu_kv_mean,
            "cache_gpu_gib_std": gpu_kv_std,
            "generated_sequence_stable_across_trials": (
                len(generated_sequences) == 1
            ),
        },
        "rows": rows,
        "note": (
            "This is the local HeadInfer source-code baseline. "
            "It does not claim to reproduce all adaptive grouping "
            "or million-token results reported in the paper."
        ),
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["results"], indent=2))
    print(f"raw={raw_path}")
    print(f"report={report_path}")
    print("headinfer_reference_benchmark=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())