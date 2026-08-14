"""Repeated Standard vs layer-offloaded KV-cache benchmark for Qwen2.5-7B.

Each method/sequence-length pair runs in an isolated child process. A worker
loads the model once, performs a short warm-up, then runs repeated trials that
measure prefill and autoregressive decode separately.
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
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, DynamicCache, OffloadedCache


MODEL_PATH = "/root/ModelScope/model/Qwen2.5-7B-Instruct"
RESULTS_DIR = "/root/headinfer/results/layer_offload"
LENGTHS = (4096, 8192, 16384, 32768)
METHODS = ("standard", "layer_offload")
MARKER = "__LAYER_RESULT__="


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def cache_placement(cache: Any) -> tuple[int, int, int]:
    cpu_bytes = 0
    gpu_bytes = 0
    total_bytes = 0
    for key, value in zip(cache.key_cache, cache.value_cache):
        for tensor in (key, value):
            if tensor is None:
                continue
            size = tensor.numel() * tensor.element_size()
            total_bytes += size
            if tensor.device.type == "cuda":
                gpu_bytes += size
            elif tensor.device.type == "cpu":
                cpu_bytes += size
    return cpu_bytes, gpu_bytes, total_bytes


def new_cache(method: str) -> DynamicCache | OffloadedCache:
    if method == "standard":
        return DynamicCache()
    if method == "layer_offload":
        return OffloadedCache()
    raise ValueError(f"Unknown method: {method}")


def forward_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | OffloadedCache,
) -> Any:
    return model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def warmup(model: AutoModelForCausalLM, method: str, vocab_size: int) -> None:
    ids = torch.randint(0, vocab_size, (1, 128), device="cuda")
    mask = torch.ones_like(ids)
    cache = new_cache(method)
    with torch.inference_mode():
        output = forward_hidden(model, ids, mask, cache)
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    torch.cuda.synchronize()
    del logits, output, cache, mask, ids
    gc.collect()
    torch.cuda.empty_cache()


def run_trial(
    model: AutoModelForCausalLM,
    method: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_tokens: int,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    process = psutil.Process()
    cpu_before = process.memory_info().rss
    cache = new_cache(method)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = forward_hidden(model, input_ids, attention_mask, cache)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    prefill_peak_allocated = torch.cuda.max_memory_allocated()
    prefill_peak_reserved = torch.cuda.max_memory_reserved()
    cpu_after_prefill = process.memory_info().rss

    with torch.inference_mode():
        first_logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    next_token = first_logits.argmax(dim=-1)
    generated_ids = [int(next_token.item())]
    del first_logits, output

    decode_latencies: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, decode_tokens):
        step_mask = torch.ones(
            (1, input_ids.shape[1] + step),
            dtype=attention_mask.dtype,
            device="cuda",
        )
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        with torch.inference_mode():
            decode_output = forward_hidden(model, next_token, step_mask, cache)
            logits = model.lm_head(decode_output.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        decode_latencies.append(time.perf_counter() - decode_start)
        generated_ids.append(int(next_token.item()))
        del logits, decode_output, step_mask

    decode_peak_allocated = torch.cuda.max_memory_allocated()
    decode_peak_reserved = torch.cuda.max_memory_reserved()
    cpu_after_decode = process.memory_info().rss
    torch.cuda.synchronize()
    cache_cpu, cache_gpu, cache_total = cache_placement(cache)

    config = model.config
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // config.num_attention_heads
    )
    final_cache_tokens = input_ids.shape[1] + max(0, decode_tokens - 1)
    theoretical_cache = (
        2
        * config.num_hidden_layers
        * final_cache_tokens
        * config.num_key_value_heads
        * head_dim
        * 2
    )

    result = {
        "method": method,
        "sequence_length": int(input_ids.shape[1]),
        "decode_tokens": decode_tokens,
        "generated_token_ids": json.dumps(generated_ids),
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": input_ids.shape[1] / prefill_seconds,
        "decode_seconds_mean": statistics.mean(decode_latencies),
        "decode_seconds_std": (
            statistics.stdev(decode_latencies) if len(decode_latencies) > 1 else 0.0
        ),
        "prefill_peak_gpu_allocated_gib": gib(prefill_peak_allocated),
        "prefill_peak_gpu_reserved_gib": gib(prefill_peak_reserved),
        "decode_peak_gpu_allocated_gib": gib(decode_peak_allocated),
        "decode_peak_gpu_reserved_gib": gib(decode_peak_reserved),
        "cpu_rss_before_gib": gib(cpu_before),
        "cpu_rss_after_prefill_gib": gib(cpu_after_prefill),
        "cpu_rss_after_decode_gib": gib(cpu_after_decode),
        "cache_cpu_gib": gib(cache_cpu),
        "cache_gpu_gib": gib(cache_gpu),
        "cache_total_gib": gib(cache_total),
        "theoretical_cache_gib": gib(theoretical_cache),
        "cache_ratio": cache_total / theoretical_cache,
    }

    del cache, next_token
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"loading_model={args.model_path}", flush=True)
    print(f"method={args.method} sequence_length={args.sequence_length}", flush=True)
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
        model.config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)

    print("warmup=started", flush=True)
    warmup(model, args.method, model.config.vocab_size)
    print("warmup=finished", flush=True)

    try:
        for trial in range(1, args.repeats + 1):
            result = run_trial(
                model,
                args.method,
                input_ids,
                attention_mask,
                args.decode_tokens,
            )
            result.update({"trial": trial, "status": "ok", "error": ""})
            print(MARKER + json.dumps(result), flush=True)
            print(
                f"trial={trial}/{args.repeats} "
                f"prefill={result['prefill_seconds']:.4f}s "
                f"decode={result['decode_seconds_mean']:.4f}s/token",
                flush=True,
            )
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        result = {
            "method": args.method,
            "sequence_length": args.sequence_length,
            "trial": 0,
            "status": "oom",
            "error": str(exc).replace("\n", " ")[:1000],
        }
        print(MARKER + json.dumps(result), flush=True)
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds_mean",
        "prefill_peak_gpu_allocated_gib",
        "prefill_peak_gpu_reserved_gib",
        "decode_peak_gpu_allocated_gib",
        "cpu_rss_after_prefill_gib",
        "cache_cpu_gib",
        "cache_gpu_gib",
        "cache_total_gib",
        "cache_ratio",
    )
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for length in sorted({int(row["sequence_length"]) for row in rows}):
            group = [
                row
                for row in rows
                if row.get("method") == method
                and int(row["sequence_length"]) == length
                and row.get("status") == "ok"
            ]
            if not group:
                summaries.append(
                    {
                        "method": method,
                        "sequence_length": length,
                        "status": "oom_or_missing",
                        "successful_trials": 0,
                    }
                )
                continue
            summary: dict[str, Any] = {
                "method": method,
                "sequence_length": length,
                "status": "ok",
                "successful_trials": len(group),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            summaries.append(summary)
    return summaries


def compare(
    summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    tested_lengths = sorted({int(row["sequence_length"]) for row in summary})
    for length in tested_lengths:
        standard = next(
            (
                row
                for row in summary
                if row["method"] == "standard"
                and row["sequence_length"] == length
                and row["status"] == "ok"
            ),
            None,
        )
        offload = next(
            (
                row
                for row in summary
                if row["method"] == "layer_offload"
                and row["sequence_length"] == length
                and row["status"] == "ok"
            ),
            None,
        )
        if standard is None or offload is None:
            comparisons.append(
                {"sequence_length": length, "status": "incomplete"}
            )
            continue
        comparisons.append(
            {
                "sequence_length": length,
                "status": "ok",
                "prefill_time_overhead_percent": (
                    offload["prefill_seconds_mean"] / standard["prefill_seconds_mean"] - 1
                )
                * 100,
                "decode_time_overhead_percent": (
                    offload["decode_seconds_mean_mean"]
                    / standard["decode_seconds_mean_mean"]
                    - 1
                )
                * 100,
                "prefill_allocated_gpu_saving_gib": (
                    standard["prefill_peak_gpu_allocated_gib_mean"]
                    - offload["prefill_peak_gpu_allocated_gib_mean"]
                ),
                "offloaded_cache_cpu_gib": offload["cache_cpu_gib_mean"],
                "offloaded_cache_gpu_gib": offload["cache_gpu_gib_mean"],
                "generated_sequences_match": (
                    next(
                        row["generated_token_ids"]
                        for row in raw_rows
                        if row.get("method") == "standard"
                        and int(row["sequence_length"]) == length
                        and row.get("status") == "ok"
                    )
                    == next(
                        row["generated_token_ids"]
                        for row in raw_rows
                        if row.get("method") == "layer_offload"
                        and int(row["sequence_length"]) == length
                        and row.get("status") == "ok"
                    )
                ),
            }
        )
    return comparisons


def orchestrate(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for length in args.lengths:
        for method in METHODS:
            print(f"\n=== method={method} sequence_length={length} ===", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--model-path",
                args.model_path,
                "--method",
                method,
                "--sequence-length",
                str(length),
                "--repeats",
                str(args.repeats),
                "--decode-tokens",
                str(args.decode_tokens),
                "--seed",
                str(args.seed),
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
                    rows.append(json.loads(line[len(MARKER) :]))
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"Worker failed: method={method}, length={length}, code={return_code}"
                )

            summary = summarize(rows)
            write_csv(results_dir / "layer_offload_raw.csv", rows)
            write_csv(results_dir / "layer_offload_summary.csv", summary)
            write_csv(
                results_dir / "layer_offload_comparison.csv",
                compare(summary, rows),
            )

    print(f"raw={results_dir / 'layer_offload_raw.csv'}")
    print(f"summary={results_dir / 'layer_offload_summary.csv'}")
    print(f"comparison={results_dir / 'layer_offload_comparison.csv'}")
    print("layer_offload_benchmark=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--sequence-length", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.method is None or args.sequence_length is None:
            raise ValueError("Worker mode requires --method and --sequence-length")
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
