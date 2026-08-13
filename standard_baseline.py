"""Reproducible Standard KV-cache baseline for Qwen2.5-7B.

The parent process launches one isolated worker per sequence length. Each
worker loads the model once, performs one same-length warm-up, and then runs
five measured prefill trials. Raw and aggregate results are saved as CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM


DEFAULT_MODEL_PATH = "/root/ModelScope/model/Qwen2.5-7B-Instruct"
DEFAULT_RESULTS_DIR = "/root/headinfer/results"
DEFAULT_LENGTHS = (4096, 8192, 16384, 32768)
RESULT_PREFIX = "__BASELINE_RESULT__="


@dataclass
class TrialResult:
    timestamp_utc: str
    method: str
    model_path: str
    sequence_length: int
    trial: int
    status: str
    error: str
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    prefill_seconds: float
    prefill_tokens_per_second: float
    peak_gpu_allocated_gib: float
    peak_gpu_reserved_gib: float
    measured_kv_gib: float
    theoretical_kv_gib: float
    kv_ratio: float
    process_cpu_rss_gib: float
    torch_version: str
    torch_cuda_version: str
    transformers_version: str
    gpu_name: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gib(byte_count: int | float) -> float:
    return float(byte_count) / (1024**3)


def cache_size_bytes(cache: Any) -> int:
    total = 0
    if hasattr(cache, "key_cache"):
        pairs = zip(cache.key_cache, cache.value_cache)
    else:
        pairs = ((layer[0], layer[1]) for layer in cache)

    for key, value in pairs:
        if key is not None:
            total += key.numel() * key.element_size()
        if value is not None:
            total += value.numel() * value.element_size()
    return total


def runtime_metadata() -> dict[str, str]:
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


def make_result(
    *,
    model_path: str,
    sequence_length: int,
    trial: int,
    status: str,
    error: str = "",
    config: Any | None = None,
    elapsed: float = math.nan,
    peak_allocated: int = 0,
    peak_reserved: int = 0,
    measured_kv_bytes: int = 0,
) -> TrialResult:
    layers = int(getattr(config, "num_hidden_layers", 0))
    query_heads = int(getattr(config, "num_attention_heads", 0))
    kv_heads = int(getattr(config, "num_key_value_heads", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    head_dim = int(getattr(config, "head_dim", 0) or (hidden_size // query_heads if query_heads else 0))
    theoretical = 2 * layers * sequence_length * kv_heads * head_dim * 2
    ratio = measured_kv_bytes / theoretical if theoretical else math.nan
    throughput = sequence_length / elapsed if elapsed and not math.isnan(elapsed) else math.nan

    return TrialResult(
        timestamp_utc=utc_now(),
        method="standard",
        model_path=model_path,
        sequence_length=sequence_length,
        trial=trial,
        status=status,
        error=error,
        layers=layers,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefill_seconds=elapsed,
        prefill_tokens_per_second=throughput,
        peak_gpu_allocated_gib=gib(peak_allocated),
        peak_gpu_reserved_gib=gib(peak_reserved),
        measured_kv_gib=gib(measured_kv_bytes),
        theoretical_kv_gib=gib(theoretical),
        kv_ratio=ratio,
        process_cpu_rss_gib=gib(psutil.Process().memory_info().rss),
        **runtime_metadata(),
    )


def run_forward(model: Any, input_ids: torch.Tensor) -> tuple[float, int, int, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    kv_bytes = cache_size_bytes(outputs.past_key_values)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del outputs
    return elapsed, peak_allocated, peak_reserved, kv_bytes


def worker(model_path: str, sequence_length: int, repeats: int, seed: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python interpreter")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"loading_model={model_path}", flush=True)
    print(f"sequence_length={sequence_length}", flush=True)

    model = None
    config = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.eval()
        config = model.config

        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + sequence_length)
        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(1, sequence_length),
            generator=generator,
            device="cuda",
        )

        print("warmup=started", flush=True)
        run_forward(model, input_ids)
        print("warmup=finished", flush=True)

        for trial in range(1, repeats + 1):
            elapsed, peak_allocated, peak_reserved, kv_bytes = run_forward(model, input_ids)
            result = make_result(
                model_path=model_path,
                sequence_length=sequence_length,
                trial=trial,
                status="ok",
                config=config,
                elapsed=elapsed,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                measured_kv_bytes=kv_bytes,
            )
            print(RESULT_PREFIX + json.dumps(asdict(result), ensure_ascii=True), flush=True)
            print(
                f"trial={trial}/{repeats} prefill_seconds={elapsed:.4f} "
                f"throughput={result.prefill_tokens_per_second:.2f}",
                flush=True,
            )
        return 0
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        result = make_result(
            model_path=model_path,
            sequence_length=sequence_length,
            trial=0,
            status="oom",
            error=str(exc).replace("\n", " ")[:1000],
            config=config,
        )
        print(RESULT_PREFIX + json.dumps(asdict(result), ensure_ascii=True), flush=True)
        return 0
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for length in sorted({int(row["sequence_length"]) for row in rows}):
        group = [row for row in rows if int(row["sequence_length"]) == length]
        successful = [row for row in group if row["status"] == "ok"]
        if not successful:
            summaries.append(
                {
                    "method": "standard",
                    "sequence_length": length,
                    "status": group[0]["status"],
                    "successful_trials": 0,
                    "prefill_seconds_mean": math.nan,
                    "prefill_seconds_std": math.nan,
                    "prefill_tokens_per_second_mean": math.nan,
                    "prefill_tokens_per_second_std": math.nan,
                    "peak_gpu_allocated_gib_max": math.nan,
                    "peak_gpu_reserved_gib_max": math.nan,
                    "measured_kv_gib_mean": math.nan,
                    "kv_ratio_mean": math.nan,
                }
            )
            continue

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in successful]

        prefill = values("prefill_seconds")
        throughput = values("prefill_tokens_per_second")
        summaries.append(
            {
                "method": "standard",
                "sequence_length": length,
                "status": "ok",
                "successful_trials": len(successful),
                "prefill_seconds_mean": statistics.mean(prefill),
                "prefill_seconds_std": statistics.stdev(prefill) if len(prefill) > 1 else 0.0,
                "prefill_tokens_per_second_mean": statistics.mean(throughput),
                "prefill_tokens_per_second_std": statistics.stdev(throughput) if len(throughput) > 1 else 0.0,
                "peak_gpu_allocated_gib_max": max(values("peak_gpu_allocated_gib")),
                "peak_gpu_reserved_gib_max": max(values("peak_gpu_reserved_gib")),
                "measured_kv_gib_mean": statistics.mean(values("measured_kv_gib")),
                "kv_ratio_mean": statistics.mean(values("kv_ratio")),
            }
        )
    return summaries


def save_environment(results_dir: Path) -> None:
    metadata = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    (results_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (results_dir / "requirements_freeze.txt").write_text(completed.stdout, encoding="utf-8")


def orchestrate(
    model_path: str,
    lengths: list[int],
    repeats: int,
    seed: int,
    results_dir: Path,
) -> int:
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_environment(results_dir)

    raw_rows: list[dict[str, Any]] = []
    for length in lengths:
        print(f"\n=== Standard baseline: sequence_length={length} ===", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-path",
            model_path,
            "--sequence-length",
            str(length),
            "--repeats",
            str(repeats),
            "--seed",
            str(seed),
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
            if line.startswith(RESULT_PREFIX):
                raw_rows.append(json.loads(line[len(RESULT_PREFIX) :]))
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Worker failed for sequence length {length} with exit code {return_code}")

        write_csv(results_dir / "standard_raw.csv", raw_rows)
        write_csv(results_dir / "standard_summary.csv", aggregate(raw_rows))

    print(f"\nraw_results={results_dir / 'standard_raw.csv'}")
    print(f"summary_results={results_dir / 'standard_summary.csv'}")
    print("baseline_suite=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.sequence_length is None:
            raise ValueError("--sequence-length is required in worker mode")
        return worker(args.model_path, args.sequence_length, args.repeats, args.seed)
    return orchestrate(
        model_path=args.model_path,
        lengths=args.lengths,
        repeats=args.repeats,
        seed=args.seed,
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
