from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


PROJECT_DIR = Path("/root/stagekv")
MODEL_PATH = Path("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct")
BENCHMARK = PROJECT_DIR / "stagekv_cross_layer_calibrated_benchmark.py"

# 每次正式实验必须换新名字，例如 run2、run3。
RESULTS_DIR = PROJECT_DIR / "results" / "day13_context_8k_run1"

METHODS = {
    "standard",
    "stagekv_bidirectional_r2",
    "stagekv_cross_layer_r2",
}


def read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / name
    if not path.is_file():
        raise RuntimeError(f"Missing result file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def verify_environment() -> None:
    required_files = [
        BENCHMARK,
        PROJECT_DIR / "stagekv_cpu_g2_correctness.py",
        PROJECT_DIR / "stagekv_pinned_residency_correctness.py",
        PROJECT_DIR / "stagekv_bidirectional_async_correctness.py",
        PROJECT_DIR / "stagekv_cross_layer_prefetch_correctness.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]

    if missing:
        raise RuntimeError(f"Missing source files: {missing}")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Model directory does not exist: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected PyCharm interpreter")
    if RESULTS_DIR.exists():
        raise RuntimeError(
            f"Results directory already exists: {RESULTS_DIR}\n"
            "Change RESULTS_DIR to a new run name; do not overwrite it."
        )

    print("torch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("results:", RESULTS_DIR)


def run_benchmark() -> None:
    RESULTS_DIR.mkdir(parents=True)

    command = [
        sys.executable,
        str(BENCHMARK),
        "--model-path", str(MODEL_PATH),
        "--results-dir", str(RESULTS_DIR),
        "--lengths", "8192",
        "--decode-tokens", "32",
        "--warmup-repeats", "2",
        "--repeats", "5",
        "--resident-heads", "2",
        "--stagekv-modes", "bidirectional", "cross_layer",
    ]

    log_path = RESULTS_DIR / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Benchmark failed with exit code {return_code}. See {log_path}"
        )


def audit_results() -> None:
    raw = read_csv("day12_raw.csv")
    warmup = read_csv("day12_warmup.csv")
    per_token = read_csv("day12_per_token.csv")
    summary = read_csv("day12_summary.csv")
    comparison = read_csv("day12_comparison.csv")

    manifest_path = RESULTS_DIR / "day12_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(warmup) == 6, f"Expected 6 warmups, found {len(warmup)}"
    assert len(raw) == 15, f"Expected 15 measured rows, found {len(raw)}"
    assert len(per_token) == 480, f"Expected 480 token rows, found {len(per_token)}"
    assert {row["method"] for row in raw} == METHODS
    assert manifest["lengths"] == [8192]
    assert set(manifest["methods"]) == METHODS

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["method"]].append(row)

        assert row["status"] == "ok"
        assert int(row["sequence_length"]) == 8192
        assert int(row["decode_tokens"]) == 32
        assert abs(float(row["cache_ratio"]) - 1.0) < 1e-6
        assert as_bool(row["same_generated_sequence_as_standard"])
        assert as_bool(row["all_step_top1_equal_as_standard"])

        if row["method"] != "standard":
            assert int(row["blocking_d2h_tensor_copies"]) == 0
            assert int(row["non_blocking_d2h_tensor_copies"]) > 0
            assert int(row["async_d2h_append_calls"]) > 0
            assert as_bool(row["dedicated_d2h_stream_enabled"])

        if row["method"] == "stagekv_cross_layer_r2":
            expected = {
                "layer_prefetch_calls": 868,
                "lookahead_prefetch_calls": 837,
                "layer0_fallback_prefetch_calls": 31,
                "layer_prefetch_hits": 837,
                "layer_prefetch_misses": 31,
                "layer_slot_reuse_waits": 866,
            }
            for field, value in expected.items():
                assert int(row[field]) == value, (field, row[field], value)
            assert as_bool(row["cross_layer_schedule_correct"])

    for method, rows in grouped.items():
        assert len(rows) == 5
        position_counts = Counter(int(row["round_order_position"]) for row in rows)
        assert max(position_counts.values()) - min(position_counts.values()) <= 1

    standard = grouped["standard"]
    bidirectional = grouped["stagekv_bidirectional_r2"]
    cross_layer = grouped["stagekv_cross_layer_r2"]

    standard_total = mean(standard, "cache_total_gib")
    for rows in (bidirectional, cross_layer):
        gpu_ratio = mean(rows, "cache_gpu_gib") / mean(rows, "cache_total_gib")
        cpu_ratio = mean(rows, "cache_cpu_gib") / mean(rows, "cache_total_gib")
        assert abs(gpu_ratio - 0.5) < 1e-6
        assert abs(cpu_ratio - 0.5) < 1e-6
        assert mean(rows, "h2d_gib") > 0
        assert mean(rows, "d2h_gib") > 0

    bidirectional_ms = mean(bidirectional, "decode_steady_cuda_mean_ms")
    cross_layer_ms = mean(cross_layer, "decode_steady_cuda_mean_ms")
    speed_ratio = bidirectional_ms / cross_layer_ms

    order_latency = {}
    for method, rows in grouped.items():
        values = defaultdict(list)
        for row in rows:
            values[int(row["round_order_position"])].append(
                float(row["decode_steady_cuda_mean_ms"])
            )
        order_latency[method] = {
            str(position): statistics.mean(latencies)
            for position, latencies in sorted(values.items())
        }

    report = {
        "protocol_passed": True,
        "correctness_passed": True,
        "measured_rows": len(raw),
        "warmup_rows": len(warmup),
        "per_token_rows": len(per_token),
        "standard_total_kv_gib": standard_total,
        "stagekv_gpu_kv_gib": mean(cross_layer, "cache_gpu_gib"),
        "stagekv_cpu_kv_gib": mean(cross_layer, "cache_cpu_gib"),
        "bidirectional_h2d_gib": mean(bidirectional, "h2d_gib"),
        "cross_layer_h2d_gib": mean(cross_layer, "h2d_gib"),
        "bidirectional_d2h_gib": mean(bidirectional, "d2h_gib"),
        "cross_layer_d2h_gib": mean(cross_layer, "d2h_gib"),
        "bidirectional_steady_cuda_ms": bidirectional_ms,
        "cross_layer_steady_cuda_ms": cross_layer_ms,
        "cross_layer_over_bidirectional_speed_ratio": speed_ratio,
        "minimum_speed_ratio": 1.05,
        "meets_speed_target": speed_ratio >= 1.05,
        "steady_cuda_ms_by_execution_position": order_latency,
        "paper_ready": manifest["performance_claim_protocol"]["paper_ready"],
    }

    report_path = RESULTS_DIR / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== 8K AUDIT =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("audit report:", report_path)

    if speed_ratio < 1.05:
        print("\nDECISION: 未达到 1.05x，冻结跨层稳定加速主张。")
        print("后续转为显存-延迟权衡，并开始参数化模型结构检查。")
    else:
        print("\nDECISION: 本批达到 1.05x，但仍应独立重复后再宣称稳定收益。")


if __name__ == "__main__":
    verify_environment()
    run_benchmark()
    audit_results()