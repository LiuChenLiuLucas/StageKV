"""Measure the CPU-GPU transfer path used by StageKV.

The Day-9 results imply an effective transfer rate near 0.1 GiB/s.  This
standalone probe determines whether the bottleneck comes from the host PCIe
environment or from StageKV's attention/offload implementation.  It measures
pageable and pinned host memory, H2D and D2H directions, blocking and
non-blocking copies, and both the default and a dedicated CUDA stream.

Run this script without loading an LLM and without using the debugger.  Warmup
copies are excluded from the reported measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_RESULTS_DIR = "/root/stagekv/results/pcie_bandwidth_probe"
DEFAULT_SIZES_MIB = (4, 8, 32, 128)


@dataclass(frozen=True)
class CopyCase:
    name: str
    direction: str
    pinned: bool
    non_blocking: bool
    dedicated_stream: bool


CASES = (
    CopyCase("h2d_pageable_blocking", "h2d", False, False, False),
    CopyCase("d2h_pageable_blocking", "d2h", False, False, False),
    CopyCase("h2d_pinned_blocking", "h2d", True, False, False),
    CopyCase("d2h_pinned_blocking", "d2h", True, False, False),
    CopyCase("h2d_pinned_nonblocking_default", "h2d", True, True, False),
    CopyCase("d2h_pinned_nonblocking_default", "d2h", True, True, False),
    CopyCase("h2d_pinned_nonblocking_dedicated", "h2d", True, True, True),
    CopyCase("d2h_pinned_nonblocking_dedicated", "d2h", True, True, True),
)


def percentile(values: list[float], quantile: float) -> float:
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


def gib_per_second(byte_count: int, milliseconds: float) -> float:
    if milliseconds <= 0.0:
        return math.inf
    return byte_count / 1024**3 / (milliseconds / 1000.0)


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


def query_pcie_link() -> dict[str, Any]:
    fields = (
        "name,driver_version,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [part.strip() for part in process.stdout.strip().split(",")]
        names = [
            "gpu_name",
            "driver_version",
            "pcie_generation_current",
            "pcie_generation_max",
            "pcie_width_current",
            "pcie_width_max",
        ]
        return dict(zip(names, values))
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"query_error": str(exc)}


def allocate_case(
    case: CopyCase,
    byte_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.empty(
        byte_count,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=case.pinned,
    )
    device = torch.empty(byte_count, dtype=torch.uint8, device="cuda")
    if case.direction == "h2d":
        host.fill_(17)
        return host, device
    device.fill_(23)
    torch.cuda.synchronize()
    return device, host


def one_copy(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    non_blocking: bool,
    stream: torch.cuda.Stream,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with torch.cuda.stream(stream):
        start_event.record(stream)
        destination.copy_(source, non_blocking=non_blocking)
        end_event.record(stream)
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    return wall_ms, cuda_ms


def run_case(
    case: CopyCase,
    *,
    size_mib: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    byte_count = size_mib * 1024**2
    source, destination = allocate_case(case, byte_count)
    stream = (
        torch.cuda.Stream(device=0)
        if case.dedicated_stream
        else torch.cuda.current_stream()
    )
    if case.pinned:
        host_tensor = source if source.device.type == "cpu" else destination
        if not host_tensor.is_pinned():
            raise RuntimeError(f"Pinned allocation failed for {case.name}")

    for _ in range(warmups):
        one_copy(
            source,
            destination,
            non_blocking=case.non_blocking,
            stream=stream,
        )

    rows: list[dict[str, Any]] = []
    for trial in range(1, repeats + 1):
        wall_ms, cuda_ms = one_copy(
            source,
            destination,
            non_blocking=case.non_blocking,
            stream=stream,
        )
        row = {
            **asdict(case),
            "size_mib": size_mib,
            "bytes": byte_count,
            "trial": trial,
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "wall_gib_per_s": gib_per_second(byte_count, wall_ms),
            "cuda_gib_per_s": gib_per_second(byte_count, cuda_ms),
        }
        rows.append(row)

    del source, destination, stream
    torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["name"], int(row["size_mib"])) for row in rows})
    for name, size_mib in keys:
        group = [
            row
            for row in rows
            if row["name"] == name and int(row["size_mib"]) == size_mib
        ]
        first = group[0]
        summary: dict[str, Any] = {
            "name": name,
            "direction": first["direction"],
            "pinned": first["pinned"],
            "non_blocking": first["non_blocking"],
            "dedicated_stream": first["dedicated_stream"],
            "size_mib": size_mib,
            "repeats": len(group),
        }
        for metric in ("wall_ms", "cuda_ms", "wall_gib_per_s", "cuda_gib_per_s"):
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_p50"] = percentile(values, 0.50)
            summary[f"{metric}_p95"] = percentile(values, 0.95)
        summaries.append(summary)
    return summaries


def classify_bandwidth(value: float) -> str:
    if value >= 8.0:
        return "NORMAL_FOR_STAGEKV"
    if value >= 2.0:
        return "DEGRADED"
    return "SEVERE_BOTTLENECK"


def diagnostic_report(
    summary: list[dict[str, Any]],
    largest_size_mib: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in summary
        if int(row["size_mib"]) == largest_size_mib
        and bool(row["pinned"])
        and bool(row["non_blocking"])
    ]
    h2d = [row for row in candidates if row["direction"] == "h2d"]
    d2h = [row for row in candidates if row["direction"] == "d2h"]
    if not h2d or not d2h:
        raise RuntimeError("Missing pinned non-blocking H2D or D2H measurements")
    best_h2d = max(h2d, key=lambda row: float(row["cuda_gib_per_s_p50"]))
    best_d2h = max(d2h, key=lambda row: float(row["cuda_gib_per_s_p50"]))
    h2d_value = float(best_h2d["cuda_gib_per_s_p50"])
    d2h_value = float(best_d2h["cuda_gib_per_s_p50"])
    overall = min(h2d_value, d2h_value)
    return {
        "classification": classify_bandwidth(overall),
        "diagnostic_size_mib": largest_size_mib,
        "best_pinned_h2d_case": best_h2d["name"],
        "best_pinned_h2d_cuda_p50_gib_per_s": h2d_value,
        "best_pinned_d2h_case": best_d2h["name"],
        "best_pinned_d2h_cuda_p50_gib_per_s": d2h_value,
        "decision": (
            "PCIe bandwidth is adequate. Optimize StageKV copy granularity, "
            "D2H cache writes, stream scheduling, and grouped attention."
            if overall >= 8.0
            else (
                "PCIe bandwidth is lower than expected. Verify link width, "
                "driver, PyTorch/CUDA build, and cloud host contention."
                if overall >= 2.0
                else
                "The host transfer path is severely constrained. Do not run "
                "long-context StageKV benchmarks until the environment is fixed."
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--sizes-mib",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES_MIB),
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.warmups < 1 or args.repeats < 5:
        raise ValueError("Use at least one warmup and five measured repetitions")
    if any(size < 1 for size in args.sizes_mib):
        raise ValueError("All transfer sizes must be positive")
    if len(set(args.sizes_mib)) != len(args.sizes_mib):
        raise ValueError("Transfer sizes must not contain duplicates")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(
        f"sizes_mib={args.sizes_mib} warmups={args.warmups} "
        f"repeats={args.repeats}",
        flush=True,
    )

    raw_rows: list[dict[str, Any]] = []
    for size_mib in args.sizes_mib:
        for case in CASES:
            print(f"running={case.name} size_mib={size_mib}", flush=True)
            case_rows = run_case(
                case,
                size_mib=size_mib,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            raw_rows.extend(case_rows)
            median_cuda = percentile(
                [float(row["cuda_gib_per_s"]) for row in case_rows], 0.50
            )
            median_wall = percentile(
                [float(row["wall_gib_per_s"]) for row in case_rows], 0.50
            )
            print(
                f"result={case.name} size_mib={size_mib} "
                f"cuda_p50={median_cuda:.3f}GiB/s "
                f"wall_p50={median_wall:.3f}GiB/s",
                flush=True,
            )

    summary = summarize(raw_rows)
    diagnostic = diagnostic_report(summary, max(args.sizes_mib))
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "pcie": query_pcie_link(),
    }
    document = {
        "environment": environment,
        "protocol": {
            "sizes_mib": args.sizes_mib,
            "warmups_excluded": args.warmups,
            "measured_repetitions": args.repeats,
        },
        "diagnostic": diagnostic,
        "summary": summary,
    }

    raw_path = results_dir / "pcie_bandwidth_raw.csv"
    summary_path = results_dir / "pcie_bandwidth_summary.csv"
    json_path = results_dir / "pcie_bandwidth_report.json"
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary)
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    print(f"raw={raw_path}")
    print(f"summary={summary_path}")
    print(f"report={json_path}")
    print("pcie_bandwidth_probe=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
