from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


METHODS = ("standard", "bidirectional", "cross_layer")

T_CRITICAL_95 = {
    1: 12.7062047364,
    2: 4.3026527299,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242510,
    8: 2.3060041350,
    9: 2.2621571630,
    10: 2.2281388520,
}


def as_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return 0.0
    return float(value)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        raise ValueError("Empty value list")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summary_stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    mean, std = mean_std(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else 0.0,
    }


def confidence_interval_95(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    mean, std = mean_std(values)
    n = len(values)

    if n <= 1:
        return mean, mean

    t_value = T_CRITICAL_95.get(n - 1)
    if t_value is None:
        raise ValueError(f"No t critical value configured for n={n}")

    half_width = t_value * std / math.sqrt(n)
    return mean - half_width, mean + half_width


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    return rows


def validate_formal_rows(rows: list[dict[str, str]], model_name: str) -> list[dict[str, str]]:
    formal_rows = [
        row for row in rows
        if not as_bool(row.get("is_warmup", "False"))
    ]

    if len(formal_rows) != 15:
        raise ValueError(
            f"{model_name}: expected 15 formal rows, got {len(formal_rows)}"
        )

    family_counts = Counter(row.get("family", "") for row in formal_rows)
    expected_counts = Counter({
        "standard": 5,
        "bidirectional": 5,
        "cross_layer": 5,
    })

    if family_counts != expected_counts:
        raise ValueError(
            f"{model_name}: unexpected method counts: {family_counts}"
        )

    for row in formal_rows:
        if row.get("status") != "ok":
            raise ValueError(f"{model_name}: non-ok row found: {row}")

        if int(row["sequence_length"]) != 8192:
            raise ValueError(f"{model_name}: sequence length is not 8192")

        if int(row["decode_tokens"]) != 32:
            raise ValueError(f"{model_name}: decode_tokens is not 32")

        if not as_bool(row["same_generated_sequence_as_standard"]):
            raise ValueError(f"{model_name}: generated sequence gate failed")

        if not as_bool(row["all_step_top1_equal_as_standard"]):
            raise ValueError(f"{model_name}: Top-1 gate failed")

        if not as_bool(row["cross_layer_schedule_correct"]):
            raise ValueError(f"{model_name}: schedule gate failed")

        if int(row["round_order_position"]) not in {0, 1, 2}:
            raise ValueError(f"{model_name}: invalid order position")

    return formal_rows


def make_method_summary(
    rows: list[dict[str, str]],
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    result: list[dict[str, object]] = []
    numeric_stats: dict[str, dict[str, float]] = {}

    for method in METHODS:
        method_rows = [row for row in rows if row["family"] == method]
        latency_values = [
            as_float(row, "decode_steady_cuda_mean_ms")
            for row in method_rows
        ]
        stats = summary_stats(latency_values)
        numeric_stats[method] = stats

        result.append({
            "model": model_name,
            "method": method,
            "n": stats["n"],
            "mean_ms": stats["mean"],
            "std_ms": stats["std"],
            "cv": stats["cv"],
            "gpu_kv_gib": statistics.mean(
                as_float(row, "cache_gpu_gib") for row in method_rows
            ),
            "cpu_kv_gib": statistics.mean(
                as_float(row, "cache_cpu_gib") for row in method_rows
            ),
            "cache_total_gib": statistics.mean(
                as_float(row, "cache_total_gib") for row in method_rows
            ),
            "h2d_gib": statistics.mean(
                as_float(row, "h2d_gib") for row in method_rows
            ),
            "d2h_gib": statistics.mean(
                as_float(row, "d2h_gib") for row in method_rows
            ),
            "correctness_all": all(
                as_bool(row["same_generated_sequence_as_standard"])
                and as_bool(row["all_step_top1_equal_as_standard"])
                for row in method_rows
            ),
            "schedule_all": all(
                as_bool(row["cross_layer_schedule_correct"])
                for row in method_rows
            ),
        })

    standard_gpu_kv = next(
        item["gpu_kv_gib"]
        for item in result
        if item["method"] == "standard"
    )
    standard_latency = numeric_stats["standard"]["mean"]

    for item in result:
        item["latency_ratio_vs_standard"] = (
            item["mean_ms"] / standard_latency
        )
        item["gpu_kv_saving_pct"] = (
            100.0 * (1.0 - item["gpu_kv_gib"] / standard_gpu_kv)
            if standard_gpu_kv
            else 0.0
        )

    return result, numeric_stats


def make_pairs(
    rows: list[dict[str, str]],
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_trial: dict[int, dict[str, dict[str, str]]] = {}

    for row in rows:
        trial = int(row["trial"])
        by_trial.setdefault(trial, {})[row["family"]] = row

    pairs: list[dict[str, object]] = []

    for trial in sorted(by_trial):
        trial_rows = by_trial[trial]

        for method in METHODS:
            if method not in trial_rows:
                raise ValueError(
                    f"{model_name}: trial {trial} missing {method}"
                )

        standard_ms = as_float(
            trial_rows["standard"], "decode_steady_cuda_mean_ms"
        )
        bidirectional_ms = as_float(
            trial_rows["bidirectional"], "decode_steady_cuda_mean_ms"
        )
        cross_layer_ms = as_float(
            trial_rows["cross_layer"], "decode_steady_cuda_mean_ms"
        )

        difference = bidirectional_ms - cross_layer_ms
        ratio = bidirectional_ms / cross_layer_ms

        pairs.append({
            "model": model_name,
            "trial": trial,
            "standard_ms": standard_ms,
            "bidirectional_ms": bidirectional_ms,
            "cross_layer_ms": cross_layer_ms,
            "bidirectional_minus_cross_ms": difference,
            "bidirectional_over_cross_latency_ratio": ratio,
            "cross_layer_faster": difference > 0,
            "standard_order": int(
                trial_rows["standard"]["round_order_position"]
            ),
            "bidirectional_order": int(
                trial_rows["bidirectional"]["round_order_position"]
            ),
            "cross_layer_order": int(
                trial_rows["cross_layer"]["round_order_position"]
            ),
        })

    differences = [
        float(item["bidirectional_minus_cross_ms"])
        for item in pairs
    ]
    ratios = [
        float(item["bidirectional_over_cross_latency_ratio"])
        for item in pairs
    ]

    difference_stats = summary_stats(differences)
    ratio_stats = summary_stats(ratios)

    paired_report = {
        "model": model_name,
        "n_pairs": len(pairs),
        "difference_definition": (
            "bidirectional_latency - cross_layer_latency; "
            "positive means cross_layer is faster"
        ),
        "difference_stats": difference_stats,
        "difference_ci95_ms": confidence_interval_95(differences),
        "ratio_stats": ratio_stats,
        "ratio_ci95": confidence_interval_95(ratios),
        "cross_layer_faster_trials": sum(
            1 for item in pairs if item["cross_layer_faster"]
        ),
        "pairs": pairs,
    }

    return pairs, paired_report


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen7b", type=Path, required=True)
    parser.add_argument("--qwen3b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = [
        ("Qwen2.5-7B-Instruct", args.qwen7b),
        ("Qwen2.5-3B-Instruct", args.qwen3b),
    ]

    all_summary_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "protocol": {
            "sequence_length": 8192,
            "decode_tokens": 32,
            "warmups": 2,
            "measured_repeats": 5,
            "confidence_level": 0.95,
        },
        "models": {},
    }

    for model_name, path in datasets:
        raw_rows = load_csv(path)
        formal_rows = validate_formal_rows(raw_rows, model_name)

        summary_rows, method_stats = make_method_summary(
            formal_rows,
            model_name,
        )
        pair_rows, paired_report = make_pairs(
            formal_rows,
            model_name,
        )

        all_summary_rows.extend(summary_rows)
        all_pair_rows.extend(pair_rows)

        report["models"][model_name] = {
            "source_csv": str(path),
            "method_stats": method_stats,
            "paired_comparison": paired_report,
            "validation": {
                "formal_rows": len(formal_rows),
                "all_status_ok": True,
                "all_correctness_gates_pass": True,
                "all_schedule_gates_pass": True,
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output / "formal_summary.csv",
        all_summary_rows,
    )
    write_csv(
        args.output / "formal_pairs.csv",
        all_pair_rows,
    )

    with (args.output / "statistical_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"saved={args.output / 'formal_summary.csv'}")
    print(f"saved={args.output / 'formal_pairs.csv'}")
    print(f"saved={args.output / 'statistical_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())