from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "day15_final_analysis"
RUNS = (
    ("qwen7b", "formal", ROOT / "day13_context_8k_tmux_run1"),
    ("qwen3b", "formal", ROOT / "day14_qwen3b_context_8k_r1_run1"),
    ("qwen7b", "transfer_profile", ROOT / "day15_transfer_profile_qwen7b_r2_8k_tmux_run1"),
    ("qwen3b", "transfer_profile", ROOT / "day15_transfer_profile_qwen3b_r1_8k_tmux_run1"),
)
EXPECTED_METHODS = {
    "qwen7b": {"standard", "stagekv_bidirectional_r2", "stagekv_cross_layer_r2"},
    "qwen3b": {"standard", "stagekv_bidirectional_r1", "stagekv_cross_layer_r1"},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def b(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [f(row, key) for row in rows if row.get(key, "") != ""]
    return statistics.mean(values) if values else None


def audit(model: str, kind: str, directory: Path) -> dict[str, Any]:
    names = ("day12_manifest.json", "day12_warmup.csv", "day12_raw.csv", "day12_per_token.csv", "day12_summary.csv", "day12_comparison.csv")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"{directory}: missing {missing}")

    manifest = load_json(directory / "day12_manifest.json")
    warmup = load_csv(directory / "day12_warmup.csv")
    raw = load_csv(directory / "day12_raw.csv")
    per_token = load_csv(directory / "day12_per_token.csv")
    summary = load_csv(directory / "day12_summary.csv")
    comparison = load_csv(directory / "day12_comparison.csv")
    protocol = manifest["performance_claim_protocol"]
    methods = EXPECTED_METHODS[model]
    if {row["method"] for row in raw} != methods:
        raise RuntimeError(f"{directory}: unexpected methods")

    repeats = int(protocol["measured_repetitions"])
    warmups = int(protocol["warmup_rounds_excluded"])
    decode_tokens = int(protocol["decode_tokens"])
    gates: dict[str, bool] = {
        "files_present": not missing,
        "raw_rows_expected": len(raw) == repeats * len(methods),
        "warmup_rows_expected": len(warmup) == warmups * len(methods),
        "per_token_rows_expected": len(per_token) == repeats * len(methods) * decode_tokens,
        "all_status_ok": all(row["status"] == "ok" for row in raw),
        "all_generated_sequences_match": all(b(row["same_generated_sequence_as_standard"]) for row in raw),
        "all_step_top1_match": all(b(row["all_step_top1_equal_as_standard"]) for row in raw),
        "all_cache_ratios_one": all(math.isclose(f(row, "cache_ratio"), 1.0, abs_tol=1e-6) for row in raw),
        "no_blocking_d2h": all(int(row["blocking_d2h_tensor_copies"]) == 0 for row in raw if row["method"] != "standard"),
        "all_cross_layer_schedules_correct": all(b(row["cross_layer_schedule_correct"]) for row in raw if row["family"] == "cross_layer"),
    }
    if kind == "transfer_profile":
        gates.update({
            "transfer_event_timing_enabled": bool(manifest.get("transfer_event_timing_enabled")),
            "transfer_event_timing_complete": bool(manifest.get("validated_result_gates", {}).get("transfer_event_timing_complete")),
            "per_token_transfer_timing_explicitly_unavailable": manifest.get("transfer_event_timing", {}).get("per_token_transfer_event_timing_available") is False,
            "per_token_h2d_values_zero": all(math.isclose(float(row["h2d_event_ms"]), 0.0, abs_tol=1e-9) for row in per_token),
        })

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["method"]].append(row)
    order_spread: dict[str, float | None] = {}
    for method, rows in grouped.items():
        by_position: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_position[row["round_order_position"]].append(f(row, "decode_steady_cuda_mean_ms"))
        position_means = [statistics.mean(values) for values in by_position.values()]
        order_spread[method] = max(position_means) - min(position_means) if position_means else None

    summaries = {row["method"]: row for row in summary}
    candidate = {row["candidate_method"]: row for row in comparison if row.get("comparison") == "candidate_vs_standard"}
    cross = next((row for row in comparison if row.get("comparison") == "cross_layer_vs_bidirectional_same_residency"), None)
    methods_out = []
    for method in sorted(methods):
        row = summaries[method]
        candidate_row = candidate.get(method, {})
        methods_out.append({
            "model": model,
            "run_kind": kind,
            "method": method,
            "results_dir": str(directory),
            "successful_trials": int(row["successful_trials"]),
            "steady_cuda_ms_mean": f(row, "decode_steady_cuda_mean_ms_mean"),
            "steady_cuda_ms_std": f(row, "decode_steady_cuda_mean_ms_std"),
            "steady_cuda_p95_ms": f(row, "decode_steady_cuda_p95_ms_pooled"),
            "steady_speed_vs_standard": float(candidate_row["steady_decode_cuda_speed_ratio"]) if candidate_row else None,
            "gpu_kv_gib": f(row, "cache_gpu_gib_mean"),
            "cpu_kv_gib": f(row, "cache_cpu_gib_mean"),
            "h2d_gib": f(row, "h2d_gib_mean"),
            "d2h_gib": f(row, "d2h_gib_mean"),
            "h2d_event_total_ms": float(row.get("async_h2d_event_total_ms_mean", 0.0)),
            "d2h_event_total_ms": float(row.get("async_d2h_event_total_ms_mean", 0.0)),
            "execution_order_spread_ms": order_spread[method],
            "correctness": gates["all_generated_sequences_match"] and gates["all_step_top1_match"],
            "schedule_gate": gates["all_cross_layer_schedules_correct"],
        })

    cross_out = {
        "cross_layer_over_bidirectional_speed_ratio": None if cross is None else float(cross["cross_layer_over_bidirectional_speed_ratio"]),
        "meets_confirmatory_speed_target": False if cross is None else b(cross["meets_confirmatory_speed_target"]),
        "bidirectional_h2d_event_total_ms": None if cross is None else float(cross.get("bidirectional_h2d_event_total_ms", 0.0)),
        "cross_layer_h2d_event_total_ms": None if cross is None else float(cross.get("cross_layer_h2d_event_total_ms", 0.0)),
        "bidirectional_d2h_event_total_ms": None if cross is None else float(cross.get("bidirectional_d2h_event_total_ms", 0.0)),
        "cross_layer_d2h_event_total_ms": None if cross is None else float(cross.get("cross_layer_d2h_event_total_ms", 0.0)),
    }
    return {
        "model": model,
        "run_kind": kind,
        "results_dir": str(directory),
        "model_path": manifest.get("model_path"),
        "model_structure": manifest.get("model_structure"),
        "model_structure_metadata_present": manifest.get("model_structure") is not None,
        "paper_ready": bool(protocol.get("paper_ready")),
        "protocol": {"warmup_repeats": warmups, "measured_repeats": repeats, "decode_tokens": decode_tokens},
        "gates": gates,
        "cross_comparison": cross_out,
        "methods": methods_out,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        # A spreadsheet or IDE may have a derived CSV open. Preserve it rather
        # than touching user-owned output; the JSON/Markdown report is still
        # regenerated from the source experiment files.
        print(f"preserved_locked={path}")


def write_markdown(path: Path, reports: list[dict[str, Any]], method_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# StageKV Final 8K Analysis",
        "",
        "This report is derived from existing CSV/JSON results. Original experiment files are unchanged.",
        "",
        "## Method Results",
        "",
        "| Model | Run | Method | Trials | Steady CUDA ms | Std ms | GPU KV GiB | CPU KV GiB | H2D GiB | D2H GiB | H2D event ms | D2H event ms | Order spread ms | Correctness | Schedule |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in method_rows:
        lines.append(
            "| {model} | {run_kind} | {method} | {successful_trials} | "
            "{steady_cuda_ms_mean:.3f} | {steady_cuda_ms_std:.3f} | "
            "{gpu_kv_gib:.6f} | {cpu_kv_gib:.6f} | {h2d_gib:.6f} | "
            "{d2h_gib:.6f} | {h2d_event_total_ms:.3f} | "
            "{d2h_event_total_ms:.3f} | {execution_order_spread_ms:.3f} | "
            "{correctness} | {schedule_gate} |".format(**row)
        )
    lines.extend([
        "",
        "## Cross-Layer Comparison",
        "",
        "| Model | Run | Cross/Bidirectional | Target 1.05x | Bidirectional H2D event ms | Cross H2D event ms | Bidirectional D2H event ms | Cross D2H event ms |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for report in reports:
        cross = report["cross_comparison"]
        lines.append(
            f"| {report['model']} | {report['run_kind']} | "
            f"{cross['cross_layer_over_bidirectional_speed_ratio']:.6f} | "
            f"{cross['meets_confirmatory_speed_target']} | "
            f"{cross['bidirectional_h2d_event_total_ms']:.3f} | "
            f"{cross['cross_layer_h2d_event_total_ms']:.3f} | "
            f"{cross['bidirectional_d2h_event_total_ms']:.3f} | "
            f"{cross['cross_layer_d2h_event_total_ms']:.3f} |"
        )
    lines.extend([
        "",
        "## Gate Summary",
        "",
    ])
    for report in reports:
        gate_values = report["gates"]
        lines.append(
            f"- `{report['model']}/{report['run_kind']}`: "
            f"all gates pass = `{all(gate_values.values())}`, "
            f"structure metadata present = `{report['model_structure_metadata_present']}`, "
            f"paper_ready = `{report['paper_ready']}`"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "> Freeze the cross-layer acceleration claim. Report StageKV as a persistent GPU-KV memory reduction with a latency and transfer-cost tradeoff; `paper_ready` is protocol status only and does not imply performance success.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    reports = [audit(model, kind, directory) for model, kind, directory in RUNS]
    formal = [report for report in reports if report["run_kind"] == "formal"]
    formal_targets = [report["cross_comparison"]["meets_confirmatory_speed_target"] for report in formal]
    all_gates = all(all(report["gates"].values()) for report in reports)
    final = {
        "analysis": "StageKV final multi-model 8K audit",
        "source_runs": [report["results_dir"] for report in reports],
        "all_behavioral_and_protocol_gates_pass": all_gates,
        "model_structure_metadata_complete": all(
            report["model_structure_metadata_present"] for report in reports
        ),
        "formal_speed_targets_all_met": all(formal_targets),
        "decision": "freeze_cross_layer_acceleration_claim_and_report_memory_latency_tradeoff" if all_gates and not all(formal_targets) else "manual_review_required",
        "paper_ready_is_protocol_only": True,
        "runs": reports,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final_analysis.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    method_rows = [row for report in reports for row in report["methods"]]
    write_csv(OUTPUT_DIR / "final_methods.csv", method_rows)
    write_csv(OUTPUT_DIR / "final_cross_comparisons.csv", [{"model": report["model"], "run_kind": report["run_kind"], **report["cross_comparison"]} for report in reports])
    write_markdown(OUTPUT_DIR / "final_report.md", reports, method_rows)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"final_analysis={OUTPUT_DIR / 'final_analysis.json'}")
    print(f"final_methods={OUTPUT_DIR / 'final_methods.csv'}")
    print(f"final_cross_comparisons={OUTPUT_DIR / 'final_cross_comparisons.csv'}")
    print(f"final_report={OUTPUT_DIR / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
