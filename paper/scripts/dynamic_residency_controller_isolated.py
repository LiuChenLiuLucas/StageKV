from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def parse_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list):
        raise TypeError(
            f"Invalid generated_token_ids type: {type(value)}"
        )

    return tuple(int(token) for token in value)


def get_raw_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows")

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Report does not contain a non-empty rows list")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{index}] is not a dictionary")

    return rows


def get_sequences(report: dict[str, Any]) -> list[tuple[int, ...]]:
    rows = get_raw_rows(report)

    sequences = []

    for index, row in enumerate(rows):
        if "generated_token_ids" not in row:
            raise KeyError(
                f"rows[{index}] has no generated_token_ids"
            )

        sequences.append(
            parse_token_ids(row["generated_token_ids"])
        )

    return sequences


def mean_column(
    rows: list[dict[str, Any]],
    key: str,
) -> float:
    values = []

    for index, row in enumerate(rows):
        if key not in row:
            raise KeyError(f"rows[{index}] has no {key}")

        values.append(float(row[key]))

    return statistics.mean(values)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_reference_report(
    report: dict[str, Any],
    *,
    name: str,
    sequence_length: int,
    decode_tokens: int,
    seed: int,
) -> tuple[int, ...]:
    observed_length = int(report.get("sequence_length", -1))
    observed_decode = int(report.get("decode_tokens", -1))
    observed_seed = int(report.get("seed", -1))

    if observed_length != sequence_length:
        raise RuntimeError(
            f"{name} sequence length mismatch: "
            f"expected={sequence_length}, observed={observed_length}"
        )

    if observed_decode != decode_tokens:
        raise RuntimeError(
            f"{name} decode token mismatch: "
            f"expected={decode_tokens}, observed={observed_decode}"
        )

    if observed_seed != seed:
        raise RuntimeError(
            f"{name} seed mismatch: "
            f"expected={seed}, observed={observed_seed}"
        )

    sequences = get_sequences(report)

    if len(set(sequences)) != 1:
        raise RuntimeError(f"{name} generated sequence is unstable")

    return sequences[0]


def run_worker(
    *,
    worker_script: Path,
    model_path: str,
    resident_head_blocks: int,
    sequence_length: int,
    decode_tokens: int,
    warmup_repeats: int,
    repeats: int,
    seed: int,
    result_dir: Path,
) -> tuple[dict[str, Any], int]:
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(
            f"Worker directory is not empty: {result_dir}. "
            "Use a new controller results directory."
        )

    result_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(worker_script),
        "--model-path",
        model_path,
        "--method",
        "resident",
        "--resident-head-blocks",
        str(resident_head_blocks),
        "--sequence-length",
        str(sequence_length),
        "--decode-tokens",
        str(decode_tokens),
        "--warmup-repeats",
        str(warmup_repeats),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--results-dir",
        str(result_dir),
    ]

    environment = os.environ.copy()
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault(
        "PYTHONPATH",
        "/root/headinfer/headinfer-main",
    )

    process = subprocess.run(
        command,
        cwd=str(worker_script.parent),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    (result_dir / "worker_stdout.txt").write_text(
        process.stdout,
        encoding="utf-8",
    )
    (result_dir / "worker_stderr.txt").write_text(
        process.stderr,
        encoding="utf-8",
    )

    report_path = result_dir / (
        "headinfer_reference_resident_"
        f"{sequence_length}_report.json"
    )

    if not report_path.is_file():
        stdout_tail = process.stdout[-3000:]
        stderr_tail = process.stderr[-3000:]

        raise RuntimeError(
            "Resident worker did not produce a report.\n"
            f"resident_head_blocks={resident_head_blocks}\n"
            f"exit_code={process.returncode}\n"
            f"stdout_tail:\n{stdout_tail}\n"
            f"stderr_tail:\n{stderr_tail}"
        )

    return load_json(report_path), process.returncode


def evaluate_candidate(
    report: dict[str, Any],
    *,
    reference_sequence: tuple[int, ...],
    resident_head_blocks: int,
    worker_exit_code: int,
    worker_dir: Path,
) -> dict[str, Any]:
    raw_rows = get_raw_rows(report)
    sequences = get_sequences(report)

    stable = len(set(sequences)) == 1

    matches_standard = all(
        sequence == reference_sequence
        for sequence in sequences
    )

    worker_completed = worker_exit_code == 0

    correct = (
        worker_completed
        and stable
        and matches_standard
    )

    h2d_gib = mean_column(raw_rows, "h2d_gib")
    d2h_gib = mean_column(raw_rows, "d2h_gib")

    return {
        "resident_head_blocks": resident_head_blocks,
        "worker_exit_code": worker_exit_code,
        "worker_completed": worker_completed,
        "worker_dir": str(worker_dir),
        "stable": stable,
        "matches_standard": matches_standard,
        "correct": correct,
        "gpu_kv_gib": mean_column(
            raw_rows,
            "cache_gpu_gib",
        ),
        "h2d_gib": h2d_gib,
        "d2h_gib": d2h_gib,
        "total_transfer_gib": h2d_gib + d2h_gib,
        "decode_ms_per_token": mean_column(
            raw_rows,
            "decode_end_to_end_ms_per_token",
        ),
    }


def add_constraints(
    result: dict[str, Any],
    *,
    target_transfer_gib: float,
    gpu_kv_budget_gib: float,
) -> None:
    result["transfer_target_pass"] = (
        result["total_transfer_gib"]
        <= target_transfer_gib
    )
    result["gpu_budget_pass"] = (
        result["gpu_kv_gib"]
        <= gpu_kv_budget_gib
    )
    result["feasible"] = (
        result["correct"]
        and result["transfer_target_pass"]
        and result["gpu_budget_pass"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--standard-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--headinfer-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--worker-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--gpu-kv-budget-gib",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--candidate-max",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--sweep-warmup",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--final-warmup",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise RuntimeError(
            "CUDA_LAUNCH_BLOCKING must be set to 1"
        )

    if not 0.0 < args.target_fraction <= 1.0:
        raise ValueError(
            "target-fraction must be within (0, 1]"
        )

    if args.candidate_max < 0:
        raise ValueError(
            "candidate-max must be non-negative"
        )

    if not args.worker_script.is_file():
        raise FileNotFoundError(
            f"Worker script not found: {args.worker_script}"
        )

    if (
        args.results_dir.exists()
        and any(args.results_dir.iterdir())
    ):
        raise RuntimeError(
            f"Results directory is not empty: {args.results_dir}. "
            "Use a new directory to avoid overwriting results."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    standard_report = load_json(args.standard_report)
    headinfer_report = load_json(args.headinfer_report)

    standard_sequence = validate_reference_report(
        standard_report,
        name="Standard",
        sequence_length=args.sequence_length,
        decode_tokens=args.decode_tokens,
        seed=args.seed,
    )

    headinfer_sequence = validate_reference_report(
        headinfer_report,
        name="HeadInfer",
        sequence_length=args.sequence_length,
        decode_tokens=args.decode_tokens,
        seed=args.seed,
    )

    if standard_sequence != headinfer_sequence:
        raise RuntimeError(
            "HeadInfer reference does not match Standard"
        )

    headinfer_rows = get_raw_rows(headinfer_report)

    baseline_h2d = mean_column(
        headinfer_rows,
        "h2d_gib",
    )
    baseline_d2h = mean_column(
        headinfer_rows,
        "d2h_gib",
    )
    baseline_transfer = baseline_h2d + baseline_d2h

    target_transfer = (
        baseline_transfer * args.target_fraction
    )

    print(f"baseline_transfer_gib={baseline_transfer:.6f}")
    print(f"target_transfer_gib={target_transfer:.6f}")
    print(f"gpu_kv_budget_gib={args.gpu_kv_budget_gib:.6f}")
    print(f"candidate_range=0..{args.candidate_max}")

    sweep: list[dict[str, Any]] = []
    sweep_path = args.results_dir / "controller_sweep.csv"

    for resident in range(args.candidate_max + 1):
        candidate_dir = (
            args.results_dir
            / f"candidate_r{resident:03d}"
        )

        print(
            f"running candidate r={resident}",
            flush=True,
        )

        worker_report, exit_code = run_worker(
            worker_script=args.worker_script,
            model_path=args.model_path,
            resident_head_blocks=resident,
            sequence_length=args.sequence_length,
            decode_tokens=args.decode_tokens,
            warmup_repeats=args.sweep_warmup,
            repeats=args.sweep_repeats,
            seed=args.seed,
            result_dir=candidate_dir,
        )

        result = evaluate_candidate(
            worker_report,
            reference_sequence=standard_sequence,
            resident_head_blocks=resident,
            worker_exit_code=exit_code,
            worker_dir=candidate_dir,
        )

        add_constraints(
            result,
            target_transfer_gib=target_transfer,
            gpu_kv_budget_gib=args.gpu_kv_budget_gib,
        )

        sweep.append(result)
        write_csv(sweep_path, sweep)

        print(
            f"r={resident:02d} "
            f"gpu={result['gpu_kv_gib']:.6f} "
            f"transfer={result['total_transfer_gib']:.6f} "
            f"stable={result['stable']} "
            f"matches_standard={result['matches_standard']} "
            f"correct={result['correct']} "
            f"feasible={result['feasible']}",
            flush=True,
        )

    feasible_candidates = [
        result
        for result in sweep
        if result["feasible"]
    ]

    report_path = (
        args.results_dir / "controller_report.json"
    )

    if not feasible_candidates:
        controller_report = {
            "status": "NO_FEASIBLE_CONFIGURATION",
            "model_path": args.model_path,
            "sequence_length": args.sequence_length,
            "decode_tokens": args.decode_tokens,
            "seed": args.seed,
            "gpu_kv_budget_gib": args.gpu_kv_budget_gib,
            "target_fraction": args.target_fraction,
            "headinfer_baseline_transfer_gib": (
                baseline_transfer
            ),
            "target_transfer_gib": target_transfer,
            "sweep": sweep,
        }

        report_path.write_text(
            json.dumps(
                controller_report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("controller_status=NO_FEASIBLE_CONFIGURATION")
        return 1

    selected = min(
        feasible_candidates,
        key=lambda result: result["resident_head_blocks"],
    )
    selected_r = int(selected["resident_head_blocks"])

    final_dir = (
        args.results_dir
        / f"selected_final_r{selected_r:03d}"
    )

    print(
        f"running final verification r={selected_r}",
        flush=True,
    )

    final_report, final_exit_code = run_worker(
        worker_script=args.worker_script,
        model_path=args.model_path,
        resident_head_blocks=selected_r,
        sequence_length=args.sequence_length,
        decode_tokens=args.decode_tokens,
        warmup_repeats=args.final_warmup,
        repeats=args.final_repeats,
        seed=args.seed,
        result_dir=final_dir,
    )

    final_result = evaluate_candidate(
        final_report,
        reference_sequence=standard_sequence,
        resident_head_blocks=selected_r,
        worker_exit_code=final_exit_code,
        worker_dir=final_dir,
    )

    add_constraints(
        final_result,
        target_transfer_gib=target_transfer,
        gpu_kv_budget_gib=args.gpu_kv_budget_gib,
    )

    smaller_feasible = [
        result
        for result in sweep
        if (
            result["resident_head_blocks"] < selected_r
            and result["feasible"]
        )
    ]

    selected_is_minimal = not smaller_feasible

    passed = (
        final_result["feasible"]
        and selected_is_minimal
    )

    status = (
        "PASS_MINIMAL_FEASIBLE"
        if passed
        else "FAIL_FINAL_RECHECK"
    )

    controller_report = {
        "status": status,
        "model_path": args.model_path,
        "sequence_length": args.sequence_length,
        "decode_tokens": args.decode_tokens,
        "seed": args.seed,
        "gpu_kv_budget_gib": args.gpu_kv_budget_gib,
        "target_fraction": args.target_fraction,
        "headinfer_baseline_transfer_gib": baseline_transfer,
        "target_transfer_gib": target_transfer,
        "selected_resident_head_blocks": selected_r,
        "selected_is_minimal_tested": selected_is_minimal,
        "final_result": final_result,
        "sweep": sweep,
    }

    report_path.write_text(
        json.dumps(
            controller_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": status,
                "selected_resident_head_blocks": selected_r,
                "selected_is_minimal_tested": selected_is_minimal,
                "final_correct": final_result["correct"],
                "final_gpu_kv_gib": final_result["gpu_kv_gib"],
                "final_transfer_gib": (
                    final_result["total_transfer_gib"]
                ),
                "final_feasible": final_result["feasible"],
            },
            indent=2,
        )
    )

    print(f"sweep_csv={sweep_path}")
    print(f"report_json={report_path}")
    print(f"dynamic_residency_controller={status}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())