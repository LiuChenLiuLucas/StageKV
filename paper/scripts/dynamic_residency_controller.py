from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import headinfer_reference_benchmark as base
from budgeted_headinfer_benchmark import (
    BudgetedResidentHeadInferCache,
)
from headinfer.mp import mp_headinfer


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_sequence(report: dict) -> list[int]:
    sequences = {
        tuple(json.loads(row["generated_token_ids"]))
        for row in report["rows"]
    }
    if len(sequences) != 1:
        raise RuntimeError("Standard reference sequence is unstable")
    return list(next(iter(sequences)))


def mean_transfer(report: dict) -> float:
    return statistics.mean(
        float(row["h2d_gib"]) + float(row["d2h_gib"])
        for row in report["rows"]
    )


def summarize(
    rows: list[dict],
    reference: list[int],
    target_transfer: float,
    gpu_budget: float,
) -> dict:
    sequences = [
        json.loads(row["generated_token_ids"])
        for row in rows
    ]

    correct = (
        len({tuple(sequence) for sequence in sequences}) == 1
        and all(sequence == reference for sequence in sequences)
    )

    transfer = statistics.mean(
        float(row["h2d_gib"]) + float(row["d2h_gib"])
        for row in rows
    )
    gpu_kv = statistics.mean(
        float(row["cache_gpu_gib"])
        for row in rows
    )
    latency = statistics.mean(
        float(row["decode_end_to_end_ms_per_token"])
        for row in rows
    )

    return {
        "resident_head_blocks": int(
            rows[0]["resident_head_blocks"]
        ),
        "correct": correct,
        "gpu_kv_gib": gpu_kv,
        "total_transfer_gib": transfer,
        "decode_ms_per_token": latency,
        "transfer_target_pass": transfer <= target_transfer,
        "gpu_budget_pass": gpu_kv <= gpu_budget,
        "feasible": (
            correct
            and transfer <= target_transfer
            and gpu_kv <= gpu_budget
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


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
        "--target-transfer-gib",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--candidate-max",
        type=int,
        default=None,
    )
    parser.add_argument("--sweep-warmup", type=int, default=1)
    parser.add_argument("--sweep-repeats", type=int, default=3)
    parser.add_argument("--final-warmup", type=int, default=2)
    parser.add_argument("--final-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise RuntimeError(
            "This correctness controller requires "
            "CUDA_LAUNCH_BLOCKING=1"
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    standard_report = load_report(args.standard_report)
    headinfer_report = load_report(args.headinfer_report)

    reference = reference_sequence(standard_report)
    baseline_transfer = mean_transfer(headinfer_report)

    target_transfer = (
        args.target_transfer_gib
        if args.target_transfer_gib is not None
        else baseline_transfer * args.target_fraction
    )

    config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    query_heads = int(config.num_attention_heads)
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // query_heads
    )
    total_head_blocks = layers * kv_heads

    max_cache_length = (
        args.sequence_length + args.decode_tokens - 1
    )
    block_bytes = (
        2 * max_cache_length * head_dim * 2
    )
    block_gib = block_bytes / 1024**3

    estimated_max = min(
        total_head_blocks,
        math.floor(
            args.gpu_kv_budget_gib / block_gib
        ) + 2,
    )
    candidate_max = (
        args.candidate_max
        if args.candidate_max is not None
        else estimated_max
    )
    candidate_max = min(candidate_max, total_head_blocks)

    print(f"total_head_blocks={total_head_blocks}")
    print(f"single_head_block_gib={block_gib:.9f}")
    print(f"gpu_kv_budget_gib={args.gpu_kv_budget_gib}")
    print(f"baseline_transfer_gib={baseline_transfer:.6f}")
    print(f"target_transfer_gib={target_transfer:.6f}")
    print(f"candidate_range=0..{candidate_max}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()

    mp_headinfer(model)

    state = {"resident_head_blocks": 0}

    def build_cache(method: str):
        if method != "resident":
            raise ValueError(f"Unexpected method: {method}")

        return BudgetedResidentHeadInferCache(
            total_head_blocks=total_head_blocks,
            resident_head_blocks=state[
                "resident_head_blocks"
            ],
        )

    base.build_cache = build_cache

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    input_ids = torch.randint(
        low=10,
        high=int(config.vocab_size) - 10,
        size=(1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to("cuda")

    def execute(
        resident: int,
        warmups: int,
        repeats: int,
    ) -> list[dict]:
        state["resident_head_blocks"] = resident

        for _ in range(warmups):
            base.run_once(
                model,
                method="resident",
                input_ids=input_ids,
                decode_tokens=args.decode_tokens,
                layers=layers,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )

        rows = []

        for trial in range(1, repeats + 1):
            row = base.run_once(
                model,
                method="resident",
                input_ids=input_ids,
                decode_tokens=args.decode_tokens,
                layers=layers,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            row["resident_head_blocks"] = resident
            row["trial"] = trial
            rows.append(row)

        return rows

    sweep = []

    for resident in range(candidate_max + 1):
        rows = execute(
            resident,
            args.sweep_warmup,
            args.sweep_repeats,
        )
        result = summarize(
            rows,
            reference,
            target_transfer,
            args.gpu_kv_budget_gib,
        )
        sweep.append(result)

        print(
            f"r={resident:02d} "
            f"gpu={result['gpu_kv_gib']:.6f} "
            f"transfer={result['total_transfer_gib']:.6f} "
            f"correct={result['correct']} "
            f"feasible={result['feasible']}"
        )

    feasible = [
        result
        for result in sweep
        if result["feasible"]
    ]

    if not feasible:
        report = {
            "status": "NO_FEASIBLE_CONFIGURATION",
            "gpu_kv_budget_gib": args.gpu_kv_budget_gib,
            "target_transfer_gib": target_transfer,
            "sweep": sweep,
        }
        path = args.results_dir / "controller_report.json"
        path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("No feasible resident count found")

    selected = min(
        feasible,
        key=lambda result: (
            result["resident_head_blocks"],
            result["gpu_kv_gib"],
            result["decode_ms_per_token"],
        ),
    )
    selected_r = selected["resident_head_blocks"]

    final_rows = execute(
        selected_r,
        args.final_warmup,
        args.final_repeats,
    )
    final_result = summarize(
        final_rows,
        reference,
        target_transfer,
        args.gpu_kv_budget_gib,
    )

    smaller_feasible = [
        result
        for result in sweep
        if (
            result["resident_head_blocks"] < selected_r
            and result["feasible"]
        )
    ]

    report = {
        "status": (
            "PASS"
            if final_result["feasible"]
            and not smaller_feasible
            else "FAIL"
        ),
        "model_path": args.model_path,
        "sequence_length": args.sequence_length,
        "total_head_blocks": total_head_blocks,
        "gpu_kv_budget_gib": args.gpu_kv_budget_gib,
        "headinfer_baseline_transfer_gib": (
            baseline_transfer
        ),
        "target_transfer_gib": target_transfer,
        "selected_resident_head_blocks": selected_r,
        "selected_is_minimal_tested": (
            not smaller_feasible
        ),
        "final_result": final_result,
        "sweep": sweep,
        "final_rows": final_rows,
    }

    write_csv(
        args.results_dir / "controller_sweep.csv",
        sweep,
    )
    write_csv(
        args.results_dir / "controller_final_raw.csv",
        final_rows,
    )
    (
        args.results_dir / "controller_report.json"
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("dynamic_residency_controller=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())