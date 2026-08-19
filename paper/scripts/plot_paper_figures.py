from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


T95_DF4 = 2.7764451052


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save_pipeline_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.4, 2.0, 2.2, 1.0, "CPU pinned KV\nhistorical cache"),
        (3.4, 2.0, 2.2, 1.0, "Async H2D\nKV-group transfer"),
        (6.4, 2.0, 2.2, 1.0, "GPU staging\nbuffer"),
        (9.4, 2.0, 2.2, 1.0, "Attention\ncomputation"),
    ]

    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor="#1f4e79",
            facecolor="#d9eaf7",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=11,
        )

    for start, end in [(2.6, 3.4), (5.6, 6.4), (8.6, 9.4)]:
        ax.annotate(
            "",
            xy=(end, 2.5),
            xytext=(start, 2.5),
            arrowprops={"arrowstyle": "->", "lw": 1.8},
        )

    ax.annotate(
        "Cross-layer lookahead: prefetch L+1 while computing L",
        xy=(7.5, 3.2),
        xytext=(5.8, 4.3),
        ha="center",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#b04a4a"},
        color="#8f2d2d",
    )

    ax.set_title(
        "StageKV execution path: resident KV heads plus CPU offload",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / "fig1_stagekv_pipeline.png", dpi=220)
    plt.close(fig)


def save_memory_latency_figure(
    summary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    colors = {
        "standard": "#4c78a8",
        "bidirectional": "#f58518",
        "cross_layer": "#54a24b",
    }
    labels = {
        "standard": "Standard",
        "bidirectional": "Bidirectional",
        "cross_layer": "Cross-layer",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for row in summary_rows:
        method = row["method"]
        model = "7B" if "7B" in row["model"] else "3B"
        x = f(row, "gpu_kv_gib")
        y = f(row, "mean_ms")

        ax.scatter(
            x,
            y,
            s=100,
            color=colors[method],
            label=f"{model} {labels[method]}",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Persistent GPU KV cache (GiB)")
    ax.set_ylabel("Steady decode latency (ms/token, log scale)")
    ax.set_title("GPU KV memory versus decode latency")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output / "fig2_memory_latency_tradeoff.png", dpi=220)
    plt.close(fig)


def save_ratio_figure(
    pair_rows: list[dict[str, str]],
    output: Path,
) -> None:
    models = ["Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct"]
    means = []
    errors = []

    for model in models:
        values = [
            f(row, "bidirectional_over_cross_latency_ratio")
            for row in pair_rows
            if row["model"] == model
        ]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        half = T95_DF4 * std / math.sqrt(len(values))
        means.append(mean)
        errors.append(half)

    x = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        x,
        means,
        yerr=errors,
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=1.5,
        color="#2f5597",
    )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axhline(
        1.05,
        color="#b04a4a",
        linestyle=":",
        linewidth=1.5,
        label="Target = 1.05x",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Qwen2.5-7B", "Qwen2.5-3B"])
    ax.set_ylabel("Bidirectional latency / Cross-layer latency")
    ax.set_title("Paired Cross-layer comparison with 95% CI")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig3_cross_layer_ratio_ci.png", dpi=220)
    plt.close(fig)


def save_transfer_figure(
    cross_rows: list[dict[str, str]],
    output: Path,
) -> None:
    rows = [
        row for row in cross_rows
        if row["run_kind"] == "transfer_profile"
    ]

    labels = []
    h2d_values = []
    d2h_values = []

    for row in rows:
        model = "7B" if row["model"] == "qwen7b" else "3B"

        labels.extend([
            f"{model}\nBidirectional",
            f"{model}\nCross-layer",
        ])

        h2d_values.extend([
            float(row["bidirectional_h2d_event_total_ms"]),
            float(row["cross_layer_h2d_event_total_ms"]),
        ])

        d2h_values.extend([
            float(row["bidirectional_d2h_event_total_ms"]),
            float(row["cross_layer_d2h_event_total_ms"]),
        ])

    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        [value - width / 2 for value in x],
        h2d_values,
        width,
        label="H2D event total",
        color="#4c78a8",
    )
    ax.bar(
        [value + width / 2 for value in x],
        d2h_values,
        width,
        label="D2H event total",
        color="#f58518",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CUDA event total (ms, log scale)")
    ax.set_title("Transfer profile diagnostic, one measured run")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fig4_transfer_profile.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary)
    pair_rows = read_csv(args.pairs)
    cross_rows = read_csv(args.cross)

    save_pipeline_figure(args.output)
    save_memory_latency_figure(summary_rows, args.output)
    save_ratio_figure(pair_rows, args.output)
    save_transfer_figure(cross_rows, args.output)

    print(f"saved figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())