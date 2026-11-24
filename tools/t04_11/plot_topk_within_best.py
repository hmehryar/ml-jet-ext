#!/usr/bin/env python3
"""
Create a manuscript-ready plot comparing the best joint model
and the best expert composition on joint Top-k and Within-1 metrics.

Input:
    results/t04_11/evaluation/val/summary_mean_sd.csv

Outputs:
    results/t04_11/final/validation_only/figures/best_joint_vs_expert_topk_within.pdf
    results/t04_11/final/validation_only/figures/best_joint_vs_expert_topk_within.png
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


INPUT_CSV = Path("results/t04_11/evaluation/val/summary_mean_sd.csv")
OUTPUT_DIR = Path("results/t04_11/final/validation_only/figures")
OUTPUT_BASENAME = "best_joint_vs_expert_topk_within"

BEST_JOINT = "CLIP joint"
BEST_EXPERT = "CLIP α + CLIP Q0"

# Metrics to plot: (csv field prefix, display label)
METRICS = [
    ("joint_accuracy", "Top-1"),
    ("joint_top3", "Top-3"),
    ("joint_top5", "Top-5"),
    ("joint_within1", "Within-1"),
]


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_system_row(rows, system_name: str):
    for row in rows:
        if row["system"] == system_name:
            return row
    available = [r["system"] for r in rows]
    raise ValueError(
        f"System '{system_name}' not found in {path}. "
        f"Available systems: {available}"
    )


def extract_metric_values(row, metrics):
    means = []
    sds = []
    for key, _label in metrics:
        means.append(float(row[f"{key}_mean"]) * 100.0)
        sds.append(float(row[f"{key}_sd"]) * 100.0)
    return np.array(means), np.array(sds)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(INPUT_CSV)

    joint_row = get_system_row(rows, BEST_JOINT)
    expert_row = get_system_row(rows, BEST_EXPERT)

    joint_means, joint_sds = extract_metric_values(joint_row, METRICS)
    expert_means, expert_sds = extract_metric_values(expert_row, METRICS)

    labels = [label for _key, label in METRICS]
    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    ax.bar(
        x - width / 2,
        joint_means,
        width,
        yerr=joint_sds,
        capsize=4,
        label=BEST_JOINT,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.bar(
        x + width / 2,
        expert_means,
        width,
        yerr=expert_sds,
        capsize=4,
        label=BEST_EXPERT,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 75)

    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()