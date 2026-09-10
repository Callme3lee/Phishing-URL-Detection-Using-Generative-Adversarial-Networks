from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json


METRICS = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-Score"),
    ("roc_auc", "ROC-AUC"),
)
DOSES = (10, 20, 30)
LABELS = ("Real only", "10% + real FT", "20% + real FT", "30% + real FT")
COLORS = ("#355C7D", "#62A8AC", "#4C9F70", "#E05D5D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot every CNN fine-tuning metric in separate comparison panels."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/robust_augmentation/finetuned/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/robust_augmentation/finetuned/all_metrics_comparison"),
    )
    return parser.parse_args()


def series(payload: dict, evaluation: str, metric: str) -> tuple[list[float], list[float]]:
    means = [100 * payload["real_only"][evaluation][metric]["mean"]]
    stds = [100 * payload["real_only"][evaluation][metric]["std"]]
    for dose in DOSES:
        item = payload["doses"][str(dose)][evaluation][metric]
        means.append(100 * item["mean"])
        stds.append(100 * item["std"])
    return means, stds


def zoom_limits(values: list[float], stds: list[float]) -> tuple[float, float]:
    low = min(value - std for value, std in zip(values, stds))
    high = max(value + std for value, std in zip(values, stds))
    spread = max(high - low, 0.12)
    padding = max(0.18, spread * 0.45)
    return max(0.0, low - padding), min(100.0, high + padding)


def main() -> None:
    args = parse_args()
    payload = load_json(args.summary)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 5, figsize=(25, 11))
    figure.subplots_adjust(
        left=0.075, right=0.99, bottom=0.12, top=0.86, wspace=0.24, hspace=0.48
    )
    rows = (
        ("balanced_test", "Balanced hostname-held-out test"),
        ("source_macro", "Source-macro held-out evaluation"),
    )
    x = np.arange(len(LABELS))
    for row_index, (evaluation, row_title) in enumerate(rows):
        for column_index, (metric, title) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            means, stds = series(payload, evaluation, metric)
            bars = axis.bar(
                x,
                means,
                yerr=stds,
                capsize=4,
                color=COLORS,
                edgecolor="white",
                linewidth=0.8,
                error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
            )
            axis.set_ylim(*zoom_limits(means, stds))
            axis.set_title(title, fontsize=15, fontweight="bold", pad=10)
            axis.set_xticks(x, LABELS, rotation=28, ha="right", fontsize=10)
            axis.tick_params(axis="y", labelsize=10)
            axis.set_ylabel("Mean score (%)" if column_index == 0 else "", fontsize=11)
            axis.grid(axis="y", alpha=0.28)
            axis.grid(axis="x", visible=False)
            for bar, value in zip(bars, means):
                axis.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="semibold",
                )
        axes[row_index, 0].text(
            -0.32,
            0.5,
            row_title,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=15,
            fontweight="bold",
            color="#263238",
        )

    figure.suptitle(
        "Real-Only CNN vs Quality-LSGAN Pretraining + Real-Only Fine-Tuning",
        fontsize=23,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.935,
        "Mean across seeds 42, 43, and 44  •  Error bars: ±1 standard deviation  •  Zoomed y-axes show small differences",
        ha="center",
        fontsize=13,
        color="#455A64",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
