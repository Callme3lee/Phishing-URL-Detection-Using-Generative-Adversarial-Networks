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
MODELS = (
    ("cnn", "CNN\nAug-20% + FT"),
    ("lstm", "LSTM\nReal only"),
    ("rf", "Random Forest\nReal only"),
    ("svm", "Linear SVM\nReal only"),
)
COLORS = ("#2E8B57", "#3B6EA8", "#E39C37", "#8C6BB1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the selected augmented CNN with LSTM, RF, and SVM."
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/robust_augmentation")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/robust_augmentation/finetuned/best_models_all_metrics"),
    )
    return parser.parse_args()


def selected_metric(
    model: str,
    evaluation: str,
    metric: str,
    robust: dict,
    external: dict,
    finetuned: dict,
) -> dict:
    if model == "cnn":
        return finetuned["doses"]["20"][evaluation][metric]
    if evaluation == "balanced_test":
        return robust["metrics"][model]["real_only"][metric]
    return external["metrics"][model]["real_only"]["source_macro"][metric]


def limits(means: list[float], stds: list[float]) -> tuple[float, float]:
    low = min(value - std for value, std in zip(means, stds))
    high = max(value + std for value, std in zip(means, stds))
    spread = max(high - low, 0.5)
    padding = max(0.5, spread * 0.18)
    return max(0, low - padding), min(100, high + padding)


def main() -> None:
    args = parse_args()
    robust = load_json(args.experiment_root / "three_seed_summary.json")
    external = load_json(
        args.experiment_root / "calibrated_external_three_seed_summary.json"
    )
    finetuned = load_json(args.experiment_root / "finetuned" / "summary.json")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 5, figsize=(25, 11))
    figure.subplots_adjust(
        left=0.075, right=0.99, bottom=0.12, top=0.86, wspace=0.24, hspace=0.48
    )
    rows = (
        ("balanced_test", "Balanced hostname-held-out test"),
        ("source_macro", "Source-macro held-out evaluation"),
    )
    x = np.arange(len(MODELS))
    labels = [label for _, label in MODELS]
    for row_index, (evaluation, row_title) in enumerate(rows):
        for column_index, (metric, title) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            items = [
                selected_metric(
                    model, evaluation, metric, robust, external, finetuned
                )
                for model, _ in MODELS
            ]
            means = [100 * item["mean"] for item in items]
            stds = [100 * item["std"] for item in items]
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
            axis.set_ylim(*limits(means, stds))
            axis.set_title(title, fontsize=15, fontweight="bold", pad=10)
            axis.set_xticks(x, labels, fontsize=9)
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
        "Selected CNN, LSTM, Random Forest, and SVM: All-Metric Comparison",
        fontsize=23,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.935,
        "CNN uses quality-LSGAN 20% pretraining + real-only fine-tuning; other models use their source-macro-F1-selected real-only condition",
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
