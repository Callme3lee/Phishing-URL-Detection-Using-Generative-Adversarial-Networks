from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (("cnn", "CNN"), ("lstm", "LSTM"), ("rf", "RF"), ("svm", "SVM"))
CONDITIONS = (
    ("real_only_reference_69300", "Real only", "#345995"),
    ("lsgan_40_percent", "LSGAN 40%", "#2A9D8F"),
    ("wgan_gp_40_percent", "WGAN-GP 40%", "#7E57C2"),
    ("combined_gans_40_percent", "Combined 40%", "#F28E2B"),
)
SOURCES = (("new_data_urls", "new_data_urls"), ("url_dataset", "URL dataset"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GAN-unseen external results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/augmented_experiments/external_heldout_summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))["rows"]
    lookup = {
        (row["source"], row["model"], row["condition"]): row for row in rows
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=180)
    x = np.arange(len(MODELS))
    width = 0.19
    for row_index, (source, source_label) in enumerate(SOURCES):
        for column_index, (metric, metric_label) in enumerate(
            (("f1", "F1 score (%)"), ("roc_auc", "ROC-AUC (%)"))
        ):
            ax = axes[row_index, column_index]
            for condition_index, (condition, label, color) in enumerate(CONDITIONS):
                values = [
                    100 * lookup[(source, model, condition)][metric]
                    for model, _ in MODELS
                ]
                offset = (condition_index - 1.5) * width
                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
                ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5, rotation=90)
            ax.set_xticks(x, [label for _, label in MODELS])
            ax.set_ylabel(metric_label)
            ax.set_ylim(0, 100)
            ax.set_title(f"{source_label}: {metric_label}", loc="left")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, ncols=2, loc="upper left")
    figure.suptitle(
        "GAN-Unseen External Evaluation: Frozen Thresholds",
        fontsize=17,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.055,
        0.022,
        "Held-out hostnames were excluded from GAN and classifier training. High F1 with very low ROC-AUC on URL dataset indicates "
        "near-all-malicious prediction, not reliable discrimination. Full 20/30/40% results are in the machine-readable summary.",
        fontsize=9.3,
        color="#333333",
    )
    figure.tight_layout(rect=(0.04, 0.065, 0.99, 0.94), h_pad=2.4, w_pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "external_heldout_augmented_comparison"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
