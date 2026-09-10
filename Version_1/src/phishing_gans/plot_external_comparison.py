from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("CNN\nreal only", "artifacts/classifier"),
    ("LSTM\nreal only", "artifacts/classifier_lstm"),
    ("CNN\noversampling", "artifacts/classifier_cnn_oversample_48k"),
    ("CNN\nLSGAN", "artifacts/classifier_cnn_lsgan_48k"),
    ("CNN\nWGAN-GP", "artifacts/classifier_cnn_wgan_gp_48k"),
)
DATASETS = (
    ("URL dataset", "external_metrics.json", "#4472C4"),
    ("new_data_urls", "external_new_data_metrics.json", "#ED7D31"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen classifiers across two external URL datasets."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures")
    )
    return parser.parse_args()


def load_metrics(directory: str, filename: str) -> dict:
    path = Path(directory) / filename
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=180)
    x = np.arange(len(MODELS))
    width = 0.36

    for ax, metric, title in (
        (axes[0], "f1", "A. External F1 score"),
        (axes[1], "roc_auc", "B. External ROC-AUC"),
    ):
        for dataset_index, (dataset, filename, color) in enumerate(DATASETS):
            values = np.asarray(
                [
                    100 * load_metrics(directory, filename)[metric]
                    for _, directory in MODELS
                ]
            )
            offset = (dataset_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=dataset,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [label for label, _ in MODELS])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 82)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    figure.suptitle(
        "Frozen Classifier Performance Across Independent URL Sources",
        fontsize=17,
        fontweight="bold",
        x=0.045,
        ha="left",
        y=0.98,
    )
    figure.text(
        0.045,
        0.035,
        "Thresholds were selected on the original validation set and never retuned externally. "
        "The changing model ranking demonstrates substantial source/domain shift.",
        fontsize=10.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.035, 0.085, 0.99, 0.92), w_pad=3.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "external_datasets_comparison"
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
