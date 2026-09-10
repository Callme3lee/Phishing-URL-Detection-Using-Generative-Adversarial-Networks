from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = (("cnn", "CNN"), ("lstm", "LSTM"), ("rf", "Random Forest"), ("svm", "Linear SVM"))
GANS = (
    ("lsgan", "LSTM-LSGAN", "#2A9D8F"),
    ("wgan_gp", "LSTM-WGAN-GP", "#7E57C2"),
    ("combined_gans", "Combined 50/50", "#F28E2B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GAN augmentation dose-response results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/augmented_experiments/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["rows"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)
    x_values = [0, 20, 30, 40]

    for ax, (model, label) in zip(axes.flat, MODELS, strict=True):
        reference = next(
            row for row in rows if row["model"] == model and row["generated_percentage"] == 0
        )
        all_values = [100 * reference["f1"]]
        for gan, gan_label, color in GANS:
            selected = sorted(
                (row for row in rows if row["model"] == model and row["gan"] == gan),
                key=lambda row: row["generated_percentage"],
            )
            values = [100 * reference["f1"], *[100 * row["f1"] for row in selected]]
            all_values.extend(values[1:])
            ax.plot(
                x_values,
                values,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=color,
                label=gan_label,
            )
        ax.axhline(
            100 * reference["f1"],
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )
        padding = max(0.12, (max(all_values) - min(all_values)) * 0.28)
        ax.set_ylim(min(all_values) - padding, max(all_values) + padding)
        ax.set_xticks(x_values)
        ax.set_xlabel("Generated URLs in training dataset (%)")
        ax.set_ylabel("F1 score (%)")
        ax.set_title(label, loc="left")
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0, 0].legend(frameon=False, ncols=1, loc="best")
    figure.suptitle(
        "Effect of Unique GAN-Generated URL Percentage on Classifier F1",
        fontsize=16,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.06,
        0.025,
        "All conditions use the same 69,300 real training URLs and the original fixed 252,430-URL test set. "
        "Dashed line = model-specific real-only reference. Expanded y-axes; one training seed.",
        fontsize=9.5,
        color="#333333",
    )
    figure.tight_layout(rect=(0.045, 0.065, 0.99, 0.94), h_pad=2.2, w_pad=2.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "augmented_percentage_f1"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
