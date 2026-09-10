from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("real", "oversample", "lsgan", "wgan_gp")
LABELS = ("Real only", "Real oversampling", "LSTM-LSGAN", "LSTM-WGAN-GP")
COLORS = ("#345995", "#F28E2B", "#2A9D8F", "#7E57C2")
EXTERNAL_FILES = {
    "real": Path("artifacts/classifier/external_metrics.json"),
    "oversample": Path("artifacts/classifier_cnn_oversample_48k/external_metrics.json"),
    "lsgan": Path("artifacts/classifier_cnn_lsgan_48k/external_metrics.json"),
    "wgan_gp": Path("artifacts/classifier_cnn_wgan_gp_48k/external_metrics.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the final classifier comparison.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/multiseed_10pct_summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/figures")
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_values(summary: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = []
    errors = []
    for condition in CONDITIONS:
        result = summary["conditions"][condition]["metrics"][metric]
        means.append(100 * result["mean"])
        errors.append(100 * result["ci95_half_width"])
    return np.asarray(means), np.asarray(errors)


def add_point_labels(ax: plt.Axes, x: np.ndarray, values: np.ndarray, offset: float) -> None:
    for position, value in zip(x, values, strict=True):
        ax.text(
            position,
            value + offset,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    external = {
        condition: load_json(path)["metrics"]
        for condition, path in EXTERNAL_FILES.items()
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 6.8), dpi=180)
    x = np.arange(len(CONDITIONS))

    f1_mean, f1_error = interval_values(summary, "f1")
    for position, mean, error, color in zip(x, f1_mean, f1_error, COLORS, strict=True):
        axes[0].errorbar(
            position,
            mean,
            yerr=error,
            fmt="none",
            ecolor=color,
            elinewidth=2.4,
            capsize=6,
            capthick=2.4,
            zorder=2,
        )
    axes[0].scatter(x, f1_mean, s=105, c=COLORS, edgecolor="white", linewidth=1.2, zorder=3)
    add_point_labels(axes[0], x, f1_mean, 0.07)
    axes[0].set_ylim(95.35, 96.85)
    axes[0].set_ylabel("F1 score (%)")
    axes[0].set_title("A. Limited-data F1 (10% real data)", loc="left", y=1.08)
    axes[0].text(
        0,
        1.015,
        "Expanded y-axis; mean and 95% Student-t CI, 3 seeds",
        transform=axes[0].transAxes,
        color="#555555",
        fontsize=9,
    )

    fpr_mean, fpr_error = interval_values(summary, "false_positive_rate")
    for position, mean, error, color in zip(x, fpr_mean, fpr_error, COLORS, strict=True):
        axes[1].errorbar(
            position,
            mean,
            yerr=error,
            fmt="none",
            ecolor=color,
            elinewidth=2.4,
            capsize=6,
            capthick=2.4,
            zorder=2,
        )
    axes[1].scatter(x, fpr_mean, s=105, c=COLORS, edgecolor="white", linewidth=1.2, zorder=3)
    add_point_labels(axes[1], x, fpr_mean, 0.15)
    axes[1].set_ylim(2.2, 6.5)
    axes[1].set_ylabel("False-positive rate (%)")
    axes[1].set_title("B. Limited-data false positives", loc="left", y=1.08)
    axes[1].text(
        0,
        1.015,
        "Lower is better; mean and 95% CI",
        transform=axes[1].transAxes,
        color="#555555",
        fontsize=9,
    )

    external_f1 = np.asarray([100 * external[key]["f1"] for key in CONDITIONS])
    external_roc = np.asarray([100 * external[key]["roc_auc"] for key in CONDITIONS])
    width = 0.35
    bars_f1 = axes[2].bar(
        x - width / 2,
        external_f1,
        width,
        label="F1",
        color=COLORS,
        alpha=0.72,
    )
    bars_roc = axes[2].bar(
        x + width / 2,
        external_roc,
        width,
        label="ROC-AUC",
        color=COLORS,
        edgecolor="#222222",
        linewidth=1.0,
        hatch="//",
    )
    axes[2].bar_label(bars_f1, fmt="%.1f", padding=2, fontsize=8)
    axes[2].bar_label(bars_roc, fmt="%.1f", padding=2, fontsize=8)
    axes[2].set_ylim(0, 82)
    axes[2].set_ylabel("Score (%)")
    axes[2].set_title("C. External-source generalization", loc="left", y=1.08)
    axes[2].text(
        0,
        1.015,
        "Frozen full-data CNN checkpoints",
        transform=axes[2].transAxes,
        color="#555555",
        fontsize=9,
    )
    axes[2].legend(frameon=False, ncols=2, loc="upper left")

    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=24, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    figure.suptitle(
        "GAN-Augmented Malicious URL Classification: Main Results",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.04,
        0.018,
        "Oversampling gives the highest in-domain F1; WGAN-GP gives the lowest mean FPR; "
        "LSGAN gives the strongest external ROC-AUC. Overlapping CIs mean no significant "
        "in-domain winner is established.",
        fontsize=10,
        color="#333333",
    )
    figure.tight_layout(rect=(0.03, 0.075, 0.99, 0.92), w_pad=2.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "professor_results_summary"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
