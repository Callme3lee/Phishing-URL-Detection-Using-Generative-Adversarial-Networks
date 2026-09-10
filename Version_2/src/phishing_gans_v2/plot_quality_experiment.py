from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .common import load_json
from .evaluate_generated_challenge import DISPLAY, MODELS
from .summarize_quality_experiment import CONDITIONS, CONDITION_DISPLAY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frozen quality-experiment summaries without rerunning inference.")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/results/quality_experiment/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/quality_experiment"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); payload = load_json(args.summary); grouped = payload["grouped"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_conditions = ("realism_filtered_10_percent", "realism_challenging_10_percent")
    columns = (
        "recall", "f1", "roc_auc", "false_positive_rate",
        "false_negative_rate", "accuracy", "precision", "challenge_recall",
    )
    rows = []
    labels = []
    for model in MODELS:
        for condition in selected_conditions:
            deltas = grouped[model][condition]["paired_deltas"]["random_10_percent"]
            values = [100 * deltas["real_test_locked"][metric]["mean_delta"] for metric in columns[:-1]]
            values.append(100 * deltas["synthetic_challenge_locked"]["recall"]["mean_delta"])
            rows.append(values)
            labels.append(f"{DISPLAY[model]} — {'filtered' if condition == selected_conditions[0] else 'challenging'}")
    matrix = np.asarray(rows)
    limit = float(np.max(np.abs(matrix)))
    fig, axis = plt.subplots(figsize=(13.5, 7.5))
    image = axis.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    axis.set_xticks(range(len(columns)), [value.replace("_", " ").upper() for value in columns], rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                      color="white" if abs(value) > .55 * limit else "black")
    axis.set_title("Quality-selected 10% augmentation: change vs random SeqGAN 10%\npercentage points, common real-only thresholds", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Paired mean change (percentage points)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "delta_vs_random_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "delta_vs_random_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    x = np.arange(len(CONDITIONS)); labels = [CONDITION_DISPLAY[c] for c in CONDITIONS]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9)); axes = axes.ravel()
    for axis, model in zip(axes, MODELS):
        locked = [100 * grouped[model][c]["real_test_locked"]["f1"]["mean"] for c in CONDITIONS]
        native = [100 * grouped[model][c]["real_test_native"]["f1"]["mean"] for c in CONDITIONS]
        width = .36
        axis.bar(x - width / 2, locked, width, label="Common real-only threshold", color=colors[0])
        axis.bar(x + width / 2, native, width, label="Condition validation threshold", color=colors[1])
        axis.set_xticks(x, labels, rotation=20, ha="right"); axis.set_ylabel("Real-test F1 (%)")
        axis.set_title(DISPLAY[model], fontweight="bold"); axis.grid(axis="y", alpha=.25)
        low = min(locked + native); high = max(locked + native); padding = max(.08, (high-low)*.18)
        axis.set_ylim(low-padding, high+padding)
    axes[0].legend(loc="best")
    fig.suptitle("Threshold protocol changes the apparent real-test effect\nROC-AUC is threshold-independent and remains reported separately", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.output_dir / "threshold_protocol_f1.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "threshold_protocol_f1.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
