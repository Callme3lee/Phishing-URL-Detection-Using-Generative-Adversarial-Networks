from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json


RESULT_ROOT = Path("artifacts/results/extension_v0_5_2_evaluation_2026-09-01_v1")
RESULTS = RESULT_ROOT / "results.json"
OUTPUT_PNG = RESULT_ROOT / "v0_5_1_vs_v0_5_2_all_metrics.png"
OUTPUT_PDF = RESULT_ROOT / "v0_5_1_vs_v0_5_2_all_metrics.pdf"
OUTPUT_CSV = RESULT_ROOT / "v0_5_1_vs_v0_5_2_all_metrics.csv"
METRICS = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("false_positive_rate", "FPR"),
    ("false_negative_rate", "FNR"),
)
DATASETS = (
    ("final_6k_5pct", "Final 6K\n5% phishing"),
    ("postfreeze_10k_3pct", "Post-freeze 10K\n3% phishing"),
)


def main() -> None:
    payload = load_json(RESULTS)
    versions = ("v0.5.1", "v0.5.2")
    colors = ("#58799C", "#2A9D8F")
    x = np.arange(len(DATASETS))
    width = 0.34

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), facecolor="white")
    fig.suptitle("Browser Extension v0.5.1 vs v0.5.2",
                 fontsize=24, weight="bold", color="#17324D", y=.975)
    fig.text(.5, .935,
             "Same HYB-MS + 10% MS-HardGAN-C ensemble • only the warning threshold changed",
             ha="center", fontsize=12.5, color="#526678")

    csv_rows = []
    for axis, (metric_key, metric_name) in zip(axes.ravel(), METRICS):
        old_values = []
        new_values = []
        for dataset_key, dataset_label in DATASETS:
            result = payload["datasets"][dataset_key]
            old_value = 100 * result["v0_5_1_same_scores_metrics"][metric_key]
            new_value = 100 * result["metrics"][metric_key]
            old_values.append(old_value)
            new_values.append(new_value)
            csv_rows.extend([
                {"dataset": dataset_label.replace("\n", " "), "version": versions[0],
                 "metric": metric_name, "percent": old_value},
                {"dataset": dataset_label.replace("\n", " "), "version": versions[1],
                 "metric": metric_name, "percent": new_value},
            ])
        old_bars = axis.bar(x - width / 2, old_values, width, label=versions[0], color=colors[0])
        new_bars = axis.bar(x + width / 2, new_values, width, label=versions[1], color=colors[1])
        axis.set_title(metric_name, fontsize=13, weight="bold", color="#17324D")
        axis.set_xticks(x, [label for _, label in DATASETS], fontsize=9)
        axis.grid(axis="y", alpha=.20)
        axis.set_axisbelow(True)
        axis.bar_label(old_bars, labels=[f"{value:.2f}" for value in old_values],
                       fontsize=8.5, padding=2, weight="bold")
        axis.bar_label(new_bars, labels=[f"{value:.2f}" for value in new_values],
                       fontsize=8.5, padding=2, weight="bold")
        if metric_key == "false_positive_rate":
            axis.set_ylim(0, max(1.35, max(old_values + new_values) * 1.45))
        elif metric_key == "false_negative_rate":
            axis.set_ylim(0, max(32, max(old_values + new_values) * 1.25))
        else:
            axis.set_ylim(0, 108)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=12)
    fig.text(.01, .5, "Metric (%)", rotation=90, va="center", fontsize=11, color="#526678")
    fig.tight_layout(rect=(.025, .06, 1, .91), h_pad=2.7, w_pad=1.8)
    fig.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "version", "metric", "percent"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
