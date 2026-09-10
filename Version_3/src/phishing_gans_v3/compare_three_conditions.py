from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "Random Forest", "linear_svm": "Linear SVM"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original, outside-coverage, and URLs_New-balanced external results.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        "original": load_json(args.original),
        "outside_coverage": load_json(args.coverage),
        "urls_new_balanced": load_json(args.balanced),
    }
    hashes = {summary["dataset_sha256"] for summary in summaries.values()}
    if len(hashes) != 1:
        raise RuntimeError("The three summaries do not use the same frozen external set")
    payload = {
        "dataset_sha256": next(iter(hashes)),
        "rows": summaries["original"]["rows"],
        "conditions": {},
    }
    csv_rows = []
    for condition, summary in summaries.items():
        payload["conditions"][condition] = summary["grouped"]
        for model in MODELS:
            row = {"condition": condition, "model": model}
            for metric in METRICS:
                item = summary["grouped"][model]["metrics"][metric]
                row[f"{metric}_mean"] = item["mean"]
                row[f"{metric}_sd"] = item["standard_deviation"]
            csv_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", payload)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    styles = (
        ("original", "#4C78A8", "Original URLs_New"),
        ("outside_coverage", "#F58518", "+ 3,012 outside coverage"),
        ("urls_new_balanced", "#54A24B", "URLs_New structure-balanced"),
    )
    x = np.arange(len(MODELS))
    width = 0.25
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        for offset, (condition, color, label) in enumerate(styles):
            means = [100 * summaries[condition]["grouped"][model]["metrics"][metric]["mean"] for model in MODELS]
            errors = [100 * summaries[condition]["grouped"][model]["metrics"][metric]["standard_deviation"] for model in MODELS]
            axis.bar(x + (offset - 1) * width, means, width, yerr=errors, capsize=2.5, color=color, label=label)
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS], rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(0, 100)
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Three controlled training conditions on one frozen external set (n={payload['rows']})\nmean ± SD across seeds 42/43/44",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(args.output_dir / "all_metrics_three_conditions.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_metrics_three_conditions.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir), "dataset_sha256": payload["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
