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
    parser = argparse.ArgumentParser(description="Compare two conditions evaluated on the same frozen external set.")
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--experiment-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default="legitimate_coverage_v1")
    parser.add_argument("--experiment-label", default="+ legitimate coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = load_json(args.baseline_summary)
    experiment_summary = load_json(args.experiment_summary)
    experiment = args.experiment_name
    if original["dataset_sha256"] != experiment_summary["dataset_sha256"]:
        raise RuntimeError("External summaries do not refer to the same frozen dataset")
    payload = {
        "dataset_sha256": original["dataset_sha256"], "rows": original["rows"],
        "legitimate_rows": original["legitimate_rows"], "phishing_rows": original["phishing_rows"],
        "seeds": original["seeds"], "models": {},
    }
    rows = []
    for model in MODELS:
        payload["models"][model] = {"conditions": {}, "delta_experiment_minus_original": {}}
        for condition, source in (("original_real_only", original), (experiment, experiment_summary)):
            payload["models"][model]["conditions"][condition] = source["grouped"][model]["metrics"]
            row = {"model": model, "condition": condition}
            for metric in METRICS:
                row[f"{metric}_mean"] = source["grouped"][model]["metrics"][metric]["mean"]
                row[f"{metric}_sd"] = source["grouped"][model]["metrics"][metric]["standard_deviation"]
            rows.append(row)
        for metric in METRICS:
            original_values = original["grouped"][model]["metrics"][metric]["values"]
            experiment_values = experiment_summary["grouped"][model]["metrics"][metric]["values"]
            deltas = [float(new) - float(old) for old, new in zip(original_values, experiment_values)]
            payload["models"][model]["delta_experiment_minus_original"][metric] = {
                "mean": float(np.mean(deltas)),
                "standard_deviation": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "values": deltas,
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", payload)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for axis, metric in zip(axes.ravel(), METRICS):
        for offset, (condition, color, label) in enumerate((
            ("original_real_only", "#4C78A8", "Original real-only"),
            (experiment, "#F58518", args.experiment_label),
        )):
            means = [100 * payload["models"][model]["conditions"][condition][metric]["mean"] for model in MODELS]
            errors = [100 * payload["models"][model]["conditions"][condition][metric]["standard_deviation"] for model in MODELS]
            axis.bar(x + (offset - 0.5) * width, means, width, yerr=errors, capsize=3, color=color, label=label)
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS], rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(0, 100)
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Frozen external set ({payload['legitimate_rows']} legitimate + {payload['phishing_rows']} phishing)\n"
        f"original vs {args.experiment_label}; mean ± SD",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(args.output_dir / "all_metrics_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_metrics_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"# External Comparison — Original vs {args.experiment_label}", "",
        f"Frozen dataset SHA-256: `{payload['dataset_sha256']}`. Mean ± sample SD across paired seeds 42/43/44; thresholds remain those selected on each run's unchanged internal validation set.", "",
        "| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for condition, label in (("original_real_only", "Original"), (experiment, args.experiment_label)):
            values = payload["models"][model]["conditions"][condition]
            lines.append("| " + DISPLAY[model] + " | " + label + " | " + " | ".join(
                f'{values[metric]["mean"]:.4f} ± {values[metric]["standard_deviation"]:.4f}' for metric in METRICS
            ) + " |")
    lines.extend(("", "## Paired mean deltas: experiment minus original", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
    for model in MODELS:
        values = payload["models"][model]["delta_experiment_minus_original"]
        lines.append("| " + DISPLAY[model] + " | " + " | ".join(f'{values[metric]["mean"]:+.4f}' for metric in METRICS) + " |")
    lines.extend(("", "FPR/FNR are error rates, so negative deltas are improvements. Three seeds support descriptive paired comparison, not a strong significance claim."))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "dataset_sha256": payload["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
