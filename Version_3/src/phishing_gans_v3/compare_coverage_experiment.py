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
    parser = argparse.ArgumentParser(description="Compare original and legitimate-coverage checkpoints on the unchanged test.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts/experiments/legitimate_coverage_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/legitimate_coverage_v1/internal"))
    parser.add_argument("--experiment-name", default="legitimate_coverage_v1")
    parser.add_argument("--experiment-label", default="+ legitimate coverage")
    parser.add_argument("--seeds", default="42,43,44")
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    experiment = args.experiment_name
    roots = {"original_real_only": args.baseline_dir, experiment: args.experiment_dir}
    data: dict[str, dict[str, dict[int, dict]]] = {}
    for condition, root in roots.items():
        data[condition] = {}
        for model in MODELS:
            data[condition][model] = {
                seed: load_json(root / model / f"seed_{seed}" / "metrics.json") for seed in seeds
            }
    summary: dict[str, object] = {
        "protocol": f"paired seeds; unchanged original validation/test; comparison condition: {experiment}",
        "seeds": seeds, "models": {},
    }
    csv_rows = []
    for model in MODELS:
        model_payload = {"conditions": {}, "paired_delta_experiment_minus_original": {}}
        for condition in roots:
            model_payload["conditions"][condition] = {
                metric: mean_sd([float(data[condition][model][seed][metric]) for seed in seeds])
                for metric in METRICS
            }
        for metric in METRICS:
            deltas = [
                float(data[experiment][model][seed][metric])
                - float(data["original_real_only"][model][seed][metric]) for seed in seeds
            ]
            model_payload["paired_delta_experiment_minus_original"][metric] = mean_sd(deltas)
        summary["models"][model] = model_payload
        for condition in roots:
            row = {"model": model, "condition": condition}
            for metric in METRICS:
                values = model_payload["conditions"][condition][metric]
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_sd"] = values["standard_deviation"]
            csv_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", summary)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for axis, metric in zip(axes.ravel(), METRICS):
        for offset, (condition, color, label) in enumerate((
            ("original_real_only", "#4C78A8", "Original real-only"),
            (experiment, "#F58518", args.experiment_label),
        )):
            means = [100 * summary["models"][model]["conditions"][condition][metric]["mean"] for model in MODELS]
            errors = [100 * summary["models"][model]["conditions"][condition][metric]["standard_deviation"] for model in MODELS]
            axis.bar(x + (offset - 0.5) * width, means, width, yerr=errors, capsize=3, color=color, label=label)
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS], rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(0, 100)
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Unchanged internal test: original vs {args.experiment_label}\nmean ± SD across seeds 42/43/44", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(args.output_dir / "all_metrics_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_metrics_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"# {args.experiment_label} — Unchanged Internal Test", "",
        "Mean ± sample standard deviation across paired seeds 42/43/44. Original validation/test rows and threshold policy were unchanged.", "",
        "| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for condition, label in (("original_real_only", "Original"), (experiment, args.experiment_label)):
            values = summary["models"][model]["conditions"][condition]
            lines.append("| " + DISPLAY[model] + " | " + label + " | " + " | ".join(
                f'{values[metric]["mean"]:.4f} ± {values[metric]["standard_deviation"]:.4f}' for metric in METRICS
            ) + " |")
    lines.extend(("", "## Paired mean deltas: experiment minus original", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
    for model in MODELS:
        values = summary["models"][model]["paired_delta_experiment_minus_original"]
        lines.append("| " + DISPLAY[model] + " | " + " | ".join(f'{values[metric]["mean"]:+.4f}' for metric in METRICS) + " |")
    lines.extend(("", "With only three paired seeds, these deltas are descriptive and are not presented as conclusive statistical significance."))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "models": list(MODELS)}, indent=2))


if __name__ == "__main__":
    main()
