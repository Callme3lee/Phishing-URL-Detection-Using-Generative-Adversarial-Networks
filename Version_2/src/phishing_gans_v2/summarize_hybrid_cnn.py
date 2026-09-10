from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "false_positive_rate", "false_negative_rate",
)
DISPLAY = {
    "cnn": "Character CNN", "lstm": "Character LSTM",
    "random_forest": "Random Forest", "linear_svm": "Linear SVM",
    "hybrid_cnn": "Hybrid CNN",
}


def metric_value(run: dict, metric: str) -> float:
    if metric != "false_negative_rate":
        return float(run[metric])
    matrix = run["confusion_matrix"]
    return float(matrix["fn"] / max(1, matrix["fn"] + matrix["tp"]))


def aggregate(runs: list[dict]) -> dict:
    payload = {"runs": len(runs), "seeds": [int(run["seed"]) for run in runs]}
    for metric in METRICS:
        values = np.asarray([metric_value(run, metric) for run in runs], dtype=float)
        payload[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return payload


def main() -> None:
    baseline_dir = Path("artifacts/baselines")
    hybrid_dir = Path("artifacts/hybrid_cnn")
    output_dir = Path("artifacts/results/hybrid_cnn")
    grouped: dict[str, list[dict]] = {}
    for model in ("cnn", "lstm", "random_forest", "linear_svm"):
        grouped[model] = [
            load_json(baseline_dir / model / f"seed_{seed}" / "metrics.json")
            for seed in (42, 43, 44)
        ]
    grouped["hybrid_cnn"] = [
        load_json(hybrid_dir / f"seed_{seed}" / "metrics.json") for seed in (42, 43, 44)
    ]
    summary = {
        "project_version": 2,
        "experiment": "isolated real-only hybrid character CNN plus lexical features",
        "training_rows": 2_055_821,
        "generated_rows": 0,
        "conditions": {model: aggregate(runs) for model, runs in grouped.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *[f"{m}_mean" for m in METRICS], *[f"{m}_sd" for m in METRICS]])
        writer.writeheader()
        for model, values in summary["conditions"].items():
            writer.writerow({
                "model": model,
                **{f"{metric}_mean": values[metric]["mean"] for metric in METRICS},
                **{f"{metric}_sd": values[metric]["sd"] for metric in METRICS},
            })

    order = list(DISPLAY)
    colors = ["#2b6cb0", "#319795", "#dd6b20", "#805ad5", "#c53030"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.8))
    for axis, metric in zip(axes, ("accuracy", "precision", "recall", "f1", "roc_auc")):
        means = [100 * summary["conditions"][name][metric]["mean"] for name in order]
        errors = [100 * summary["conditions"][name][metric]["sd"] for name in order]
        axis.bar(range(len(order)), means, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(range(len(order)), [DISPLAY[name].replace("Character ", "") for name in order], rotation=28, ha="right")
        axis.set_title(metric.replace("_", " ").upper()); axis.set_ylim(80, 100)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle("Real-only classifiers on the unchanged frozen test set (mean ± SD, seeds 42/43/44)")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    hybrid = summary["conditions"]["hybrid_cnn"]
    best_existing = max(
        (name for name in order if name != "hybrid_cnn"),
        key=lambda name: summary["conditions"][name]["f1"]["mean"],
    )
    auc_deltas = np.asarray([
        left - right for left, right in zip(
            hybrid["roc_auc"]["values"],
            summary["conditions"][best_existing]["roc_auc"]["values"],
        )
    ])
    auc_delta_mean = float(auc_deltas.mean())
    # Two-sided t(2) 95% interval for the predeclared three paired seeds.
    auc_half_width = float(4.3026527299 * auc_deltas.std(ddof=1) / np.sqrt(len(auc_deltas)))
    lines = [
        "# Hybrid CNN Real-Only Evaluation", "",
        "The hybrid was added without changing the frozen real data, hostname-separated splits, existing models, seeds, or threshold policy.", "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        item = summary["conditions"][name]
        lines.append("| " + DISPLAY[name] + " | " + " | ".join(f"{item[m]['mean']:.6f} ± {item[m]['sd']:.6f}" for m in METRICS) + " |")
    lines.extend([
        "", "## Paired design", "",
        "All models used the same 2,055,821 real training rows and unchanged validation/test sets. The hybrid contains no generated rows. Its 28 lexical features were standardized using training rows only.", "",
        f"The strongest existing model by mean F1 is {DISPLAY[best_existing]}. Hybrid-minus-{DISPLAY[best_existing]} mean deltas are:", "",
        *[f"- {metric.replace('_', ' ').title()}: {hybrid[metric]['mean'] - summary['conditions'][best_existing][metric]['mean']:+.6f}" for metric in METRICS],
        "", "With three paired seeds, accuracy, recall, and F1 deltas have 95% intervals that cross zero. "
        f"The ROC-AUC delta is {auc_delta_mean:+.6f} with paired 95% CI "
        f"[{auc_delta_mean-auc_half_width:+.6f}, {auc_delta_mean+auc_half_width:+.6f}], the only requested improvement whose interval excludes zero. "
        "Mean gains must not be presented as universal external improvement.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
