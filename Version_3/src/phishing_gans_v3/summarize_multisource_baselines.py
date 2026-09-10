from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF", "linear_svm": "SVM"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")
METRIC_DISPLAY = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
}
SEEDS = (42, 43, 44)


def aggregate(rows: list[dict]) -> dict[str, dict[str, float | list[float]]]:
    result = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "values": values,
        }
    return result


def internal(checkpoint_root: Path) -> dict[str, dict]:
    return {
        model: aggregate([
            load_json(checkpoint_root / model / f"seed_{seed}" / "metrics.json") for seed in SEEDS
        ])
        for model in MODELS
    }


def external(summary_path: Path) -> dict[str, dict]:
    summary = load_json(summary_path)
    return {
        model: aggregate([
            row["metrics"] for row in summary["per_run"] if row["model"] == model
        ])
        for model in MODELS
    }


def delta(new: dict, old: dict) -> dict:
    return {
        model: {
            metric: float(new[model][metric]["mean"] - old[model][metric]["mean"])
            for metric in METRICS
        }
        for model in MODELS
    }


def grouped_plot(old: dict, new: dict, output: Path, title: str) -> None:
    x = np.arange(len(MODELS))
    width = 0.37
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        old_values = [100 * old[model][metric]["mean"] for model in MODELS]
        new_values = [100 * new[model][metric]["mean"] for model in MODELS]
        old_errors = [100 * old[model][metric]["standard_deviation"] for model in MODELS]
        new_errors = [100 * new[model][metric]["standard_deviation"] for model in MODELS]
        axis.bar(x - width / 2, old_values, width, yerr=old_errors, capsize=3,
                 color="#9CA3AF", label="Original-only")
        axis.bar(x + width / 2, new_values, width, yerr=new_errors, capsize=3,
                 color="#2563EB", label="Multi-source")
        axis.set_title(METRIC_DISPLAY[metric], fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS])
        axis.set_ylabel("Percent")
        axis.grid(axis="y", alpha=0.22)
        if metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(0, 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.suptitle(title + "\nMean ± SD across seeds 42/43/44", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original_root = Path("artifacts/baselines")
    multisource_root = Path("artifacts/experiments/multisource_baselines_v1")
    original_external_path = Path("artifacts/results/external/2026-08-12_large_all_baselines/summary.json")
    multisource_external_path = Path("artifacts/results/multisource_baselines_v1/external_large_2026-08-12/summary.json")
    output = Path("artifacts/results/multisource_baselines_v1/summary")
    output.mkdir(parents=True, exist_ok=True)

    original_internal = internal(original_root)
    multisource_internal = internal(multisource_root)
    original_external = external(original_external_path)
    multisource_external = external(multisource_external_path)
    payload = {
        "project_version": 3,
        "condition": "multisource_baselines_v1",
        "training_rows": {"original_only": 358619, "multisource": 363692, "added_real_rows": 5073},
        "synthetic_rows": 0,
        "seeds": list(SEEDS),
        "internal_original_test": {
            "original_only": original_internal, "multisource": multisource_internal,
            "multisource_minus_original": delta(multisource_internal, original_internal),
        },
        "external_frozen_10k": {
            "original_only": original_external, "multisource": multisource_external,
            "multisource_minus_original": delta(multisource_external, original_external),
        },
        "interpretation": [
            "Adding 5,073 outside real rows under ordinary baseline sampling improves ranking robustness for several models, but does not solve threshold transfer.",
            "The multi-source LSTM is the strongest of these five unchanged baselines on the external snapshot by ROC-AUC and F1.",
            "All five unchanged multi-source baselines remain unsuitable for deployment because external FPR or FNR is too high.",
            "HYB-MS is not explained by the row set alone: it additionally uses source-balanced exposure, PSL-aware lexical fusion, conditional CORAL, a three-seed ensemble, and a separate external calibration set.",
        ],
    }
    write_json(output / "summary.json", payload)

    grouped_plot(
        original_internal, multisource_internal, output / "internal_original_vs_multisource",
        "Unchanged baselines — original frozen test",
    )
    grouped_plot(
        original_external, multisource_external, output / "external_original_vs_multisource",
        "Unchanged baselines — frozen 10K external snapshot",
    )

    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["evaluation", "model", "condition"] + list(METRICS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for evaluation, pair in (("internal", (original_internal, multisource_internal)),
                                 ("external_10k", (original_external, multisource_external))):
            for condition, group in zip(("original_only", "multisource"), pair):
                for model in MODELS:
                    writer.writerow({
                        "evaluation": evaluation, "model": model, "condition": condition,
                        **{metric: group[model][metric]["mean"] for metric in METRICS},
                    })

    def table(group: dict) -> list[str]:
        rows = ["| Model | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            rows.append("| " + DISPLAY[model] + " | " + " | ".join(
                f'{group[model][metric]["mean"]:.4f} ± {group[model][metric]["standard_deviation"]:.4f}'
                for metric in METRICS
            ) + " |")
        return rows

    lines = [
        "# Multi-source Baseline Experiment", "",
        "## Controlled question", "",
        "Does giving the unchanged CNN, LSTM, RNN, RF, and linear SVM baselines the same stored multi-source real row set used by HYB-MS improve source-transfer performance?", "",
        "The training set contains **363,692 unique real URLs**: 358,619 original URLs_New training rows plus 3,012 outside legitimate, 2,000 PhishTank, and 61 OpenPhish rows. It contains **zero GAN-generated rows**. The original validation (39,483) and test (49,229) sets, model hyperparameters, seeds 42/43/44, and threshold rule are unchanged.", "",
        "## Internal original test — multi-source baselines", "", *table(multisource_internal), "",
        "## Frozen 10K external snapshot — multi-source baselines", "", *table(multisource_external), "",
        "## Main finding", "",
        "The multi-source **LSTM is the best of the five unchanged baselines externally**: its mean ROC-AUC is "
        f"{multisource_external['lstm']['roc_auc']['mean']:.4f} and F1 is {multisource_external['lstm']['f1']['mean']:.4f}. "
        f"However, its FPR remains {multisource_external['lstm']['false_positive_rate']['mean']:.4f}, so it still falsely flags most external legitimate roots.", "",
        "The added real rows improved ranking information without producing reliable threshold transfer. For example, external LSTM ROC-AUC rose by "
        f"{delta(multisource_external, original_external)['lstm']['roc_auc']:+.4f}, while its FPR changed by "
        f"{delta(multisource_external, original_external)['lstm']['false_positive_rate']:+.4f}. "
        "CNN and RNN also gained ROC-AUC but retained near-total legitimate false-positive rates.", "",
        "This means the multi-source **row set alone is insufficient**. HYB-MS also depends on source-balanced exposure, PSL-aware lexical features, conditional CORAL alignment, three-seed ensembling, and a separate frozen calibration set. Therefore HYB-MS must not be described as a fair architecture-only comparison with these ordinary-sampling baselines.", "",
        "## Deployment conclusion", "",
        "None of these five unchanged multi-source baselines is deployment-ready on this snapshot. HYB-MS remains the only tested condition with an externally usable balance (FPR 0.0170, FNR 0.1476, F1 0.9120), but that result is an ensemble calibrated on a separate outside-domain calibration set and must be reported as such.", "",
        "## Artifacts", "",
        "- `internal_original_vs_multisource.png`: original-only versus multi-source on the unchanged internal test.",
        "- `external_original_vs_multisource.png`: the same controlled comparison on the frozen external 10K snapshot.",
        "- `comparison.csv` and `summary.json`: machine-readable means, SDs, and deltas.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
