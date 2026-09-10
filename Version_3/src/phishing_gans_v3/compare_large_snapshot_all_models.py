from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm", "hybrid")
LABELS = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF",
          "linear_svm": "SVM", "hybrid": "HYB"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
TITLES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR",
          "false_negative_rate": "FNR"}


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def main() -> None:
    baseline_path = Path("artifacts/results/external/2026-08-12_large_all_baselines/summary.json")
    hybrid_path = Path("artifacts/results/hybrid_cnn_coral_v1/external_large_2026-08-12/summary.json")
    output = Path("artifacts/results/external/2026-08-12_large_all_models")
    baselines = load_json(baseline_path)
    hybrid = load_json(hybrid_path)
    if baselines["dataset_sha256"] != hybrid["dataset_sha256"] or baselines["rows"] != hybrid["rows"]:
        raise RuntimeError("Baseline and hybrid results are not from the same frozen dataset")

    models = {}
    for name in MODELS[:-1]:
        runs = [row for row in baselines["per_run"] if row["model"] == name]
        models[name] = {
            "display": LABELS[name], "runs": len(runs),
            "metrics": {metric: mean_sd([float(row["metrics"][metric]) for row in runs]) for metric in METRICS},
            "mean_confusion_matrix": {
                cell: float(np.mean([row["metrics"]["confusion_matrix"][cell] for row in runs]))
                for cell in ("tn", "fp", "fn", "tp")
            },
            "threshold": "unchanged internal-validation threshold per seed",
        }
    models["hybrid"] = {
        "display": "HYB", "runs": 1,
        "metrics": {metric: {"mean": float(hybrid["metrics"][metric]), "standard_deviation": 0.0}
                    for metric in METRICS},
        "mean_confusion_matrix": hybrid["metrics"]["confusion_matrix"],
        "threshold": hybrid["threshold"],
        "note": "frozen three-seed mean-score ensemble; not three independent runs",
    }
    payload = {
        "project_version": 3, "dataset": baselines["dataset"], "dataset_sha256": baselines["dataset_sha256"],
        "rows": baselines["rows"], "legitimate_rows": baselines["legitimate_rows"],
        "phishing_rows": baselines["phishing_rows"], "models": models,
        "protocol": "all checkpoints and thresholds frozen; no retraining, reselection, or external threshold tuning",
        "comparability_note": (
            "Real-only baselines use their frozen internal thresholds and report mean ± sample SD across seeds 42/43/44. "
            "HYB is the frozen calibrated three-seed ensemble and therefore has one result rather than a seed SD."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", payload)

    colors = ("#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#2A9D8F")
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        means = [100 * models[name]["metrics"][metric]["mean"] for name in MODELS]
        errors = [100 * models[name]["metrics"][metric]["standard_deviation"] for name in MODELS]
        bars = axis.bar(x, means, yerr=errors, capsize=3, color=colors)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=2, fontsize=7.5)
        axis.set_xticks(x, [LABELS[name] for name in MODELS])
        axis.set_title(TITLES.get(metric, metric.upper()), fontweight="bold")
        axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
    fig.suptitle(
        "Frozen 10K external snapshot — all URL classifiers\n"
        "5,000 Tranco legitimate proxies + 5,000 OpenPhish/PhishTank phishing URLs",
        fontsize=17, fontweight="bold",
    )
    fig.text(.5, .012, "Baselines: mean ± SD across seeds 42/43/44; HYB: frozen calibrated 3-seed ensemble",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .93))
    fig.savefig(output / "all_models_all_metrics.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "all_models_all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Frozen 10K External Snapshot — Baselines vs HYB", "",
             f"The exact frozen dataset contains {payload['legitimate_rows']:,} legitimate proxies and "
             f"{payload['phishing_rows']:,} phishing URLs. SHA-256: `{payload['dataset_sha256']}`.", "",
             payload["protocol"] + ".", "", payload["comparability_note"], "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        values = models[name]["metrics"]
        lines.append("| " + LABELS[name] + " | " + " | ".join(
            f"{values[metric]['mean']:.4f}" for metric in METRICS) + " |")
    lines.extend(("", "## Interpretation", "",
                  "HYB is the only model with a practical false-positive rate on this snapshot. The real-only neural "
                  "baselines retain high phishing recall but achieve it by labelling almost all legitimate roots as phishing. "
                  "RF and SVM also show ranking reversal (ROC-AUC below 0.5), so their failures cannot be repaired by threshold adjustment alone.",
                  "", "The snapshot remains external evaluation data and must not be moved into training or threshold selection."))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
