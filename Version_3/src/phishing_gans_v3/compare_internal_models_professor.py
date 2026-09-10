from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .calibrate_hybrid_threshold import ensemble_scores
from .common import load_json, require_cuda, write_json
from .metrics import classification_metrics


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
ORDER = ("cnn", "lstm", "rnn", "random_forest", "linear_svm", "hybrid")
LABELS = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF",
          "linear_svm": "SVM", "hybrid": "HYB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a professor-ready same-test model comparison.")
    parser.add_argument("--baseline-summary", type=Path, default=Path("artifacts/results/baseline_summary.json"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/processed/test_manifest.csv"))
    parser.add_argument("--threshold-record", type=Path, default=Path("artifacts/results/hybrid_cnn_coral_v1/calibrated_ensemble_v1/threshold.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/professor_model_comparison"))
    parser.add_argument("--maximum-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_payload = load_json(args.baseline_summary)
    baselines = baseline_payload.get("models", baseline_payload)
    with args.test_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    device = require_cuda()
    scores, _, checkpoints = ensemble_scores(rows, args, device)
    threshold = float(load_json(args.threshold_record)["threshold"])
    hybrid_metrics = classification_metrics(labels, scores, threshold)

    payload = {
        "project_version": 3, "evaluation_dataset": str(args.test_manifest), "rows": len(rows),
        "dataset_contract": "same frozen hostname-held-out test split from the original URLs_New dataset",
        "models": {}, "hybrid_threshold": threshold, "hybrid_checkpoints": checkpoints,
        "important_comparability_note": (
            "CNN/LSTM/RNN/RF/SVM were trained on the same 358,619 original real rows. HYB is evaluated on the same "
            "original test rows but was trained under its registered 363,692-row multi-source domain-adaptation condition."
        ),
    }
    for name in ORDER[:-1]:
        payload["models"][name] = {
            "display": LABELS[name], "metrics": baselines[name]["metrics"],
            "training_condition": "original real-only baseline",
        }
    payload["models"]["hybrid"] = {
        "display": "HYB", "metrics": {
            metric: {"mean": float(hybrid_metrics[metric]), "standard_deviation": 0.0} for metric in METRICS
        },
        "training_condition": "hybrid CNN + 35 lexical features + class-conditional CORAL; three-seed mean-score ensemble",
        "ensemble_single_result": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "comparison.json", payload)

    colors = ("#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#2A9D8F")
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.5))
    x = np.arange(len(ORDER))
    for axis, metric in zip(axes.ravel(), METRICS):
        means = [100 * payload["models"][name]["metrics"][metric]["mean"] for name in ORDER]
        errors = [100 * payload["models"][name]["metrics"][metric]["standard_deviation"] for name in ORDER]
        bars = axis.bar(x, means, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(x, [LABELS[name] for name in ORDER])
        axis.set_title(metric.replace("false_positive_rate", "FPR").replace("false_negative_rate", "FNR")
                       .replace("roc_auc", "ROC-AUC").replace("pr_auc", "PR-AUC").upper(), fontweight="bold")
        axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=2, fontsize=7.5)
    fig.suptitle(
        f"URL phishing classifiers — original frozen test set ({len(rows):,} URLs)\n"
        "CNN/LSTM/RNN/RF/SVM: mean ± SD across seeds 42/43/44; HYB: calibrated 3-seed ensemble",
        fontsize=17, fontweight="bold",
    )
    fig.text(.5, .012,
             "HYB uses the same test set but its registered multi-source CORAL training condition; see comparison report.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .93))
    fig.savefig(args.output_dir / "original_test_all_models.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "original_test_all_models.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Original-Test Classifier Comparison", "", payload["dataset_contract"] + ".", "",
             payload["important_comparability_note"], "",
             "Abbreviations: `RF` = Random Forest; `SVM` = linear Support Vector Machine; "
             "`HYB` = calibrated three-seed Hybrid CNN + Lexical + CORAL ensemble.", "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        values = payload["models"][name]["metrics"]
        lines.append("| " + LABELS[name] + " | " + " | ".join(f"{values[m]['mean']:.4f}" for m in METRICS) + " |")
    lines.extend(("", "The five baseline rows are means across three seeds. HYB is one mean-score ensemble result, so no seed SD is shown."))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
