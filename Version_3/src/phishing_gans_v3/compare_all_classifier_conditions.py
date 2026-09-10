from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)
QUALITY = METRICS[:6]
ERRORS = METRICS[6:]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
}
BASELINES = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
SHORT = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF", "linear_svm": "SVM"}
ROWS = tuple(
    item for model in BASELINES for item in ((f"{model}_orig", f"{SHORT[model]}-ORIG"), (f"{model}_ms", f"{SHORT[model]}-MS"))
) + (("hyb_orig", "HYB-ORIG"), ("hyb_ms", "HYB-MS*"))


def means(group: dict) -> dict[str, float]:
    return {metric: float(group[metric]["mean"]) for metric in METRICS}


def sds(group: dict) -> dict[str, float]:
    return {metric: float(group[metric].get("standard_deviation", 0.0)) for metric in METRICS}


def load_results() -> tuple[dict, dict]:
    baseline = load_json(Path("artifacts/results/multisource_baselines_v1/summary/summary.json"))
    professor = load_json(Path("artifacts/results/professor_model_comparison/comparison.json"))
    hybrid_original = load_json(Path("artifacts/results/hybrid_original_v1/summary/summary.json"))
    hybrid_ms_external = load_json(Path("artifacts/results/hybrid_cnn_coral_v1/external_large_2026-08-12/summary.json"))

    internal: dict[str, dict] = {}
    external: dict[str, dict] = {}
    for model in BASELINES:
        internal[f"{model}_orig"] = {
            "metrics": means(baseline["internal_original_test"]["original_only"][model]),
            "sd": sds(baseline["internal_original_test"]["original_only"][model]),
        }
        internal[f"{model}_ms"] = {
            "metrics": means(baseline["internal_original_test"]["multisource"][model]),
            "sd": sds(baseline["internal_original_test"]["multisource"][model]),
        }
        external[f"{model}_orig"] = {
            "metrics": means(baseline["external_frozen_10k"]["original_only"][model]),
            "sd": sds(baseline["external_frozen_10k"]["original_only"][model]),
        }
        external[f"{model}_ms"] = {
            "metrics": means(baseline["external_frozen_10k"]["multisource"][model]),
            "sd": sds(baseline["external_frozen_10k"]["multisource"][model]),
        }
    internal["hyb_orig"] = {
        "metrics": means(hybrid_original["internal_original_test"]),
        "sd": sds(hybrid_original["internal_original_test"]),
    }
    external["hyb_orig"] = {
        "metrics": means(hybrid_original["external_10k"]),
        "sd": sds(hybrid_original["external_10k"]),
    }
    internal["hyb_ms"] = {
        "metrics": means(professor["models"]["hybrid"]["metrics"]),
        "sd": sds(professor["models"]["hybrid"]["metrics"]),
    }
    external["hyb_ms"] = {
        "metrics": {metric: float(hybrid_ms_external["metrics"][metric]) for metric in METRICS},
        "sd": {metric: 0.0 for metric in METRICS},
    }
    return internal, external


def annotation(value: float) -> str:
    percentage = 100 * value
    return f"{percentage:.2f}" if percentage < 1 else f"{percentage:.1f}"


def heatmap(axis: plt.Axes, data: dict, metrics: tuple[str, ...], title: str, errors: bool = False) -> None:
    matrix = np.asarray([[100 * data[key]["metrics"][metric] for metric in metrics] for key, _ in ROWS])
    image = axis.imshow(matrix, cmap="RdYlGn_r" if errors else "RdYlGn", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(np.arange(len(metrics)), [METRIC_LABELS[metric] for metric in metrics], fontsize=10)
    axis.set_yticks(np.arange(len(ROWS)), [label for _, label in ROWS], fontsize=10)
    axis.set_title(title, fontsize=14, fontweight="bold", pad=10)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            normalized_goodness = (100 - value) if errors else value
            color = "white" if normalized_goodness < 27 or normalized_goodness > 82 else "black"
            axis.text(column, row, annotation(value / 100), ha="center", va="center",
                      fontsize=8.2 if len(metrics) > 2 else 9.2, fontweight="semibold", color=color)
    for boundary in (1.5, 3.5, 5.5, 7.5, 9.5):
        axis.axhline(boundary, color="white", linewidth=2.2)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def table_lines(data: dict) -> list[str]:
    result = [
        "| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in ROWS:
        result.append("| " + label + " | " + " | ".join(
            f'{data[key]["metrics"][metric]:.4f}' for metric in METRICS
        ) + " |")
    return result


def main() -> None:
    internal, external = load_results()
    output = Path("artifacts/results/complete_classifier_dashboard")
    output.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 19), facecolor="#F8FAFC")
    grid = fig.add_gridspec(2, 2, width_ratios=(3.1, 1.15), hspace=0.20, wspace=0.10)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    heatmap(axes[0], internal, QUALITY, "A. Internal original test — detection metrics")
    heatmap(axes[1], internal, ERRORS, "B. Internal original test — error rates", errors=True)
    heatmap(axes[2], external, QUALITY, "C. Frozen external 10K — detection metrics")
    heatmap(axes[3], external, ERRORS, "D. Frozen external 10K — error rates", errors=True)
    fig.suptitle(
        "Version 3 complete classifier comparison — Original-only vs Multi-source",
        fontsize=22, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.956,
        "Internal: 49,229 frozen URLs  |  External: 10,000 identity-disjoint URLs  |  Cell values are percentages",
        ha="center", fontsize=13,
    )
    fig.text(
        0.5, 0.025,
        "ORIG = 358,619 original training URLs; MS = 363,692 multi-source real URLs; no GAN rows.  "
        "Baseline and HYB-ORIG cells are means across seeds 42/43/44.  *HYB-MS is one calibrated three-seed ensemble result.  "
        "Higher is better for A/C; lower is better for FPR/FNR in B/D.",
        ha="center", va="bottom", fontsize=11.5, wrap=True,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#E2E8F0", "edgecolor": "#94A3B8"},
    )
    fig.savefig(output / "complete_internal_external_dashboard.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "complete_internal_external_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "project_version": 3,
        "title": "Complete original-only and multi-source classifier comparison",
        "metrics": list(METRICS),
        "conditions": {label: key for key, label in ROWS},
        "training_contract": {
            "ORIG": {"rows": 358619, "gan_rows": 0},
            "MS": {"rows": 363692, "gan_rows": 0},
            "same_stored_ms_rows": True,
            "important_note": "HYB-MS additionally uses source-balanced exposure, PSL-aware fusion, CORAL, ensembling, and separate calibration.",
        },
        "internal_original_test": internal,
        "external_frozen_10k": external,
    }
    write_json(output / "comparison.json", payload)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("evaluation", "condition", *METRICS))
        writer.writeheader()
        for evaluation, data in (("internal_original_test", internal), ("external_frozen_10k", external)):
            for key, label in ROWS:
                writer.writerow({"evaluation": evaluation, "condition": label, **data[key]["metrics"]})

    report = [
        "# Complete Classifier Dashboard", "",
        "This dashboard combines all directly comparable Version 3 classifier conditions on the unchanged internal test and frozen external 10K snapshot.", "",
        "- `ORIG`: 358,619 original real training URLs.",
        "- `MS`: the same 363,692 stored multi-source real URLs for every MS condition.",
        "- No condition in this figure contains GAN-generated training rows.",
        "- Baselines and HYB-ORIG are mean results across seeds 42/43/44.",
        "- `HYB-MS*` is a single three-seed mean-score ensemble using its separately frozen calibration threshold.",
        "- HYB-MS has the same stored MS row pool but additionally uses source-balanced sampling, PSL-aware lexical fusion, conditional CORAL, ensembling, and calibration.", "",
        "## Internal original test (49,229 URLs)", "", *table_lines(internal), "",
        "## Frozen external test (10,000 URLs)", "", *table_lines(external), "",
        "## Reading the graph", "",
        "Green means better. In the detection panels, high values are good. In the error panels, low FPR/FNR values are good, so their colour scale is reversed. The external panels should guide deployment decisions because the internal dataset substantially overstates cross-source performance.", "",
        "The decisive external result is HYB-MS: accuracy 0.9177, F1 0.9120, ROC-AUC 0.9663, FPR 0.0170, and FNR 0.1476. The unchanged MS baselines improve some ranking metrics, but their thresholds still transfer poorly.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
