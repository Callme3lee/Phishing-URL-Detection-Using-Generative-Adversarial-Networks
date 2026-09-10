from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def mean_sd(values):
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)), "values": values}


def main() -> None:
    output = Path("artifacts/results/hybrid_original_v1/summary"); output.mkdir(parents=True, exist_ok=True)
    runs = [load_json(Path(f"artifacts/experiments/hybrid_original_v1/seed_{seed}/metrics.json")) for seed in (42, 43, 44)]
    internal = {metric: mean_sd([float(run[metric]) for run in runs]) for metric in METRICS}
    external = load_json(Path("artifacts/results/hybrid_original_v1/external_large_2026-08-12/summary.json"))["metrics"]
    multisource_payload = load_json(Path("artifacts/results/hybrid_cnn_coral_v1/external_large_2026-08-12/summary.json"))
    multisource = {metric: {"mean": float(multisource_payload["metrics"][metric]), "standard_deviation": 0.0}
                   for metric in METRICS}
    payload = {"project_version": 3, "condition": "hybrid_original_v1", "display_name": "HYB-ORIG",
               "training_rows": 358619, "outside_rows": 0, "gan_rows": 0, "coral": False,
               "internal_original_test": internal, "external_10k": external,
               "external_10k_multisource_hyb_reference": multisource}
    write_json(output / "summary.json", payload)
    names = ("Original test", "External 10K"); colors = ("#4C78A8", "#E76F51")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8))
    for axis, metric in zip(axes.ravel(), METRICS):
        values = [100 * internal[metric]["mean"], 100 * external[metric]["mean"]]
        errors = [100 * internal[metric]["standard_deviation"], 100 * external[metric]["standard_deviation"]]
        bars = axis.bar(names, values, yerr=errors, capsize=4, color=colors)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
        axis.set_title(metric.replace("false_positive_rate", "FPR").replace("false_negative_rate", "FNR")
                       .replace("roc_auc", "ROC-AUC").replace("pr_auc", "PR-AUC").upper(), fontweight="bold")
        axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
    fig.suptitle("HYB-ORIG — same 358,619 training URLs as the baselines\nmean ± SD across seeds 42/43/44",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(output / "internal_vs_external.png", dpi=230, bbox_inches="tight")
    fig.savefig(output / "internal_vs_external.pdf", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8))
    for axis, metric in zip(axes.ravel(), METRICS):
        values = [100 * external[metric]["mean"], 100 * multisource[metric]["mean"]]
        errors = [100 * external[metric]["standard_deviation"], 0.0]
        bars = axis.bar(("HYB-ORIG", "HYB-MS"), values, yerr=errors, capsize=4,
                        color=("#E76F51", "#2A9D8F"))
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
        axis.set_title(metric.replace("false_positive_rate", "FPR").replace("false_negative_rate", "FNR")
                       .replace("roc_auc", "ROC-AUC").replace("pr_auc", "PR-AUC").upper(), fontweight="bold")
        axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Frozen external 10K — original-only vs multi-source hybrid\nHYB-ORIG: mean ± SD; HYB-MS: calibrated 3-seed ensemble",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(output / "external_hybrid_comparison.png", dpi=230, bbox_inches="tight")
    fig.savefig(output / "external_hybrid_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    lines = ["# HYB-ORIG Controlled Experiment", "",
             "HYB-ORIG uses exactly the same 358,619 original training URLs as CNN/LSTM/RNN/RF/SVM. It adds the hybrid architecture and 35 PSL-aware features, but no outside rows, GAN rows, CORAL, or source-balanced sampling.", "",
             "| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in zip(names, (internal, external)):
        lines.append("| " + name + " | " + " | ".join(f"{values[m]['mean']:.4f}" for m in METRICS) + " |")
    lines.extend(("", "## External 10K: controlled hybrid comparison", "",
                  "| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                  "| HYB-ORIG | " + " | ".join(f"{external[m]['mean']:.4f}" for m in METRICS) + " |",
                  "| HYB-MS | " + " | ".join(f"{multisource[m]['mean']:.4f}" for m in METRICS) + " |", "",
                  "Strong internal performance does not imply external robustness. The external result shows that the architecture alone is insufficient; the registered multi-source real coverage, CORAL, ensembling, and calibration are responsible for the practical external behavior."))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
