from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc",
           "false_positive_rate", "false_negative_rate")
CONDITIONS = (
    ("original", "lstm", "Original LSTM"),
    ("outside_coverage", "lstm", "Coverage LSTM"),
    ("urls_new_balanced", "lstm", "Balanced LSTM"),
)
SETS = {
    "topdomains_phishtank": (
        Path("artifacts/results/urls_new_structure_balanced_v1/external_topdomains_phishtank/three_conditions/comparison.json"),
        Path("artifacts/results/hybrid_cnn_coral_v1/external_topdomains_phishtank/summary.json"),
        "Top-Domain / PhishTank (500)",
    ),
    "openphish_tranco": (
        Path("artifacts/results/urls_new_structure_balanced_v1/external_openphish_tranco/three_conditions/comparison.json"),
        Path("artifacts/results/hybrid_cnn_coral_v1/external_openphish_tranco/summary.json"),
        "Tranco / OpenPhish (400)",
    ),
}


def main() -> None:
    output_dir = Path("artifacts/results/hybrid_cnn_coral_v1/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"comparison_status": "exploratory, not confirmatory", "datasets": {}}
    for name, (prior_path, hybrid_path, title) in SETS.items():
        prior, hybrid = load_json(prior_path), load_json(hybrid_path)
        if prior["dataset_sha256"] != hybrid["dataset_sha256"]:
            raise RuntimeError(f"Dataset mismatch for {name}")
        systems = {}
        for condition, model, label in CONDITIONS:
            systems[label] = prior["conditions"][condition][model]["metrics"]
        systems["Hybrid robust"] = hybrid["robust"]
        systems["Hybrid original-threshold"] = hybrid["original_threshold"]
        payload["datasets"][name] = {
            "title": title, "rows": hybrid["rows"], "dataset_sha256": hybrid["dataset_sha256"],
            "systems": systems,
        }
    write_json(output_dir / "comparison.json", payload)

    labels = [item[2] for item in CONDITIONS] + ["Hybrid robust", "Hybrid original-threshold"]
    colors = ("#9EA3A8", "#6C8EBF", "#B4A7D6", "#2A9D8F", "#E76F51")
    for dataset in payload["datasets"].values():
        fig, axes = plt.subplots(2, 4, figsize=(17, 9))
        for axis, metric in zip(axes.ravel(), METRICS):
            means = [100 * dataset["systems"][label][metric]["mean"] for label in labels]
            errors = [100 * dataset["systems"][label][metric]["standard_deviation"] for label in labels]
            axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=3, color=colors)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
            axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
            axis.set_ylim(0, 100); axis.grid(axis="y", alpha=.25)
        axes.ravel()[-1].axis("off")
        fig.suptitle(f"External distribution-shift comparison — {dataset['title']}\nmean ± SD, seeds 42/43/44",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .93))
        slug = dataset["title"].split(" (")[0].lower().replace(" / ", "_").replace("-", "_")
        fig.savefig(output_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
        fig.savefig(output_dir / f"{slug}.pdf", bbox_inches="tight")
        plt.close(fig)

    lines = ["# Hybrid External Comparison", "",
             "This is an exploratory comparison on the same frozen rows used in the earlier shift diagnosis. A later untouched snapshot is required for confirmation.", ""]
    for dataset in payload["datasets"].values():
        lines.extend((f"## {dataset['title']}", "",
                      "| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"))
        for label in labels:
            row = dataset["systems"][label]
            lines.append("| " + label + " | " + " | ".join(
                f"{row[metric]['mean']:.4f} ± {row[metric]['standard_deviation']:.4f}" for metric in METRICS
            ) + " |")
        lines.append("")
    lines.extend(("## Interpretation", "",
                  "The hybrid sharply improves external ranking (ROC-AUC) and legitimate rejection (FPR). The conservative robust threshold nearly eliminates false positives but misses many phishing URLs. The original-validation threshold recovers high phishing recall but still has a 22–25% external FPR. This means the representation shift is greatly reduced, while operating-point calibration remains unresolved.", "",
                  "Do not select between these thresholds using these external labels. Freeze a deployment-like calibration set, calibrate once, and confirm once on a genuinely later untouched snapshot."))
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "datasets": list(payload["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
