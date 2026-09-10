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
CONDITIONS = ("real_only", "seq_10", "seq_20", "seq_30", "seq_40",
              "coverage_only", "coverage_seq_10", "coverage_seq_20", "coverage_seq_30", "coverage_seq_40")
LABELS = ("Real", "S10", "S20", "S30", "S40", "Coverage", "Cov+S10", "Cov+S20", "Cov+S30", "Cov+S40")
EVALUATIONS = ("internal", "topdomains_phishtank", "tranco_openphish")


def main() -> None:
    source = load_json(Path("artifacts/results/gan_augmentation_v1/summary/summary.json"))
    output = Path("artifacts/results/gan_augmentation_v1/error_rate_analysis")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    paired = {}
    for evaluation in EVALUATIONS:
        paired[evaluation] = {}
        for condition in CONDITIONS:
            paired[evaluation][condition] = {}
            baseline = "coverage_only" if condition.startswith("coverage_seq") else "real_only"
            for model in MODELS:
                values = source["evaluations"][evaluation][condition][model]
                record = {
                    "evaluation": evaluation, "condition": condition, "model": model,
                    "fpr_mean": values["false_positive_rate"]["mean"],
                    "fpr_sd": values["false_positive_rate"]["sd"],
                    "fnr_mean": values["false_negative_rate"]["mean"],
                    "fnr_sd": values["false_negative_rate"]["sd"],
                }
                rows.append(record)
                base = source["evaluations"][evaluation][baseline][model]
                paired[evaluation][condition][model] = {
                    "baseline": baseline,
                    "fpr_delta": record["fpr_mean"] - base["false_positive_rate"]["mean"],
                    "fnr_delta": record["fnr_mean"] - base["false_negative_rate"]["mean"],
                    "both_improved": record["fpr_mean"] < base["false_positive_rate"]["mean"]
                                     and record["fnr_mean"] < base["false_negative_rate"]["mean"],
                }
    write_json(output / "error_rates.json", {"rows": rows, "paired_comparisons": paired})
    with (output / "error_rates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    x = np.arange(len(CONDITIONS))
    for row_index, evaluation in enumerate(EVALUATIONS):
        for column, metric in enumerate(("fpr", "fnr")):
            axis = axes[row_index, column]
            for model in MODELS:
                selected = [row for row in rows if row["evaluation"] == evaluation and row["model"] == model]
                means = [100 * row[f"{metric}_mean"] for row in selected]
                errors = [100 * row[f"{metric}_sd"] for row in selected]
                axis.errorbar(x, means, yerr=errors, marker="o", capsize=3, linewidth=1.8, label=DISPLAY[model])
            axis.set_title(f"{evaluation.replace('_', ' ').title()} — {metric.upper()}", fontweight="bold")
            axis.set_ylabel("Percent (lower is better)")
            axis.set_xticks(x, LABELS, rotation=32, ha="right")
            axis.grid(alpha=.25)
            axis.axvline(4.5, color="black", linestyle="--", alpha=.35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle("Version 3 false-positive and false-negative rates — mean ± SD across seeds", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(output / "fpr_fnr_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "fpr_fnr_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
