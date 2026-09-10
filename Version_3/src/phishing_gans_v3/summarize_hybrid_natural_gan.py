from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, write_json


ROOT = Path("artifacts/results/hybrid_ms_natural_fair_v1")
EXPERIMENTS = Path("artifacts/experiments")
CONDITIONS = {
    "real_only": ("Real-only", "hybrid_ms_natural_v1", 0, 363692),
    "gan_10pct": ("GAN 10%", "hybrid_ms_natural_gan_10pct_v1", 40410, 404102),
    "gan_20pct": ("GAN 20%", "hybrid_ms_natural_gan_20pct_v1", 90923, 454615),
    "gan_30pct": ("GAN 30%", "hybrid_ms_natural_gan_30pct_v1", 155868, 519560),
    "gan_40pct": ("GAN 40%", "hybrid_ms_natural_gan_40pct_v1", 242461, 606153),
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
DISPLAY = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
           "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
           "false_positive_rate": "FPR", "false_negative_rate": "FNR"}


def aggregate(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)),
            "values": [float(value) for value in values]}


def internal(condition_dir: str) -> dict:
    runs = [load_json(EXPERIMENTS / condition_dir / f"seed_{seed}" / "metrics.json")
            for seed in (42, 43, 44)]
    return {metric: aggregate([run[metric] for run in runs]) for metric in METRICS}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    result = {"experiment": "hybrid_ms_natural_fair_v1", "seeds": [42, 43, 44],
              "sampling": "natural: each stored row exactly once per epoch with seeded shuffling",
              "real_rows": 363692, "conditions": {}}
    rows = []
    for key, (label, directory, synthetic_rows, total_rows) in CONDITIONS.items():
        evaluations = {
            "internal": internal(directory),
            "external_10k": load_json(ROOT / key / "external_10k" / "summary.json")["robust"],
            "external_30k": load_json(ROOT / key / "external_30k" / "summary.json")["robust"],
        }
        result["conditions"][key] = {
            "label": label, "checkpoint_directory": str(EXPERIMENTS / directory),
            "real_rows": 363692, "synthetic_rows": synthetic_rows, "total_rows": total_rows,
            "synthetic_final_share": synthetic_rows / total_rows, "evaluations": evaluations,
        }
        for evaluation, metrics in evaluations.items():
            row = {"condition": key, "label": label, "evaluation": evaluation,
                   "real_rows": 363692, "synthetic_rows": synthetic_rows, "total_rows": total_rows}
            for metric in METRICS:
                row[f"{metric}_mean"] = metrics[metric]["mean"]
                row[f"{metric}_sd"] = metrics[metric]["standard_deviation"]
            rows.append(row)
    baseline = result["conditions"]["real_only"]["evaluations"]
    for key, condition in result["conditions"].items():
        condition["deltas_from_real_only"] = {
            evaluation: {metric: condition["evaluations"][evaluation][metric]["mean"] - baseline[evaluation][metric]["mean"]
                         for metric in METRICS}
            for evaluation in ("internal", "external_10k", "external_30k")
        }
    write_json(ROOT / "summary.json", result)
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    labels = [value[0] for value in CONDITIONS.values()]
    colors = ["#355C7D", "#4ECDC4", "#F6BD60", "#F28482", "#9B5DE5"]
    figure, axes = plt.subplots(3, 4, figsize=(18, 12))
    for row_index, evaluation in enumerate(("internal", "external_10k", "external_30k")):
        for column_index, metric in enumerate(("accuracy", "recall", "f1", "roc_auc")):
            axis = axes[row_index, column_index]
            means = [100 * result["conditions"][key]["evaluations"][evaluation][metric]["mean"] for key in CONDITIONS]
            errors = [100 * result["conditions"][key]["evaluations"][evaluation][metric]["standard_deviation"] for key in CONDITIONS]
            axis.bar(np.arange(5), means, yerr=errors, capsize=3, color=colors)
            axis.set_xticks(np.arange(5), labels, rotation=20, ha="right")
            axis.set_ylim(0, 102); axis.grid(axis="y", alpha=.22)
            axis.set_title(DISPLAY[metric], fontweight="bold")
            if column_index == 0:
                axis.set_ylabel({"internal": "Internal test (%)", "external_10k": "Frozen 10K (%)",
                                 "external_30k": "Imbalanced 30K (%)"}[evaluation], fontweight="bold")
    figure.suptitle("Fair HYB-MS: Natural Sampling + Frozen GAN Pools\nmean ± SD across seeds 42/43/44",
                    fontsize=18, fontweight="bold")
    figure.text(.5, .01, "Same 363,692 real URLs in every condition • no source-weighted resampling • registered robust thresholds",
                ha="center", fontsize=11)
    figure.tight_layout(rect=(0, .035, 1, .93))
    figure.savefig(ROOT / "comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(ROOT / "comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    lines = ["# Fair HYB-MS Natural-Sampling GAN Experiment", "",
             "Every condition contains the exact same 363,692 real URLs. Each stored row is visited once per epoch with seeded shuffling. The augmented conditions add the already frozen nested GAN pools; no 65/35 source-weighted sampler is used.", "",
             "Metrics are mean ± sample standard deviation across seeds 42/43/44. Thresholds were frozen from validation/domain-validation and were not tuned on either external set.", ""]
    for evaluation, title in (("internal", "Internal frozen test"), ("external_10k", "Frozen external 10K"),
                              ("external_30k", "Imbalanced external 30K")):
        lines += [f"## {title}", "", "| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key in CONDITIONS:
            condition = result["conditions"][key]
            values = condition["evaluations"][evaluation]
            lines.append("| " + condition["label"] + " | " + " | ".join(
                f"{values[m]['mean']:.4f} ± {values[m]['standard_deviation']:.4f}" for m in METRICS) + " |")
        lines.append("")
    lines += ["## Interpretation", "",
              "- GAN 10% gives a very small internal threshold-metric increase over real-only, but its internal ROC-AUC and PR-AUC decrease.",
              "- On both frozen external sets, real-only is best among these natural-sampling conditions for accuracy, recall, F1, ROC-AUC, PR-AUC, and FNR.",
              "- Increasing the GAN share causes a generally monotonic external ranking decline. On the 30K set, ROC-AUC falls from 0.9399 (real-only) to 0.8614 (GAN 40%).",
              "- This means the current GAN pool does not replace real external coverage. Its synthetic phishing distribution increasingly pulls the model away from later real URLs.",
              "- The experiment is fair with respect to stored real rows and sampling. It does not claim identical feature representations between fundamentally different baseline architectures.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    with PdfPages(ROOT / "REPORT.pdf") as pdf:
        page = plt.figure(figsize=(11.7, 8.3)); page.patch.set_facecolor("white")
        page.text(.06, .91, "Fair HYB-MS Natural-Sampling GAN Experiment", fontsize=22, fontweight="bold", color="#203354")
        page.text(.06, .84, "Same 363,692 real URLs • natural sampling • seeds 42/43/44", fontsize=13)
        narrative = ("Outcome: real-only transfers best. GAN 10% provides only a tiny internal F1 increase, while all "
                     "GAN conditions reduce external ranking and threshold performance. Higher synthetic shares worsen "
                     "the decline, indicating synthetic-to-real distribution mismatch rather than useful new coverage.")
        page.text(.06, .73, narrative, fontsize=14, wrap=True, va="top", linespacing=1.5)
        y = .54
        for evaluation, title in (("internal", "Internal"), ("external_10k", "External 10K"), ("external_30k", "External 30K")):
            best = max(CONDITIONS, key=lambda key: result["conditions"][key]["evaluations"][evaluation]["f1"]["mean"])
            values = result["conditions"][best]["evaluations"][evaluation]
            page.text(.08, y, f"{title}: best F1 = {result['conditions'][best]['label']} ({values['f1']['mean']:.4f}); "
                      f"ROC-AUC {values['roc_auc']['mean']:.4f}", fontsize=13)
            y -= .09
        page.text(.06, .14, "External labels were evaluation-only; no threshold was selected from the 10K or 30K test sets.", fontsize=11)
        pdf.savefig(page, bbox_inches="tight"); plt.close(page)
        image = plt.imread(ROOT / "comparison.png")
        page = plt.figure(figsize=(11.7, 8.3)); axis = page.add_axes((.02, .02, .96, .96)); axis.imshow(image); axis.axis("off")
        pdf.savefig(page, bbox_inches="tight"); plt.close(page)

    print(json.dumps({"output": str(ROOT), "report": str(ROOT / "REPORT.pdf")}, indent=2))


if __name__ == "__main__":
    main()
