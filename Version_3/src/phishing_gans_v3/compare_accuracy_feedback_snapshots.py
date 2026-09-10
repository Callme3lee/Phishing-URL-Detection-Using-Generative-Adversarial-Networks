from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
DISPLAY = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the independent 5% and 10% fresh snapshots.")
    parser.add_argument("--five", type=Path, default=Path("artifacts/results/accuracy_feedback_pretest_2026-08-24_5pct"))
    parser.add_argument("--ten", type=Path, default=Path("artifacts/results/accuracy_feedback_pretest_2026-08-24_10pct"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/accuracy_feedback_pretest_2026-08-24_comparison"))
    return parser.parse_args()


def pct(value: float, signed: bool = False) -> str:
    return f"{100 * value:+.2f}%" if signed else f"{100 * value:.2f}%"


def graph(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), facecolor="white")
    for axis, metric, label in zip(axes.ravel(), METRICS, DISPLAY):
        values = [100 * payload["snapshots"][key]["deltas_gan_minus_real"][metric] for key in ("five_percent", "ten_percent")]
        bars = axis.bar(["5% phishing", "10% phishing"], values, color=["#5B8DB8", "#E67E22"], width=.62)
        axis.axhline(0, color="#526678", linewidth=.8)
        axis.set_title(label, fontsize=13, weight="bold", color="#17324D")
        axis.grid(axis="y", alpha=.22)
        axis.bar_label(bars, labels=[f"{value:+.2f} pp" for value in values], padding=3, fontsize=9)
        margin = max(.4, max(abs(value) for value in values) * .35)
        axis.set_ylim(min(0, min(values)) - margin, max(0, max(values)) + margin)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GAN Minus Real-Only Across Two Independent Fresh Snapshots", fontsize=21, weight="bold", color="#17324D")
    fig.text(.5, .935, "Positive is better except FPR/FNR, where negative is better", ha="center", fontsize=12, color="#526678")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_pdf(path: Path, payload: dict, graph_path: Path) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .92, "Two Fresh Frozen Evaluations", fontsize=24, weight="bold", color="#17324D")
        fig.text(.055, .865, "Independent cohorts • no URL/hostname overlap • identical models and thresholds", fontsize=12, color="#3B6EA8")
        columns = ["Cohort", "Condition", *DISPLAY]
        rows = []
        for cohort_key, cohort_label in (("five_percent", "5% phishing"), ("ten_percent", "10% phishing")):
            snapshot = payload["snapshots"][cohort_key]
            for condition_key, condition_label in (("real_only", "Real-only"), ("gan_10pct", "+10% GAN")):
                metrics = snapshot["conditions"][condition_key]
                rows.append([cohort_label, condition_label, *[pct(metrics[key]) for key in METRICS]])
        axis = fig.add_axes([.025, .48, .95, .29]); axis.axis("off")
        table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center",
                           colWidths=[.11, .13, .09, .10, .08, .08, .09, .09, .075, .075])
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.95)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0: cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
            elif row in (2, 4): cell.set_facecolor("#E7F3F1")
        fig.text(.055, .35, "Stable finding", fontsize=15, weight="bold", color="#17324D")
        fig.text(.055, .285, "Across both independent cohorts, GAN augmentation improved accuracy, precision, recall, F1 and PR-AUC while reducing both FPR and FNR.", fontsize=11.5, color="#526678")
        fig.text(.055, .205, "ROC-AUC was mixed: -0.13 points on the 5% cohort and +0.41 points on the 10% cohort.", fontsize=11.5, color="#526678")
        fig.text(.055, .12, "The higher 10% recall is not caused by prevalence itself: the second cohort contains a different, older active-phishing slice that proved easier for both models.", fontsize=10.8, weight="bold", color="#B45309")
        pdf.savefig(fig); plt.close(fig)

        image = plt.imread(graph_path)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .93, "Augmentation Deltas", fontsize=23, weight="bold", color="#17324D")
        axis = fig.add_axes([.045, .07, .91, .80]); axis.imshow(image); axis.axis("off")
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Comparison already exists at {args.output}")
    snapshots = {}
    for key, root in (("five_percent", args.five), ("ten_percent", args.ten)):
        result = load_json(root / "results.json")
        paired = load_json(root / "paired_analysis.json")
        if paired["dataset"]["sha256"] != result["dataset"]["sha256"]:
            raise RuntimeError(f"Dataset hash mismatch in {root}")
        snapshots[key] = {
            "dataset": paired["dataset"],
            "conditions": paired["conditions"],
            "deltas_gan_minus_real": paired["deltas_gan_minus_real"],
            "strict_rule": paired["strict_rule"],
            "paired_all_rows": paired["paired_exact_comparison"]["all_rows"],
            "result_sha256": sha256_file(root / "results.json"),
            "paired_analysis_sha256": sha256_file(root / "paired_analysis.json"),
        }
    payload = {
        "comparison": "independent August 24 pre-feedback-database snapshots",
        "snapshots": snapshots,
        "stable_findings": [
            "GAN improved accuracy, precision, recall, F1 and PR-AUC in both cohorts.",
            "GAN reduced FPR and FNR in both cohorts.",
            "ROC-AUC direction was mixed across the cohorts.",
            "The 10% phishing cohort used different older still-online PhishTank URLs, so absolute recall differences are distribution differences rather than a prevalence effect.",
        ],
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "snapshot_comparison.json", payload)
    graph_path = args.output / "augmentation_deltas.png"
    graph(graph_path, payload)
    make_pdf(args.output / "snapshot_comparison.pdf", payload, graph_path)
    five = snapshots["five_percent"]; ten = snapshots["ten_percent"]
    report = f"""# Comparison of the two fresh snapshots

Both snapshots used the same frozen real-only and +10% GAN ensembles and unchanged thresholds. They share no exact URLs or hostnames.

The GAN model improved accuracy, precision, recall, F1 and PR-AUC and reduced FPR/FNR in both independent cohorts. The strict augmentation rule failed on the 5% cohort only because ROC-AUC changed by {pct(five['deltas_gan_minus_real']['roc_auc'], True)}, then passed on the 10% cohort where ROC-AUC changed by {pct(ten['deltas_gan_minus_real']['roc_auc'], True)}.

The second cohort should not be interpreted as proof that increasing phishing prevalence improves the model. It uses a different phishing slice: 1,000 still-online PhishTank URLs verified June 16–25, while the first cohort used newer eligible OpenPhish/PhishTank URLs. The difference in recall therefore reflects sample distribution, not prevalence alone.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strict_rule": {
        key: snapshots[key]["strict_rule"]["supported"] for key in snapshots}}, indent=2))


if __name__ == "__main__":
    main()
