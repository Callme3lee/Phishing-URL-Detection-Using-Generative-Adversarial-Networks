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


ROOT = Path("artifacts/results/external/2026-08-13-imbalanced-30k")
REPORT = Path("data/external/2026-08-13-imbalanced-30k/collection_report.json")
METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)
HIGHER = METRICS[:6]
SHORT = {
    "accuracy": "ACC", "precision": "PREC", "recall": "REC", "f1": "F1",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
}
BASE_DISPLAY = {
    "cnn": "CNN", "lstm": "LSTM", "rnn": "RNN",
    "random_forest": "RF", "linear_svm": "SVM",
}
FAIR_DISPLAY = {
    "random_forest": "RF-PSL", "linear_svm": "SVM-PSL",
    "cnn_psl": "CNN+PSL", "hyb_ms": "HYB-MS",
}
COND_DISPLAY = {
    "real_only": "Real", "gan_10pct": "+10%", "gan_20pct": "+20%",
    "gan_30pct": "+30%", "gan_40pct": "+40%",
}


def mean_metrics(summary: dict, prefix: str) -> list[dict]:
    rows = []
    for model, payload in summary["grouped"].items():
        rows.append({
            "model": f"{prefix}-{BASE_DISPLAY[model]}", "family": prefix,
            "condition": "Real", "runs": 3,
            **{metric: float(payload["metrics"][metric]["mean"]) for metric in METRICS},
        })
    return rows


def rank_rows(rows: list[dict]) -> list[dict]:
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        order = np.argsort(-values if metric in HIGHER else values)
        for rank, index in enumerate(order, start=1):
            rows[index][f"rank_{metric}"] = rank
    # Equal weight across the eight requested metrics; errors rank ascending.
    for row in rows:
        row["mean_metric_rank"] = float(np.mean([row[f"rank_{metric}"] for metric in METRICS]))
    return sorted(rows, key=lambda row: (row["mean_metric_rank"], -row["f1"]))


def metric_table(axis, rows: list[dict], title: str) -> None:
    axis.axis("off")
    cells = []
    for row in rows:
        cells.append([
            row["model"], row["condition"],
            *[f"{100 * row[metric]:.2f}" for metric in METRICS],
            f"{row['mean_metric_rank']:.1f}",
        ])
    table = axis.table(
        cellText=cells,
        colLabels=["Model", "Cond.", *[SHORT[m] for m in METRICS], "Mean\nrank"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(7.2); table.scale(1, 1.45)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#17324D"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EDF3F8")
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)


def main() -> None:
    collection = load_json(REPORT)
    original = load_json(ROOT / "original_baselines" / "summary.json")
    multisource = load_json(ROOT / "multisource_baselines" / "summary.json")
    fair = load_json(ROOT / "fair_conditions" / "external_10k.json")
    hybrid = load_json(ROOT / "calibrated_hybrid_ensembles.json")
    hashes = {original["dataset_sha256"], multisource["dataset_sha256"], fair["dataset_sha256"], hybrid["external_sha256"]}
    if hashes != {collection["dataset_sha256"]}:
        raise RuntimeError("Not every result uses the same frozen 30K dataset")

    rows = mean_metrics(original, "ORIG") + mean_metrics(multisource, "MS")
    for condition, models in fair["grouped"].items():
        for model, policies in models.items():
            if model == "hyb_ms":
                continue
            payload = policies["native"]
            rows.append({
                "model": FAIR_DISPLAY[model], "family": "Feature-fair",
                "condition": COND_DISPLAY[condition], "runs": 3,
                **{metric: float(payload[metric]["mean"]) for metric in METRICS},
            })
    for condition, payload in hybrid["results"].items():
        metrics = payload["external_10k"]
        rows.append({
            "model": "HYB-MS", "family": "Calibrated ensemble",
            "condition": COND_DISPLAY[condition], "runs": 1,
            **{metric: float(metrics[metric]) for metric in METRICS},
        })
    ranked = rank_rows(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "all_models_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["overall_rank", "model", "family", "condition", "runs", *METRICS, "mean_metric_rank"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(ranked, start=1):
            writer.writerow({"overall_rank": index, **{field: row[field] for field in fields if field != "overall_rank"}})

    winner = ranked[0]
    hybrid_rows = [row for row in ranked if row["model"] == "HYB-MS"]
    payload = {
        "project_version": 3, "dataset": collection["dataset"],
        "dataset_sha256": collection["dataset_sha256"], "rows": collection["rows"],
        "legitimate_rows": collection["legitimate_rows"], "phishing_rows": collection["phishing_rows"],
        "phishing_prevalence": collection["class_count_selection"]["selected_phishing_prevalence"],
        "configurations_compared": len(ranked), "ranking_method": (
            "mean ordinal rank across Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR and FNR; "
            "higher is better for the first six and lower is better for FPR/FNR"
        ),
        "winner": winner, "ranked": ranked,
        "guard": "The composite is descriptive and equal-weighted; operational cost may justify a different metric weighting.",
    }
    write_json(ROOT / "comparison.json", payload)

    labels = [f"{row['model']}\n{row['condition']}" for row in hybrid_rows]
    x = np.arange(len(hybrid_rows))
    colors = ["#355C7D" if row["condition"] == "Real" else "#2A9D8F" if row["condition"] == "+10%" else "#E9C46A" for row in hybrid_rows]
    with PdfPages(ROOT / "imbalanced_30k_complete_report.pdf") as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); axis = fig.add_subplot(111); axis.axis("off")
        axis.text(.5, .83, "New Imbalanced 30K External Evaluation", ha="center", fontsize=24, weight="bold", color="#17324D")
        axis.text(.5, .72, f"{collection['legitimate_rows']:,} legitimate + {collection['phishing_rows']:,} phishing ({100*payload['phishing_prevalence']:.2f}% phishing)", ha="center", fontsize=15)
        axis.text(.5, .61, f"30,000 unique URLs, hostnames and encoded identities\nSHA-256: {collection['dataset_sha256']}", ha="center", fontsize=10)
        axis.text(.5, .45, f"Best balanced configuration: {winner['model']} {winner['condition']}\nF1 {100*winner['f1']:.2f}%  |  ROC-AUC {100*winner['roc_auc']:.2f}%  |  FPR {100*winner['false_positive_rate']:.2f}%", ha="center", fontsize=17, weight="bold", color="#1B7F6B")
        axis.text(.5, .24, "All checkpoints and thresholds were frozen before this test.\nNo retraining, recalibration, or test-set threshold tuning.", ha="center", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for start in range(0, len(ranked), 10):
            fig, axis = plt.subplots(figsize=(11.69, 8.27))
            metric_table(axis, ranked[start:start+10], f"Complete metrics — overall ranks {start+1} to {min(start+10, len(ranked))}")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
        for axis, metric in zip(axes.ravel(), METRICS):
            values = [100 * row[metric] for row in hybrid_rows]
            bars = axis.bar(x, values, color=colors)
            axis.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=8, padding=2)
            axis.set_title(SHORT[metric], weight="bold"); axis.set_xticks(x, labels, fontsize=8)
            axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
        fig.suptitle("Calibrated HYB-MS ensembles on the frozen imbalanced 30K", fontsize=16, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        top = ranked[:10]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        y = np.arange(len(top))
        axes[0].barh(y, [100*r["f1"] for r in top], color="#2A9D8F"); axes[0].invert_yaxis()
        axes[0].set_yticks(y, [f"{r['model']} {r['condition']}" for r in top]); axes[0].set_title("Top 10 by composite order — F1")
        axes[1].barh(y, [100*r["false_positive_rate"] for r in top], color="#E76F51"); axes[1].invert_yaxis()
        axes[1].set_yticks(y, [f"{r['model']} {r['condition']}" for r in top]); axes[1].set_title("False-positive rate (lower is better)")
        for axis in axes: axis.grid(axis="x", alpha=.25); axis.set_xlabel("Percent")
        fig.suptitle("Leading configurations and browser false-alarm trade-off", fontsize=16, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27)); axis = fig.add_subplot(111); axis.axis("off")
        real = next(row for row in hybrid_rows if row["condition"] == "Real")
        gan10 = next(row for row in hybrid_rows if row["condition"] == "+10%")
        lines = [
            "Interpretation",
            "",
            f"HYB-MS +10% is best overall under the stated equal-weight metric ranking ({gan10['mean_metric_rank']:.1f}).",
            f"Compared with calibrated real-only, F1 changes {100*real['f1']:.2f}% → {100*gan10['f1']:.2f}%.",
            f"Recall changes {100*real['recall']:.2f}% → {100*gan10['recall']:.2f}%; FNR changes {100*real['false_negative_rate']:.2f}% → {100*gan10['false_negative_rate']:.2f}%.",
            f"FPR changes {100*real['false_positive_rate']:.2f}% → {100*gan10['false_positive_rate']:.2f}%.",
            "",
            "The +20% to +40% conditions reduce FPR further, but lose too much phishing recall and F1.",
            "The result confirms the earlier frozen 10K direction without changing the chosen +10% condition.",
            "",
            "Limitations",
            "• Tranco roots are legitimate proxies rather than individually audited pages.",
            "• Legitimate roots and phishing full URLs differ structurally.",
            "• PhishTank's current feed contains still-online historical reports, so this is not wholly temporal.",
            "• 30.07% phishing is a seeded stress-test prevalence, not measured browser prevalence.",
            "• Strong metrics support a deployment candidate, not deployment readiness by themselves.",
        ]
        axis.text(.06, .94, "\n".join(lines), va="top", fontsize=13, linespacing=1.45)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    report_lines = [
        "# Imbalanced 30K External Evaluation", "",
        f"Frozen dataset: {collection['rows']:,} URLs — {collection['legitimate_rows']:,} legitimate and {collection['phishing_rows']:,} phishing ({100*payload['phishing_prevalence']:.2f}% phishing).",
        f"SHA-256: `{collection['dataset_sha256']}`.", "",
        "Class count was selected by seeded random draw before evaluating any model. All rows are URL-, hostname-, and encoded-identity-disjoint from training, development, calibration and prior external snapshots.", "",
        f"Best equal-weight overall configuration: **{winner['model']} {winner['condition']}**.", "",
        "| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        report_lines.append("| " + " | ".join([
            row["model"], row["condition"], *[f"{row[m]:.4f}" for m in METRICS], f"{row['mean_metric_rank']:.1f}",
        ]) + " |")
    report_lines.extend(["", "The composite ranking is descriptive. Deployment metric weights must reflect real false-alarm and missed-phish costs."])
    (ROOT / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "configurations": len(ranked), "output": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
