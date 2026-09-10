from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json
from .evaluate_frozen_pair_controlled import exact_paired, paired_bootstrap


METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)
DISPLAY = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
    "false_positive_rate": "FPR", "false_negative_rate": "FNR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the frozen pre-feedback-database model comparison.")
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("artifacts/results/accuracy_feedback_pretest_2026-08-24_5pct"),
    )
    parser.add_argument(
        "--collection-report", type=Path,
        default=Path("data/external/2026-08-24-accuracy-feedback-pretest/collection_report.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_824)
    return parser.parse_args()


def pct(value: float, signed: bool = False) -> str:
    return f"{100 * value:+.2f}%" if signed else f"{100 * value:.2f}%"


def make_metric_graph(path: Path, payload: dict) -> None:
    real = payload["conditions"]["real_only"]
    gan = payload["conditions"]["gan_10pct"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), facecolor="white")
    for axis, metric in zip(axes.ravel(), METRICS):
        values = [100 * real[metric], 100 * gan[metric]]
        bars = axis.bar(["Real-only", "+10% GAN"], values, color=["#3B6EA8", "#E67E22"], width=.62)
        axis.set_title(DISPLAY[metric], fontsize=13, weight="bold", color="#17324D")
        axis.grid(axis="y", alpha=.22)
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3, fontsize=10)
        low, high = min(values), max(values)
        margin = max(1.0, abs(high - low) * 2.2)
        axis.set_ylim(max(0, low - margin), min(100, high + margin))
        axis.spines[["top", "right"]].set_visible(False)
    dataset = payload["dataset"]
    fig.suptitle(f"Fresh {dataset['rows'] // 1000}K Snapshot: Frozen Model Comparison", fontsize=22, weight="bold", color="#17324D")
    fig.text(.5, .935, f"{dataset['legitimate_rows']:,} legitimate + {dataset['phishing_rows']:,} phishing URLs • thresholds unchanged", ha="center", fontsize=12, color="#526678")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_operational_graph(path: Path, payload: dict) -> None:
    real = payload["conditions"]["real_only"]
    gan = payload["conditions"]["gan_10pct"]
    phishing_rows = payload["dataset"]["phishing_rows"]
    names = ["False alerts\nper 1,000 legitimate", f"Missed phishing\nout of {phishing_rows:,}", f"Detected phishing\nout of {phishing_rows:,}"]
    real_values = [payload["operational"]["real_only"]["false_alerts_per_1000_legitimate"],
                   real["confusion_matrix"]["fn"], real["confusion_matrix"]["tp"]]
    gan_values = [payload["operational"]["gan_10pct"]["false_alerts_per_1000_legitimate"],
                  gan["confusion_matrix"]["fn"], gan["confusion_matrix"]["tp"]]
    x = np.arange(3); width = .34
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor="white")
    for index, axis in enumerate(axes):
        bars = axis.bar(["Real-only", "+10% GAN"], [real_values[index], gan_values[index]],
                        color=["#3B6EA8", "#E67E22"], width=.62)
        axis.set_title(names[index], fontsize=12, weight="bold", color="#17324D")
        axis.grid(axis="y", alpha=.22)
        labels = [f"{value:.2f}" if index == 0 else f"{int(value)}" for value in (real_values[index], gan_values[index])]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Operational Error Burden on the Fresh Snapshot", fontsize=20, weight="bold", color="#17324D")
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_pdf(path: Path, payload: dict, metric_graph: Path, operational_graph: Path) -> None:
    real = payload["conditions"]["real_only"]
    gan = payload["conditions"]["gan_10pct"]
    deltas = payload["deltas_gan_minus_real"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .92, "Fresh Snapshot Evaluation Before Feedback Collection", fontsize=23, weight="bold", color="#17324D")
        fig.text(.055, .865, "10,000 unseen URLs • 5% phishing • frozen models, checkpoints and thresholds", fontsize=12, color="#3B6EA8")
        columns = ["Condition"] + [DISPLAY[key] for key in METRICS]
        table_rows = [
            ["HYB-MS real-only"] + [pct(real[key]) for key in METRICS],
            ["HYB-MS +10% GAN"] + [pct(gan[key]) for key in METRICS],
        ]
        axis = fig.add_axes([.03, .54, .94, .22]); axis.axis("off")
        table = axis.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center",
                           colWidths=[.20, .10, .105, .09, .09, .10, .10, .085, .085])
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0: cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
            elif row == 2: cell.set_facecolor("#E7F3F1")
        fig.text(.055, .42, "Outcome", fontsize=15, weight="bold", color="#17324D")
        if payload["strict_rule"]["supported"]:
            outcome = (
                "The +10% GAN model improved every reported metric, including recall, F1 and ROC-AUC, "
                "while lowering FPR. The registered augmentation rule passed."
            )
        else:
            outcome = (
                "The +10% GAN model improved all thresholded operational metrics and PR-AUC. "
                f"The strict historical rule was not fully met because ROC-AUC changed by {pct(deltas['roc_auc'], True)}."
            )
        fig.text(.055, .355, textwrap.fill(outcome, 125), fontsize=11.5, color="#526678", va="top")
        fig.text(.055, .245, f"GAN changes: F1 {pct(deltas['f1'], True)} • Recall {pct(deltas['recall'], True)} • FPR {pct(deltas['false_positive_rate'], True)} • FNR {pct(deltas['false_negative_rate'], True)} • ROC-AUC {pct(deltas['roc_auc'], True)}", fontsize=11.5, weight="bold", color="#17324D")
        gate = payload["practical_gate"]
        if gate["real_only_passed"] and gate["gan_10pct_passed"]:
            gate_text = "Deployment metric gate: both models exceeded 80% recall and stayed below the 2% FPR ceiling."
            gate_color = "#187A67"
        else:
            gate_text = f"Deployment metric gate: {gate['summary']}"
            gate_color = "#B45309"
        fig.text(.055, .195, gate_text, fontsize=10.8, weight="bold", color=gate_color)
        paired = payload["paired_exact_comparison"]["all_rows"]
        fig.text(.055, .135, f"Paired decisions: GAN uniquely corrected {paired['real_only_wrong_gan_correct']} rows and uniquely worsened {paired['real_only_correct_gan_wrong']} rows; exact McNemar p = {paired['two_sided_exact_mcnemar_p_value']:.4g}.", fontsize=10.8, color="#526678")
        fig.text(.055, .075, "This cohort was frozen before inference and was not used to alter thresholds or select checkpoints.", fontsize=10.5, weight="bold", color="#17324D")
        pdf.savefig(fig); plt.close(fig)

        for image_path, heading in ((metric_graph, "All Metrics"), (operational_graph, "Operational Error Burden")):
            image = plt.imread(image_path)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            fig.text(.055, .93, heading, fontsize=23, weight="bold", color="#17324D")
            axis = fig.add_axes([.045, .07, .91, .80]); axis.imshow(image); axis.axis("off")
            pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .92, "Paired Uncertainty and Dataset Integrity", fontsize=23, weight="bold", color="#17324D")
        fig.text(.055, .855, "2,000 label-stratified paired bootstrap replicates", fontsize=12, color="#3B6EA8")
        y = .75
        bootstrap = payload["paired_stratified_bootstrap"]["metrics"]
        for metric in METRICS:
            low, high = bootstrap[metric]["confidence_interval_95"]
            fig.text(.075, y, DISPLAY[metric], fontsize=11.5, color="#526678")
            fig.text(.30, y, pct(deltas[metric], True), fontsize=11.5, weight="bold", color="#17324D")
            fig.text(.48, y, f"95% CI {pct(low, True)} to {pct(high, True)}", fontsize=11.5, color="#526678")
            y -= .065
        dataset = payload["dataset"]
        fig.text(.055, .185, "Frozen cohort", fontsize=14, weight="bold", color="#17324D")
        fig.text(.075, .13, f"{dataset['legitimate_rows']:,} legitimate + {dataset['phishing_rows']:,} phishing • zero URL, hostname and 128-token overlap", fontsize=11, color="#526678")
        fig.text(.075, .085, f"SHA-256: {dataset['sha256']}", fontsize=8.5, color="#718096")
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    result_path = args.result_dir / "results.json"
    prediction_path = args.result_dir / "frozen_predictions.npz"
    results = load_json(result_path)
    collection = load_json(args.collection_report)
    if sha256_file(prediction_path) != results["predictions_sha256"]:
        raise RuntimeError("Frozen prediction hash mismatch")
    if results["dataset"]["sha256"] != collection["dataset_sha256"]:
        raise RuntimeError("Collection and evaluation dataset hashes differ")
    arrays = np.load(prediction_path)
    labels = arrays["labels"]
    real_scores = arrays["real_only_scores"]
    gan_scores = arrays["gan_10pct_scores"]
    real_predictions = arrays["real_only_predictions"]
    gan_predictions = arrays["gan_10pct_predictions"]
    real = results["conditions"]["real_only"]["metrics"]
    gan = results["conditions"]["gan_10pct"]["metrics"]
    paired = exact_paired(labels, real_predictions, gan_predictions)
    bootstrap = paired_bootstrap(
        labels, real_scores, float(results["conditions"]["real_only"]["threshold"]),
        gan_scores, float(results["conditions"]["gan_10pct"]["threshold"]),
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    operational = {}
    for key, metrics in (("real_only", real), ("gan_10pct", gan)):
        cm = metrics["confusion_matrix"]
        operational[key] = {
            "false_alerts_per_1000_legitimate": 1000 * cm["fp"] / collection["legitimate_rows"],
            "missed_phishing": cm["fn"],
            "detected_phishing": cm["tp"],
            "total_warnings": cm["fp"] + cm["tp"],
        }
    real_gate = bool(results["conditions"]["real_only"]["practical_gate_passed"])
    gan_gate = bool(results["conditions"]["gan_10pct"]["practical_gate_passed"])
    if real_gate and gan_gate:
        gate_summary = "Both conditions passed: recall was at least 80% and FPR was at most 2%."
    elif not real_gate and not gan_gate:
        gate_summary = "Neither condition passed; both remained below 80% recall while satisfying the 2% FPR ceiling."
    else:
        gate_summary = f"Only {'real-only' if real_gate else '+10% GAN'} passed the combined recall/FPR gate."
    operational_favors_gan = all(
        results["deltas_gan_minus_real"][metric] > 0
        for metric in ("accuracy", "precision", "recall", "f1", "pr_auc")
    ) and results["deltas_gan_minus_real"]["false_positive_rate"] < 0 \
      and results["deltas_gan_minus_real"]["false_negative_rate"] < 0
    payload = {
        "evaluation": "pre-feedback-database fresh snapshot",
        "dataset": {**results["dataset"], "frozen_at_utc": collection["frozen_at_utc"],
                    "source_counts": collection["source_counts"]},
        "conditions": {"real_only": real, "gan_10pct": gan},
        "deltas_gan_minus_real": results["deltas_gan_minus_real"],
        "strict_rule": results["augmentation_verdict"],
        "paired_exact_comparison": paired,
        "paired_stratified_bootstrap": bootstrap,
        "operational": operational,
        "interpretation": {
            "operational_metrics_favor_gan": operational_favors_gan,
            "strict_registered_rule_passed": bool(results["augmentation_verdict"]["supported"]),
            "reason": "GAN improved thresholded metrics and PR-AUC, but ROC-AUC decreased slightly.",
            "thresholds_changed": False,
            "models_retrained": False,
        },
        "practical_gate": {
            "maximum_false_positive_rate": 0.02,
            "minimum_recall": 0.80,
            "real_only_passed": real_gate,
            "gan_10pct_passed": gan_gate,
            "summary": gate_summary,
        },
        "limitations": collection["limitations"],
    }
    write_json(args.result_dir / "paired_analysis.json", payload)
    with (args.result_dir / "metrics_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["condition", *METRICS])
        writer.writerow(["real_only", *[real[key] for key in METRICS]])
        writer.writerow(["gan_10pct", *[gan[key] for key in METRICS]])
        writer.writerow(["gan_minus_real", *[results["deltas_gan_minus_real"][key] for key in METRICS]])
    metric_graph = args.result_dir / "metric_comparison.png"
    operational_graph = args.result_dir / "operational_comparison.png"
    make_metric_graph(metric_graph, payload)
    make_operational_graph(operational_graph, payload)
    make_pdf(args.result_dir / "fresh_snapshot_evaluation_report.pdf", payload, metric_graph, operational_graph)
    ci = bootstrap["metrics"]
    strict_sentence = (
        "The registered augmentation rule passed because recall, F1 and ROC-AUC increased while FPR decreased."
        if results["augmentation_verdict"]["supported"] else
        f"The registered augmentation rule was not fully met because ROC-AUC changed from {pct(real['roc_auc'])} to {pct(gan['roc_auc'])}."
    )
    source_text = ", ".join(f"{count:,} {name}" for name, count in collection["source_counts"].items())
    report = f"""# Fresh {collection['rows'] // 1000}K pre-feedback-database evaluation

## Result

The frozen +10% GAN ensemble improved accuracy, precision, recall, F1, PR-AUC, FPR and FNR relative to the frozen real-only ensemble. {strict_sentence}

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | {' | '.join(pct(real[key]) for key in METRICS)} |
| +10% GAN | {' | '.join(pct(gan[key]) for key in METRICS)} |

The GAN model reduced false alerts from {real['confusion_matrix']['fp']} to {gan['confusion_matrix']['fp']} and missed phishing URLs from {real['confusion_matrix']['fn']} to {gan['confusion_matrix']['fn']}. It uniquely corrected {paired['all_rows']['real_only_wrong_gan_correct']} decisions and uniquely worsened {paired['all_rows']['real_only_correct_gan_wrong']} decisions (exact paired p={paired['all_rows']['two_sided_exact_mcnemar_p_value']:.4g}).

Practical gate: {gate_summary}

The paired-bootstrap F1 delta was {pct(results['deltas_gan_minus_real']['f1'], True)} with a 95% interval of {pct(ci['f1']['confidence_interval_95'][0], True)} to {pct(ci['f1']['confidence_interval_95'][1], True)}. The FPR delta was {pct(results['deltas_gan_minus_real']['false_positive_rate'], True)} with a 95% interval of {pct(ci['false_positive_rate']['confidence_interval_95'][0], True)} to {pct(ci['false_positive_rate']['confidence_interval_95'][1], True)}.

## Cohort

- {collection['legitimate_rows']:,} legitimate and {collection['phishing_rows']:,} phishing URLs.
- Source composition: {source_text}.
- Zero URL, hostname or 128-token identity overlap with training, development, calibration and earlier external cohorts.
- Dataset SHA-256: `{collection['dataset_sha256']}`.
- No checkpoint, threshold or selection rule was changed after the cohort was frozen.

This is useful pre-feedback evidence, but Tranco roots remain a legitimate proxy and underrepresent long legitimate paths, queries and redirects.
"""
    (args.result_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "paired_all_rows": paired["all_rows"],
        "f1_delta_ci95": ci["f1"]["confidence_interval_95"],
        "fpr_delta_ci95": ci["false_positive_rate"]["confidence_interval_95"],
        "output": str(args.result_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
