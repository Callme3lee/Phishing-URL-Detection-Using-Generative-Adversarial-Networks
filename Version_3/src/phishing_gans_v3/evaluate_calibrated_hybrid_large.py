from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .calibrate_hybrid_threshold import ensemble_scores, read_frozen
from .common import load_json, require_cuda, sha256_file, write_json
from .metrics import classification_metrics


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen calibrated hybrid ensemble on a large external set.")
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-12-large-frozen/external_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-12-large-frozen/collection_report.json"))
    parser.add_argument("--threshold-record", type=Path, default=Path("artifacts/results/hybrid_cnn_coral_v1/calibrated_ensemble_v1/threshold.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/hybrid_cnn_coral_v1/external_large_2026-08-12"))
    parser.add_argument("--maximum-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Large external evaluation is already frozen at {args.output_dir}")
    rows, labels, report = read_frozen(args.dataset, args.collection_report)
    threshold_record = load_json(args.threshold_record)
    threshold = float(threshold_record["threshold"])
    device = require_cuda()
    scores, seed_scores, checkpoints = ensemble_scores(rows, args, device)
    metrics = classification_metrics(labels, scores, threshold)
    predictions = (scores >= threshold).astype(np.uint8)

    legitimate_mask = labels == 0
    phishing_mask = labels == 1
    later_mask = np.asarray([int(row.get("verified_after_calibration", "0")) == 1 for row in rows])
    per_source = {}
    for source in sorted({row["source"] for row in rows}):
        mask = np.asarray([row["source"] == source for row in rows])
        source_labels = labels[mask]
        source_predictions = predictions[mask]
        if np.all(source_labels == 1):
            per_source[source] = {
                "rows": int(mask.sum()), "recall": float(source_predictions.mean()),
                "false_negative_rate": float(1.0 - source_predictions.mean()),
                "mean_score": float(scores[mask].mean()),
            }
        else:
            per_source[source] = {
                "rows": int(mask.sum()), "false_positive_rate": float(source_predictions.mean()),
                "specificity": float(1.0 - source_predictions.mean()),
                "mean_score": float(scores[mask].mean()),
            }
    temporal = {
        "rows": int(later_mask.sum()),
        "recall": float(predictions[later_mask].mean()) if later_mask.any() else None,
        "false_negative_rate": float(1.0 - predictions[later_mask].mean()) if later_mask.any() else None,
        "mean_score": float(scores[later_mask].mean()) if later_mask.any() else None,
        "claim": "phishing-only temporal subset; it has no legitimate rows and therefore cannot produce accuracy, precision, F1, ROC-AUC, or FPR",
    }
    payload = {
        "project_version": 3, "condition": "frozen calibrated hybrid three-seed ensemble",
        "dataset": str(args.dataset), "dataset_sha256": report["dataset_sha256"], "rows": len(rows),
        "legitimate_rows": int(legitimate_mask.sum()), "phishing_rows": int(phishing_mask.sum()),
        "threshold": threshold, "threshold_record": str(args.threshold_record),
        "threshold_record_sha256": sha256_file(args.threshold_record),
        "threshold_policy": threshold_record["policy"], "metrics": metrics,
        "per_source": per_source, "post_calibration_phishing_subset": temporal,
        "checkpoints": checkpoints, "device": torch.cuda.get_device_name(device),
        "methodological_status": "identity-disjoint external source-transfer evaluation; model, ensemble, and threshold remained frozen",
        "limitations": report["limitations"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, payload)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("url", "label", "source", "verified_after_calibration", "seed_42_score",
                         "seed_43_score", "seed_44_score", "ensemble_score", "prediction"))
        writer.writerows(zip(
            [row["url"] for row in rows], labels, [row["source"] for row in rows],
            [row.get("verified_after_calibration", "0") for row in rows], seed_scores["42"],
            seed_scores["43"], seed_scores["44"], scores, predictions,
        ))

    values = [100 * float(metrics[metric]) for metric in METRICS]
    colors = ["#264653", "#2A9D8F", "#4CAF50", "#3A86FF", "#6C63FF", "#8E6CFF", "#E76F51", "#F4A261"]
    fig, axis = plt.subplots(figsize=(13, 6.8))
    bars = axis.bar([metric.replace("_", "\n").upper() for metric in METRICS], values, color=colors)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3, fontsize=9)
    axis.set_ylim(0, 105); axis.set_ylabel("Percent"); axis.grid(axis="y", alpha=.25)
    axis.set_title(
        f"Frozen calibrated hybrid ensemble — large external evaluation\n"
        f"{len(rows):,} URLs ({legitimate_mask.sum():,} legitimate + {phishing_mask.sum():,} phishing), threshold {threshold:.6f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "all_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Large External Evaluation — Frozen Calibrated Hybrid Ensemble", "",
        f"Dataset: `{len(rows):,}` URLs (`{legitimate_mask.sum():,}` legitimate + `{phishing_mask.sum():,}` phishing).", "",
        f"Frozen threshold: `{threshold:.8f}`. Neither the model, ensemble, nor threshold was changed after seeing this dataset.", "",
        "| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{metrics[metric]:.4f}" for metric in METRICS) + " |", "", "## Source diagnostics", "",
        "| Source | Rows | Positive recall / legitimate specificity | Error rate | Mean score |", "|---|---:|---:|---:|---:|",
    ]
    for source, result in per_source.items():
        success = result.get("recall", result.get("specificity"))
        error = result.get("false_negative_rate", result.get("false_positive_rate"))
        lines.append(f"| {source} | {result['rows']:,} | {success:.4f} | {error:.4f} | {result['mean_score']:.4f} |")
    lines.extend((
        "", "## Temporal phishing diagnostic", "",
        f"Of the PhishTank rows, `{temporal['rows']:,}` were explicitly verified after the calibration freeze. "
        f"Recall on this phishing-only subset is `{temporal['recall']:.4f}` and FNR is `{temporal['false_negative_rate']:.4f}`. "
        "Because it contains no legitimate rows, it cannot provide FPR or the other two-class metrics.", "",
        "## Interpretation guard", "",
        "This is a large identity-disjoint source-transfer evaluation. It is not wholly temporal because the live feeds retain older URLs. "
        "Tranco membership is a legitimate proxy and the balanced prevalence is not browser prevalence.",
    ))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
