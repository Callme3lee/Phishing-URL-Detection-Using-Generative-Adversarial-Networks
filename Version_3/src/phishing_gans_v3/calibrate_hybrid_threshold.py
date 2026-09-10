from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics, select_threshold
from .url_data import encode_url, psl_aware_lexical_features


EXTERNAL = {
    "topdomains_phishtank": (
        Path("data/external/2026-08-11/external_urls.csv"),
        Path("data/external/2026-08-11/collection_report.json"),
    ),
    "tranco_openphish": (
        Path("data/external/2026-08-11-openphish-tranco/external_urls.csv"),
        Path("data/external/2026-08-11-openphish-tranco/collection_report.json"),
    ),
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a calibrated threshold for the three-seed hybrid ensemble.")
    parser.add_argument("--dataset", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/hybrid_cnn_coral_v1/calibrated_ensemble_v1"))
    parser.add_argument("--maximum-fpr", type=float, default=0.01)
    parser.add_argument("--maximum-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def read_frozen(path: Path, report_path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    report = load_json(report_path)
    if sha256_file(path) != report["dataset_sha256"]:
        raise RuntimeError(f"Frozen dataset hash mismatch: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != int(report["rows"]):
        raise RuntimeError(f"Frozen row-count mismatch: {path}")
    return rows, labels, report


def ensemble_scores(rows: list[dict[str, str]], args: argparse.Namespace,
                    device: torch.device) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    raw_lexical = np.stack([psl_aware_lexical_features(row["url"]) for row in rows]).astype(np.float32)
    outputs: dict[str, np.ndarray] = {}
    checkpoint_records = []
    for seed in (42, 43, 44):
        checkpoint_path = args.checkpoint_dir / f"seed_{seed}" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(row["url"], checkpoint["vocabulary"], args.maximum_length) for row in rows])
        scale = np.asarray(checkpoint["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_lexical - np.asarray(checkpoint["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(checkpoint).to(device); model.load_state_dict(checkpoint["state_dict"])
        outputs[str(seed)] = score(model, tokens, lexical, args.batch_size, device)
        checkpoint_records.append({"seed": seed, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)})
        del model
        gc.collect(); torch.cuda.empty_cache()
    return np.mean(np.stack(list(outputs.values())), axis=0), outputs, checkpoint_records


def main() -> None:
    args = parse_args()
    threshold_path = args.output_dir / "threshold.json"
    if threshold_path.exists():
        raise FileExistsError(f"Calibrated threshold already frozen at {threshold_path}")
    calibration_rows, calibration_labels, calibration_report = read_frozen(args.dataset, args.collection_report)
    if "calibration only" not in calibration_report["purpose"]:
        raise RuntimeError("Dataset is not declared as calibration-only")
    device = require_cuda()
    calibration_scores, seed_scores, checkpoints = ensemble_scores(calibration_rows, args, device)
    selected = select_threshold(calibration_labels, calibration_scores, maximum_fpr=args.maximum_fpr)
    threshold = float(selected["threshold"])
    calibration_metrics = classification_metrics(calibration_labels, calibration_scores, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    external_results = {}
    for name, (dataset_path, report_path) in EXTERNAL.items():
        rows, labels, report = read_frozen(dataset_path, report_path)
        scores, _, external_checkpoints = ensemble_scores(rows, args, device)
        if external_checkpoints != checkpoints:
            raise RuntimeError("Checkpoint identity changed during calibration evaluation")
        metrics = classification_metrics(labels, scores, threshold)
        external_results[name] = {
            "dataset": str(dataset_path), "dataset_sha256": report["dataset_sha256"],
            "rows": len(rows), "metrics": metrics,
            "status": "exploratory because this external set informed hybrid design; not threshold selection",
        }
        with (args.output_dir / f"{name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(("url", "label", "ensemble_score", "prediction"))
            writer.writerows(zip([row["url"] for row in rows], labels, scores, (scores >= threshold).astype(np.uint8)))

    payload = {
        "project_version": 3, "condition": "hybrid_cnn_coral_v1", "deployment_candidate": "three-seed mean-score ensemble",
        "threshold": threshold, "threshold_selection": selected,
        "policy": f"maximum calibration F1 subject to calibration FPR <= {args.maximum_fpr:.2%}",
        "calibration_dataset": str(args.dataset), "calibration_dataset_sha256": calibration_report["dataset_sha256"],
        "calibration_rows": len(calibration_rows), "calibration_metrics_resubstitution": calibration_metrics,
        "calibration_metrics_warning": "optimistic because the same calibration labels selected the threshold; not a test result",
        "checkpoints": checkpoints, "external_exploratory": external_results,
        "confirmation_status": "pending a genuinely later, untouched snapshot after source feeds advance",
    }
    write_json(threshold_path, payload)
    with (args.output_dir / "calibration_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("url", "label", "seed_42_score", "seed_43_score", "seed_44_score", "ensemble_score", "prediction"))
        writer.writerows(zip(
            [row["url"] for row in calibration_rows], calibration_labels,
            seed_scores["42"], seed_scores["43"], seed_scores["44"], calibration_scores,
            (calibration_scores >= threshold).astype(np.uint8),
        ))

    names = ("Calibration (resubstitution)", "Top-Domain / PhishTank", "Tranco / OpenPhish")
    metric_sets = (calibration_metrics, external_results["topdomains_phishtank"]["metrics"],
                   external_results["tranco_openphish"]["metrics"])
    colors = ("#6C8EBF", "#2A9D8F", "#E76F51")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        values = [100 * result[metric] for result in metric_sets]
        axis.bar(np.arange(3), values, color=colors)
        axis.set_xticks(np.arange(3), names, rotation=20, ha="right")
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_ylim(0, 100); axis.grid(axis="y", alpha=.25)
    fig.suptitle(f"Frozen calibrated hybrid ensemble — threshold {threshold:.6f}\nexternal sets are exploratory, not confirmation",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.output_dir / "calibrated_all_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "calibrated_all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Calibrated Hybrid Ensemble", "",
             f"Frozen threshold: `{threshold:.8f}`.", "",
             f"Policy: {payload['policy']}.", "",
             "The calibration row is resubstitution performance and is not an unbiased test result. The two older external sets are exploratory because they informed the hybrid design.", "",
             "| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in zip(names, metric_sets):
        lines.append("| " + name + " | " + " | ".join(f"{result[metric]:.4f}" for metric in METRICS) + " |")
    lines.extend(("", "A genuinely later untouched confirmation snapshot is still required before deployment claims."))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
