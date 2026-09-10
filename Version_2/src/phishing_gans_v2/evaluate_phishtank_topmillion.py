from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_generated_challenge import DISPLAY, MODELS, NEURAL, neural_scores
from .metrics import classification_metrics
from .summarize_quality_experiment import (
    CONDITIONS,
    CONDITION_DISPLAY,
    checkpoint_path,
    load_model,
    mean_sd,
)
from .url_data import encode_url, lexical_features


METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "false_positive_rate", "false_negative_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all controlled models on frozen PhishTank/Top-Million data.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-11/external_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-11/collection_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/external/2026-08-11_phishtank_topmillion"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def add_fnr(metrics: dict) -> dict:
    return {**metrics, "false_negative_rate": 1.0 - float(metrics["recall"])}


def main() -> None:
    args = parse_args(); project = load_json(args.config); report = load_json(args.collection_report)
    if sha256_file(args.dataset) != report["dataset_sha256"]:
        raise RuntimeError("Frozen external dataset hash mismatch")
    rows = list(csv.DictReader(args.dataset.open("r", encoding="utf-8", newline="")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != 500 or int((labels == 0).sum()) != 250 or int((labels == 1).sum()) != 250:
        raise RuntimeError("Expected frozen balanced 500-row snapshot")
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(project["data"]["maximum_encoded_length"])
    tokens = np.stack([encode_url(row["url"], vocabulary, maximum) for row in rows])
    features = np.stack([lexical_features(row["url"]) for row in rows]).astype(np.float32)
    seeds = tuple(int(value) for value in project["seeds"])
    device = require_cuda(); records = []; prediction_columns = {
        "url": [row["url"] for row in rows],
        "label": labels.tolist(),
        "source": [row["source"] for row in rows],
    }

    for model_name in MODELS:
        for seed in seeds:
            baseline_path = checkpoint_path(CONDITIONS["real_only"], model_name, seed)
            baseline_model, common_threshold = load_model(baseline_path, model_name, device)
            del baseline_model; gc.collect(); torch.cuda.empty_cache()
            for condition, root in CONDITIONS.items():
                path = checkpoint_path(root, model_name, seed)
                model, native_threshold = load_model(path, model_name, device)
                if model_name in NEURAL:
                    scores = neural_scores(model, tokens, args.batch_size, device)
                elif model_name == "random_forest":
                    scores = model.predict_proba(features)[:, 1].astype(np.float32)
                else:
                    scores = model.decision_function(features).astype(np.float32)
                native = add_fnr(classification_metrics(labels, scores, native_threshold))
                common = add_fnr(classification_metrics(labels, scores, common_threshold))
                record = {
                    "condition": condition,
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256_file(path),
                    "native_validation_threshold": native_threshold,
                    "common_real_only_threshold": common_threshold,
                    "native_metrics": native,
                    "common_threshold_metrics": common,
                }
                records.append(record)
                key = f"{condition}__{model_name}__seed_{seed}"
                prediction_columns[f"{key}__score"] = scores.tolist()
                prediction_columns[f"{key}__prediction_native"] = (scores >= native_threshold).astype(np.uint8).tolist()
                print(json.dumps({
                    "condition": condition, "model": model_name, "seed": seed,
                    "accuracy": native["accuracy"], "f1": native["f1"], "roc_auc": native["roc_auc"],
                }), flush=True)
                del model, scores; gc.collect(); torch.cuda.empty_cache()

    grouped: dict[str, dict] = {}
    for model_name in MODELS:
        grouped[model_name] = {}
        for condition in CONDITIONS:
            selected = [row for row in records if row["model"] == model_name and row["condition"] == condition]
            grouped[model_name][condition] = {
                "seeds": list(seeds),
                "native_metrics": {
                    metric: {**mean_sd([float(row["native_metrics"][metric]) for row in selected]),
                             "values": [float(row["native_metrics"][metric]) for row in selected]}
                    for metric in METRICS
                },
                "common_threshold_metrics": {
                    metric: {**mean_sd([float(row["common_threshold_metrics"][metric]) for row in selected]),
                             "values": [float(row["common_threshold_metrics"][metric]) for row in selected]}
                    for metric in METRICS
                },
                "confusion_matrix_mean_native": {
                    key: float(np.mean([row["native_metrics"]["confusion_matrix"][key] for row in selected]))
                    for key in ("tn", "fp", "fn", "tp")
                },
            }

    summary = {
        "project_version": 2,
        "protocol": "prospectively frozen PhishTank verified-online versus Top-Million root-proxy evaluation",
        "dataset": str(args.dataset),
        "dataset_sha256": report["dataset_sha256"],
        "rows": len(rows),
        "legitimate_rows": int((labels == 0).sum()),
        "phishing_rows": int((labels == 1).sum()),
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "models": list(MODELS),
        "primary_threshold_protocol": "each checkpoint's frozen internal-validation threshold; no external calibration",
        "secondary_threshold_protocol": "matching real-only model/seed threshold applied across conditions",
        "collection_report": report,
        "grouped": grouped,
        "per_run": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_run.json", records)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(prediction_columns); writer.writerows(zip(*prediction_columns.values()))
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "condition"]
        fields += [f"{metric}_mean" for metric in METRICS] + [f"{metric}_sd" for metric in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for model_name in MODELS:
            for condition in CONDITIONS:
                metrics = grouped[model_name][condition]["native_metrics"]
                writer.writerow({
                    "model": model_name, "condition": condition,
                    **{f"{metric}_mean": metrics[metric]["mean"] for metric in METRICS},
                    **{f"{metric}_sd": metrics[metric]["sd"] for metric in METRICS},
                })

    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    labels_display = [CONDITION_DISPLAY[c] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    for model_name in MODELS:
        fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.2)); axes = axes.ravel()
        for axis, metric in zip(axes, METRICS):
            means = [100 * grouped[model_name][c]["native_metrics"][metric]["mean"] for c in CONDITIONS]
            errors = [100 * grouped[model_name][c]["native_metrics"][metric]["sd"] for c in CONDITIONS]
            axis.bar(x, means, yerr=errors, capsize=4, color=colors)
            axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
            axis.set_xticks(x, labels_display, rotation=22, ha="right"); axis.grid(axis="y", alpha=.25)
        axes[-1].axis("off")
        fig.suptitle(f"{DISPLAY[model_name]} — PhishTank/Top-Million external snapshot\nmean ± SD across seeds; frozen validation thresholds", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.output_dir / f"{model_name}_all_metrics.png", dpi=220, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{model_name}_all_metrics.pdf", bbox_inches="tight"); plt.close(fig)

    matrix = np.asarray([
        [100 * grouped[model][condition]["native_metrics"][metric]["mean"] for metric in METRICS]
        for model in MODELS for condition in CONDITIONS
    ])
    row_labels = [f"{DISPLAY[model]} — {CONDITION_DISPLAY[condition]}" for model in MODELS for condition in CONDITIONS]
    fig, axis = plt.subplots(figsize=(12.8, 10)); image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(METRICS)), [metric.replace("_", " ").upper() for metric in METRICS], rotation=24, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8)
    axis.set_title("External PhishTank/Top-Million evaluation — all metrics (%)", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Metric (%)"); fig.tight_layout()
    fig.savefig(args.output_dir / "all_models_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_models_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    lines = [
        "# PhishTank / Top-Million External Evaluation", "",
        "This prospectively frozen snapshot contains 250 PhishTank verified-online phishing URLs and 250 Top-Million legitimate root proxies.", "",
        "Every URL, 128-token encoding, and hostname is unique and has zero overlap with the Version 2 master data or the prior external snapshot.", "",
        "The primary results use each checkpoint's frozen internal-validation threshold. No model, threshold, or selection rule was fitted to this snapshot.", "",
    ]
    for model_name in MODELS:
        lines.extend((f"## {DISPLAY[model_name]}", "", "| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|"))
        for condition in CONDITIONS:
            metric_values = grouped[model_name][condition]["native_metrics"]
            lines.append(f"| {CONDITION_DISPLAY[condition]} | " + " | ".join(
                f'{metric_values[metric]["mean"]:.6f}' for metric in METRICS
            ) + " |")
        lines.append("")
    lines.extend((
        "## Limitations", "",
        "- Top-Million membership is a legitimate proxy, not a manual safety audit.",
        "- Root-domain legitimate URLs are easier than arbitrary benign deep links.",
        "- This balanced 500-row source-specific snapshot is not a population-wide prevalence or deployment estimate.",
        "- Three identical SVM runs reflect deterministic convergence, not independent stochastic replications.",
    ))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "evaluations": len(records), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
