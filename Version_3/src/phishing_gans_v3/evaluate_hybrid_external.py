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
from .metrics import classification_metrics
from .models import HybridCharacterCNN
from .url_data import encode_url, psl_aware_lexical_features


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen hybrid CNN on an external URL snapshot.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def create_model(checkpoint: dict) -> HybridCharacterCNN:
    config = checkpoint["model_config"]
    return HybridCharacterCNN(
        len(checkpoint["vocabulary"]), int(config["lexical_feature_count"]),
        embedding_dim=int(config["embedding_dim"]), channels_per_kernel=int(config["channels_per_kernel"]),
        kernel_sizes=tuple(config["kernel_sizes"]), lexical_hidden=int(config["lexical_hidden"]),
        lexical_embedding=int(config["lexical_embedding"]), fusion_hidden=int(config["fusion_hidden"]),
        dropout=float(config["dropout"]),
    )


@torch.inference_mode()
def score(model: torch.nn.Module, tokens: np.ndarray, lexical: np.ndarray,
          batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(tokens), batch_size):
        token_batch = torch.from_numpy(tokens[start:start + batch_size].astype(np.int64)).to(device)
        lexical_batch = torch.from_numpy(lexical[start:start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(token_batch, lexical_batch)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def aggregate(records: list[dict], policy: str) -> dict:
    selected = [row[policy] for row in records]
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in selected])),
            "standard_deviation": float(np.std([row[metric] for row in selected], ddof=1)),
            "values": [float(row[metric]) for row in selected],
        } for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    report = load_json(args.collection_report)
    if sha256_file(args.dataset) != report["dataset_sha256"]:
        raise RuntimeError("External snapshot hash differs from its frozen report")
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != int(report["rows"]):
        raise RuntimeError("External snapshot row count differs from its frozen report")

    raw_lexical = np.stack([psl_aware_lexical_features(row["url"]) for row in rows]).astype(np.float32)
    device = require_cuda()
    records, prediction_columns = [], {
        "url": [row["url"] for row in rows], "label": labels.tolist(),
        "source": [row["source"] for row in rows],
    }
    checkpoint_condition = None
    for seed in (42, 43, 44):
        checkpoint_path = args.checkpoint_dir / f"seed_{seed}" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_condition is None:
            checkpoint_condition = checkpoint.get("condition", "unknown_hybrid_condition")
        elif checkpoint.get("condition") != checkpoint_condition:
            raise RuntimeError("Checkpoint directory mixes multiple experimental conditions")
        tokens = np.stack([
            encode_url(row["url"], checkpoint["vocabulary"], args.maximum_length) for row in rows
        ])
        scale = np.asarray(checkpoint["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_lexical - np.asarray(checkpoint["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(checkpoint).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        scores = score(model, tokens, lexical, args.batch_size, device)
        robust_threshold = float(checkpoint["threshold"])
        original_threshold = float(checkpoint["original_validation_threshold"]["threshold"])
        records.append({
            "seed": seed, "checkpoint_sha256": sha256_file(checkpoint_path),
            "robust_threshold": robust_threshold, "original_validation_threshold": original_threshold,
            "robust": classification_metrics(labels, scores, robust_threshold),
            "original_threshold": classification_metrics(labels, scores, original_threshold),
        })
        prediction_columns[f"seed_{seed}_score"] = scores.tolist()
        prediction_columns[f"seed_{seed}_robust_prediction"] = (scores >= robust_threshold).astype(np.uint8).tolist()
        prediction_columns[f"seed_{seed}_original_prediction"] = (scores >= original_threshold).astype(np.uint8).tolist()
        del model
        gc.collect(); torch.cuda.empty_cache()

    summary = {
        "project_version": 3, "condition": checkpoint_condition, "dataset": str(args.dataset),
        "dataset_sha256": report["dataset_sha256"], "rows": len(rows),
        "legitimate_rows": int((labels == 0).sum()), "phishing_rows": int((labels == 1).sum()),
        "evaluation_status": "exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation",
        "threshold_policies": {
            "robust": "max of frozen 1%-FPR thresholds from original and cross-source domain validation",
            "original_threshold": "frozen 1%-FPR threshold from original validation only; diagnostic, not selected on external data",
        },
        "robust": aggregate(records, "robust"),
        "original_threshold": aggregate(records, "original_threshold"),
        "per_seed": records, "collection_report": report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(prediction_columns); writer.writerows(zip(*prediction_columns.values()))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    policies = ("robust", "original_threshold")
    colors = ("#355C7D", "#F67280")
    for axis, metric in zip(axes.ravel(), METRICS):
        means = [100 * summary[policy][metric]["mean"] for policy in policies]
        errors = [100 * summary[policy][metric]["standard_deviation"] for policy in policies]
        axis.bar((0, 1), means, yerr=errors, capsize=5, color=colors)
        axis.set_xticks((0, 1), ("Robust threshold", "Original threshold"), rotation=12, ha="right")
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.grid(axis="y", alpha=.25); axis.set_ylim(0, 100)
    fig.suptitle(f"Hybrid CNN + PSL lexical + CORAL — {len(rows)} external URLs\nmean ± SD across seeds 42/43/44",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.output_dir / "all_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Hybrid CNN External Evaluation", "", summary["evaluation_status"] + ".", "",
             f"Dataset: {len(rows)} rows ({summary['legitimate_rows']} legitimate, {summary['phishing_rows']} phishing).", "",
             "| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, title in (("robust", "Robust"), ("original_threshold", "Original-validation")):
        lines.append("| " + title + " | " + " | ".join(
            f"{summary[policy][metric]['mean']:.4f} ± {summary[policy][metric]['standard_deviation']:.4f}"
            for metric in METRICS) + " |")
    lines.extend(("", "The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.", "",
                  "## Source limitations", "", *[f"- {item}" for item in report.get("limitations", [])]))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "robust": summary["robust"],
                      "original_threshold": summary["original_threshold"]}, indent=2))


if __name__ == "__main__":
    main()
