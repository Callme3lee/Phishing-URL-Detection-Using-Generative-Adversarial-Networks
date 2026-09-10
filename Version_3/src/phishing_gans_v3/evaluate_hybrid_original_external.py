from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics
from .url_data import encode_url, psl_aware_lexical_features


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HYB-ORIG with frozen per-seed thresholds.")
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-12-large-frozen/external_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-12-large-frozen/collection_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/hybrid_original_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/hybrid_original_v1/external_large_2026-08-12"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, object]:
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)), "values": values}


def main() -> None:
    args = parse_args(); report = load_json(args.collection_report)
    if sha256_file(args.dataset) != report["dataset_sha256"]: raise RuntimeError("Frozen dataset hash mismatch")
    if any(report.get("measured_overlap", {}).values()): raise RuntimeError("External overlap is nonzero")
    with args.dataset.open("r", encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    raw_lexical = np.stack([psl_aware_lexical_features(row["url"]) for row in rows]).astype(np.float32)
    device = require_cuda(); records = []; columns = {"url": [row["url"] for row in rows], "label": labels.tolist()}
    for seed in (42, 43, 44):
        path = args.checkpoint_dir / f"seed_{seed}" / "checkpoint.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(row["url"], checkpoint["vocabulary"], 128) for row in rows])
        scale = np.asarray(checkpoint["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_lexical - np.asarray(checkpoint["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(checkpoint).to(device); model.load_state_dict(checkpoint["state_dict"])
        scores = score(model, tokens, lexical, args.batch_size, device); threshold = float(checkpoint["threshold"])
        metrics = classification_metrics(labels, scores, threshold)
        records.append({"seed": seed, "threshold": threshold, "checkpoint": str(path),
                        "checkpoint_sha256": sha256_file(path), "metrics": metrics})
        columns[f"seed_{seed}_score"] = scores.tolist()
        columns[f"seed_{seed}_prediction"] = (scores >= threshold).astype(np.uint8).tolist()
        del model; gc.collect(); torch.cuda.empty_cache()
    summary = {"project_version": 3, "condition": "hybrid_original_v1", "display_name": "HYB-ORIG",
               "dataset": str(args.dataset), "dataset_sha256": report["dataset_sha256"], "rows": len(rows),
               "legitimate_rows": int((labels == 0).sum()), "phishing_rows": int((labels == 1).sum()),
               "threshold_policy": "unchanged original-validation threshold per seed; no external calibration",
               "metrics": {metric: mean_sd([float(row["metrics"][metric]) for row in records]) for metric in METRICS},
               "per_run": records, "device": torch.cuda.get_device_name(device)}
    args.output_dir.mkdir(parents=True, exist_ok=True); write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(columns); writer.writerows(zip(*columns.values()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
