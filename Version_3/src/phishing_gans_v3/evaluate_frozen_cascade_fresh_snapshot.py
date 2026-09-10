from __future__ import annotations

import argparse
import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .train_lexical_verifier import LexicalVerifier
from .url_data import encode_url, psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the already-frozen local URL cascade on a later untouched snapshot."
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/external/2026-08-28-fresh-raw-5pct/external_urls.csv"),
    )
    parser.add_argument(
        "--collection-report", type=Path,
        default=Path("data/external/2026-08-28-fresh-raw-5pct/collection_report.json"),
    )
    parser.add_argument(
        "--freeze", type=Path,
        default=Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/frozen_cascade.json"),
    )
    parser.add_argument(
        "--stage1-root", type=Path,
        default=Path("artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms"),
    )
    parser.add_argument(
        "--lexical-scaler", type=Path,
        default=Path("data/experiments/recent_update_gan10_v1/real_only/processed/psl_lexical_scaler.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/frozen_cascade_fresh_raw_2026-08-28_5pct"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


@torch.inference_mode()
def verifier_score(
    model: torch.nn.Module, values: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start:start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Fresh cascade result already exists at {args.output}")

    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen dataset hash mismatch")
    if not collection.get("selection_locked_before_inference"):
        raise RuntimeError("Dataset was not frozen before inference")
    freeze = load_json(args.freeze)
    if freeze.get("selection_status") != "FROZEN BEFORE FINAL EVALUATION COHORT CONSTRUCTION":
        raise RuntimeError("Cascade does not have the required pre-evaluation freeze status")

    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != collection["rows"] or int(labels.sum()) != collection["phishing_rows"]:
        raise RuntimeError("Dataset counts differ from the collection report")

    device = require_cuda()
    raw_features = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)

    stage1_seed_scores = []
    stage1_checkpoints = []
    for seed in (42, 43, 44):
        path = args.stage1_root / f"seed_{seed}" / "checkpoint.pt"
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = (
            (raw_features - np.asarray(saved["lexical_scaler_center"], dtype=np.float32))
            / scale
        ).astype(np.float32)
        model = create_model(saved).to(device)
        model.load_state_dict(saved["state_dict"])
        stage1_seed_scores.append(neural_score(model, tokens, lexical, args.batch_size, device))
        stage1_checkpoints.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256_file(path),
        })
        del model, tokens, lexical
        gc.collect()
        torch.cuda.empty_cache()
    stage1_scores = np.mean(np.stack(stage1_seed_scores), axis=0)

    scaler = np.load(args.lexical_scaler)
    lexical_scale = np.where(scaler["scale"] == 0, 1.0, scaler["scale"])
    verifier_features = (
        (raw_features - scaler["center"]) / lexical_scale
    ).astype(np.float32)
    stage2_seed_scores = []
    verifier_checkpoints = []
    for item in freeze["checkpoints"]:
        path = Path(item["path"])
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            raise RuntimeError(f"Frozen verifier hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        model = LexicalVerifier(int(saved["feature_count"])).to(device)
        model.load_state_dict(saved["state_dict"])
        stage2_seed_scores.append(verifier_score(model, verifier_features, 2048, device))
        verifier_checkpoints.append({**item, "verified_sha256": actual_hash})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    stage2_scores = np.mean(np.stack(stage2_seed_scores), axis=0)

    screening_threshold = float(freeze["screening_rule"]["threshold"])
    immediate_threshold = float(freeze["immediate_block_rule"]["threshold"])
    routed = stage1_scores >= screening_threshold
    immediate = stage1_scores >= immediate_threshold
    cascade_score = np.where(immediate, 2.0, np.where(routed, stage2_scores, -1.0))

    points = []
    metric_keys = (
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "false_positive_rate", "false_negative_rate",
    )
    for cascade_config, single_config in zip(
        freeze["cascade_operating_points"], freeze["single_stage_operating_points"]
    ):
        verifier_threshold = float(cascade_config["verifier_threshold"])
        single_threshold = float(single_config["threshold"])
        single = classification_metrics(labels, stage1_scores, single_threshold)
        cascade = classification_metrics(labels, cascade_score, verifier_threshold)
        points.append({
            "maximum_calibration_fpr": float(cascade_config["maximum_fpr"]),
            "single_stage_threshold": single_threshold,
            "verifier_threshold": verifier_threshold,
            "single_stage": single,
            "cascade": cascade,
            "delta_cascade_minus_single": {
                key: cascade[key] - single[key] for key in metric_keys
            },
        })

    payload = {
        "project_version": 3,
        "experiment": "frozen_cascade_fresh_raw_2026-08-28_5pct",
        "evaluation_status": "ONE-SHOT TEMPORAL TRANSFER; NO POST-TEST TUNING",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": collection["dataset_sha256"],
            "rows": len(rows),
            "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "phishing_prevalence": float(labels.mean()),
            "measured_overlap": collection["measured_overlap"],
        },
        "frozen_cascade": {
            "path": str(args.freeze),
            "sha256": sha256_file(args.freeze),
            "screening_threshold": screening_threshold,
            "immediate_threshold": immediate_threshold,
            "frozen_before_snapshot": True,
        },
        "stage1": "Updated HYB-MS +10% GAN, three-seed mean",
        "stage2": "real-only 35-feature lexical MLP, three-seed mean",
        "stage2_gan_training_rows": 0,
        "stage1_checkpoints": stage1_checkpoints,
        "verifier_checkpoints": verifier_checkpoints,
        "routed_rows": int(routed.sum()),
        "routed_fraction": float(routed.mean()),
        "immediate_rows": int(immediate.sum()),
        "operating_points": points,
        "auc_contract": (
            "Single-stage AUC uses Stage-1 probabilities. Cascade AUC uses the frozen "
            "continuous cascade decision score: immediate rows rank first, routed rows "
            "use verifier probability, and non-routed rows rank last."
        ),
        "limitations": [
            "The snapshot labels were not used to choose any threshold, but this result must not be used to retune the frozen cascade.",
            "Both stages remain URL-only and cannot observe page content, redirect chains, DNS, TLS, reputation, or domain age.",
            "Tranco membership is a legitimate proxy and feed membership supplies the phishing labels.",
        ],
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        labels=labels,
        stage1_scores=stage1_scores,
        stage2_scores=stage2_scores,
        routed=routed,
        immediate=immediate,
    )
    payload["predictions_sha256"] = sha256_file(args.output / "predictions.npz")
    write_json(args.output / "results.json", payload)
    print(json.dumps({
        "routed_rows": payload["routed_rows"],
        "routed_fraction": payload["routed_fraction"],
        "operating_points": points,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
