from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .calibrate_deployment_precision import threshold_with_gate
from .common import require_cuda, sha256_file, write_json
from .metrics import classification_metrics, select_threshold
from .train_lexical_verifier import LexicalVerifier
from .url_data import psl_aware_lexical_features


DATASET = Path("data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv")
STAGE1 = Path("artifacts/results/updated_models_calibration_v1/gan_10pct/calibration_predictions.npz")
VERIFIERS = Path("artifacts/experiments/two_stage_url_cascade_v1/lexical_verifier")
SCALER = Path("data/experiments/recent_update_gan10_v1/real_only/processed/psl_lexical_scaler.npz")
OUTPUT = Path("artifacts/experiments/two_stage_url_cascade_v1/calibration")
CAPS = (.0025, .005, .01)


@torch.inference_mode()
def score_verifier(model: torch.nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    for start in range(0, len(values), 2048):
        batch = torch.from_numpy(values[start:start + 2048]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, minimum: float) -> float:
    positive = np.sort(scores[labels == 1])[::-1]
    index = min(len(positive) - 1, max(0, int(np.ceil(minimum * len(positive))) - 1))
    return float(positive[index])


def optimize_cascade(labels: np.ndarray, stage1: np.ndarray, stage2: np.ndarray,
                     screen: float, hard: float, cap: float) -> dict:
    routed = stage1 >= screen; immediate = stage1 >= hard
    candidates = np.unique(stage2[routed & ~immediate])
    best = None
    for verifier_threshold in np.r_[np.inf, candidates[::-1]]:
        prediction = immediate | (routed & (stage2 >= verifier_threshold))
        fp = int(((prediction == 1) & (labels == 0)).sum())
        fpr = fp / int((labels == 0).sum())
        if fpr > cap: continue
        tp = int(((prediction == 1) & (labels == 1)).sum())
        fn = int(((prediction == 0) & (labels == 1)).sum())
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, int(prediction.sum()))
        f1 = 2 * precision * recall / max(1e-15, precision + recall)
        key = (recall, f1, -fpr)
        if best is None or key > best[0]:
            best = (key, float(verifier_threshold), prediction)
    if best is None: raise RuntimeError(f"No cascade threshold satisfies FPR cap {cap}")
    _, threshold, prediction = best
    # A binary cascade has no meaningful ranking AUC, so use the Stage-1 continuous score
    # only for AUC and report that scope explicitly.
    metrics = classification_metrics(labels, prediction.astype(np.float32), .5)
    stage1_auc = classification_metrics(labels, stage1, .5)
    metrics["roc_auc"] = stage1_auc["roc_auc"]
    metrics["pr_auc"] = stage1_auc["pr_auc"]
    return {"maximum_fpr": cap, "verifier_threshold": threshold, "metrics": metrics,
            "routed_rows": int(routed.sum()), "routed_fraction": float(routed.mean()),
            "immediate_block_rows": int(immediate.sum())}


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"Frozen cascade calibration already exists: {OUTPUT}")
    rows = list(csv.DictReader(DATASET.open("r", encoding="utf-8", newline="")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    saved_stage1 = np.load(STAGE1); stage1 = saved_stage1["scores"].astype(np.float32)
    if not np.array_equal(labels, saved_stage1["labels"]): raise RuntimeError("Stage-1 label/order mismatch")
    raw = np.stack([psl_aware_lexical_features(row["url"]) for row in rows]).astype(np.float32)
    scaler = np.load(SCALER); scale = np.where(scaler["scale"] == 0, 1, scaler["scale"])
    features = ((raw - scaler["center"]) / scale).astype(np.float32)
    device = require_cuda(); verifier_seed_scores = []; checkpoints = []
    for seed in (42, 43, 44):
        path = VERIFIERS / f"seed_{seed}/checkpoint.pt"; saved = torch.load(path, map_location=device, weights_only=False)
        model = LexicalVerifier(int(saved["feature_count"])).to(device); model.load_state_dict(saved["state_dict"])
        verifier_seed_scores.append(score_verifier(model, features, device))
        checkpoints.append({"seed": seed, "path": str(path), "sha256": sha256_file(path)})
        del model; gc.collect(); torch.cuda.empty_cache()
    stage2 = np.mean(np.stack(verifier_seed_scores), axis=0).astype(np.float32)
    screen = threshold_for_recall(labels, stage1, .95)
    hard = float(select_threshold(labels, stage1, .001)["threshold"])
    cascades = [optimize_cascade(labels, stage1, stage2, screen, hard, cap) for cap in CAPS]
    singles = []
    for cap in CAPS:
        selected = select_threshold(labels, stage1, cap)
        singles.append({"maximum_fpr": cap, "threshold": selected["threshold"],
                        "metrics": classification_metrics(labels, stage1, selected["threshold"])})
    payload = {
        "project_version": 3, "experiment": "two_stage_url_cascade_v1",
        "selection_status": "FROZEN BEFORE FINAL EVALUATION COHORT CONSTRUCTION",
        "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
        "stage1": "Updated HYB-MS +10% GAN, three-seed mean", "stage2": "real-only 35-feature lexical MLP, three-seed mean",
        "stage2_training_rows": 367184, "stage2_gan_training_rows": 0,
        "screening_rule": {"minimum_calibration_recall": .95, "threshold": screen},
        "immediate_block_rule": {"maximum_calibration_fpr": .001, "threshold": hard},
        "cascade_operating_points": cascades, "single_stage_operating_points": singles,
        "auc_scope": "Cascade ROC-AUC/PR-AUC use the unchanged continuous Stage-1 ranking; cascade decisions themselves are binary.",
        "checkpoints": checkpoints,
        "artifact_hashes": {"stage1_predictions": sha256_file(STAGE1), "scaler": sha256_file(SCALER)},
        "no_test_tuning_contract": "All thresholds and FPR budgets are locked here. The final cohort may only be scored once and may not change these rules.",
    }
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "calibration_predictions.npz", labels=labels, stage1_scores=stage1,
                        stage2_scores=stage2, stage2_seed_scores=np.stack(verifier_seed_scores))
    write_json(OUTPUT / "frozen_cascade.json", payload)
    print(json.dumps({"screen": screen, "hard": hard, "cascade": cascades, "single": singles}, indent=2))


if __name__ == "__main__": main()
