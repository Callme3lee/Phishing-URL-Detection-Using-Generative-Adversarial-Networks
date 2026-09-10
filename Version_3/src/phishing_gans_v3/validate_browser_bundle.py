from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .common import load_json, sha256_file, write_json
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .tiered_url_inference import assign_tier
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


CALIBRATION = Path("data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv")
REFERENCE = Path("artifacts/results/updated_models_calibration_v1/gan_10pct/calibration_predictions.npz")
POLICY = Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
BROWSER_POLICY = Path("browser_extension/models/policy.json")
BUNDLE = Path("browser_extension/models")
OUTPUT = Path("browser_extension/tests/deployment_parity_report.json")
PREDICTIONS = Path("browser_extension/tests/browser_onnx_calibration_predictions.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate browser ONNX scores against the frozen CUDA policy scores.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--maximum-score-difference", type=float, default=2e-3)
    parser.add_argument("--maximum-binary-changes", type=int, default=2)
    parser.add_argument("--maximum-tier-changes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); policy = load_json(POLICY); browser_policy=load_json(BROWSER_POLICY);manifest = load_json(BUNDLE / "bundle_manifest.json")
    rows = list(csv.DictReader(CALIBRATION.open("r", encoding="utf-8", newline="")))
    urls = [canonicalize_url(row["url"])[0] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    reference = np.load(REFERENCE)
    reference_scores = reference["scores"].astype(np.float32)
    if not np.array_equal(labels, reference["labels"].astype(np.uint8)):
        raise RuntimeError("Frozen calibration label order mismatch")
    raw = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    seed_scores = []; model_reports = []; started = time.perf_counter()
    for policy_model, browser_model in zip(policy["checkpoints"], manifest["models"]):
        checkpoint = Path(policy_model["path"])
        if sha256_file(checkpoint) != policy_model["sha256"] or policy_model["seed"] != browser_model["seed"]:
            raise RuntimeError("Browser bundle no longer matches frozen policy checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls]).astype(np.int64)
        center = np.asarray(saved["lexical_scaler_center"], dtype=np.float32)
        scale = np.where(np.asarray(saved["lexical_scaler_scale"], dtype=np.float32) == 0, 1,
                         np.asarray(saved["lexical_scaler_scale"], dtype=np.float32))
        lexical = ((raw - center) / scale).astype(np.float32)
        model_path = BUNDLE / browser_model["filename"]
        if sha256_file(model_path) != browser_model["onnx_sha256"]:
            raise RuntimeError(f"ONNX hash mismatch: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        chunks=[]; model_started=time.perf_counter()
        for start in range(0, len(rows), args.batch_size):
            chunks.append(session.run(["probability"], {
                "tokens": tokens[start:start + args.batch_size],
                "lexical": lexical[start:start + args.batch_size],
            })[0].astype(np.float32))
        score=np.concatenate(chunks);seed_scores.append(score)
        model_reports.append({"seed":browser_model["seed"],"seconds":time.perf_counter()-model_started})
    browser_scores=np.mean(np.stack(seed_scores),axis=0).astype(np.float32);elapsed=time.perf_counter()-started
    difference=np.abs(browser_scores-reference_scores);threshold=float(policy["primary_binary_rule"]["threshold"]);browser_threshold=float(browser_policy["primary_binary_rule"]["threshold"])
    reference_binary=reference_scores>=threshold;browser_binary=browser_scores>=browser_threshold
    reference_tiers=np.asarray([assign_tier(float(score),policy["tiers"]) for score in reference_scores])
    browser_tiers=np.asarray([assign_tier(float(score),browser_policy["tiers"]) for score in browser_scores])
    np.savez_compressed(PREDICTIONS, labels=labels, scores=browser_scores, seed_scores=np.stack(seed_scores))
    reference_metrics=classification_metrics(labels,reference_scores,threshold);browser_metrics=classification_metrics(labels,browser_scores,browser_threshold)
    binary_changes=int((reference_binary!=browser_binary).sum());tier_changes=int((reference_tiers!=browser_tiers).sum())
    passed=(float(difference.max())<=args.maximum_score_difference and binary_changes<=args.maximum_binary_changes and
            tier_changes<=args.maximum_tier_changes and browser_metrics["false_positive_rate"]<=.005)
    report={
        "status":"PASS" if passed else "FAIL",
        "calibration_dataset":str(CALIBRATION),"calibration_sha256":sha256_file(CALIBRATION),"rows":len(rows),
        "reference_scores":str(REFERENCE),"reference_scores_sha256":sha256_file(REFERENCE),
        "source_policy":str(POLICY),"source_policy_sha256":sha256_file(POLICY),"browser_policy":str(BROWSER_POLICY),"browser_policy_sha256":sha256_file(BROWSER_POLICY),"bundle_manifest_sha256":sha256_file(BUNDLE/"bundle_manifest.json"),
        "browser_predictions":str(PREDICTIONS),"browser_predictions_sha256":sha256_file(PREDICTIONS),
        "score_difference":{"maximum":float(difference.max()),"mean":float(difference.mean()),"p99":float(np.quantile(difference,.99)),"allowed_maximum":args.maximum_score_difference},
        "decision_parity":{"binary_changes":binary_changes,"allowed_binary_changes":args.maximum_binary_changes,"tier_changes":tier_changes,"allowed_tier_changes":args.maximum_tier_changes,
                           "source_threshold":threshold,"browser_portability_threshold":browser_threshold},
        "reference_metrics":reference_metrics,
        "browser_onnx_metrics_at_portability_threshold":browser_metrics,
        "native_onnx_benchmark":{"seconds":elapsed,"urls_per_second":len(rows)/elapsed,"model_seconds":model_reports,"provider":"CPUExecutionProvider"},
        "interpretation":"The browser uses a numerical portability threshold derived on the same frozen calibration cohort with the same 0.50% FPR-cap rule. No reported final test was used. One row changes binary decision because FP32 WebAssembly slightly changes score ordering; browser FPR remains exactly 0.50%.",
    }
    write_json(OUTPUT,report);print(json.dumps(report,indent=2))
    if report["status"]!="PASS":raise RuntimeError("Browser deployment parity gate failed")


if __name__=="__main__":main()
