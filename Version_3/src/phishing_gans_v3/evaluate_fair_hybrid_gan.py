from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics
from .url_data import encode_url, psl_aware_lexical_features


CONDITIONS = ("real_only", "gan_10pct", "gan_20pct", "gan_30pct", "gan_40pct")
MODELS = ("random_forest", "linear_svm", "cnn_psl", "hyb_ms")
NEURAL = {"cnn_psl", "hyb_ms"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fair PSL/hybrid conditions on external real and synthetic challenge data.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--real-hybrid-root", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--external", type=Path, default=Path("data/external/2026-08-12-large-frozen/external_urls.csv"))
    parser.add_argument("--external-report", type=Path, default=Path("data/external/2026-08-12-large-frozen/collection_report.json"))
    parser.add_argument("--challenge", type=Path, default=Path("data/generated/ms_hardgan_v2_final/challenge_holdout.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/results/fair_hybrid_gan_v1/evaluation"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)), "values": values}


def checkpoint_dir(args: argparse.Namespace, condition: str, model: str) -> Path:
    if condition == "real_only" and model == "hyb_ms":
        return args.real_hybrid_root
    return args.checkpoint_root / condition / model


def main() -> None:
    args = parse_args(); report = load_json(args.external_report)
    if sha256_file(args.external) != report["dataset_sha256"]: raise RuntimeError("External hash mismatch")
    with args.external.open("r", encoding="utf-8", newline="") as handle: external_rows = list(csv.DictReader(handle))
    with args.challenge.open("r", encoding="utf-8", newline="") as handle: challenge_rows = list(csv.DictReader(handle))
    external_labels = np.asarray([int(row["label"]) for row in external_rows], dtype=np.uint8)
    if len(external_rows) != int(report["rows"]): raise RuntimeError("External row count mismatch")
    external_urls = [row["url"] for row in external_rows]; challenge_urls = [row["url"] for row in challenge_rows]
    external_raw = np.stack([psl_aware_lexical_features(url) for url in external_urls]).astype(np.float32)
    challenge_raw = np.stack([psl_aware_lexical_features(url) for url in challenge_urls]).astype(np.float32)
    scaler = np.load(args.data_root / "real_only" / "processed" / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32); scale_values = np.asarray(scaler["scale"], dtype=np.float32)
    scale_values = np.where(scale_values == 0, 1.0, scale_values)
    external_psl = ((external_raw - center) / scale_values).astype(np.float32)
    challenge_psl = ((challenge_raw - center) / scale_values).astype(np.float32)
    vocabulary = load_json(args.data_root / "real_only" / "processed" / "vocabulary.json")
    external_tokens = np.stack([encode_url(url, vocabulary, 128) for url in external_urls])
    challenge_tokens = np.stack([encode_url(url, vocabulary, 128) for url in challenge_urls])
    device = require_cuda(); external_per_run = []; challenge_per_run = []

    for condition in CONDITIONS:
        for model_name in MODELS:
            root = checkpoint_dir(args, condition, model_name)
            for seed in (42, 43, 44):
                if model_name in NEURAL:
                    path = root / f"seed_{seed}" / "checkpoint.pt"
                    saved = torch.load(path, map_location=device, weights_only=False)
                    model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
                    external_scores = score(model, external_tokens, external_psl, args.batch_size, device)
                    challenge_scores = score(model, challenge_tokens, challenge_psl, args.batch_size, device)
                    fair_threshold = float(saved["original_validation_threshold"]["threshold"])
                    native_threshold = float(saved.get("robust_threshold", saved["threshold"])) if model_name == "hyb_ms" else fair_threshold
                    del model; gc.collect(); torch.cuda.empty_cache()
                else:
                    path = root / model_name / f"seed_{seed}" / "model.joblib" if root.name not in MODELS else root / f"seed_{seed}" / "model.joblib"
                    saved = joblib.load(path); model = saved["model"]
                    external_scores = (model.predict_proba(external_psl)[:, 1] if model_name == "random_forest" else model.decision_function(external_psl))
                    challenge_scores = (model.predict_proba(challenge_psl)[:, 1] if model_name == "random_forest" else model.decision_function(challenge_psl))
                    fair_threshold = native_threshold = float(saved["threshold"])
                policies = {"fair_original_validation": fair_threshold, "native": native_threshold}
                for policy, threshold in policies.items():
                    metrics = classification_metrics(external_labels, external_scores, threshold)
                    external_per_run.append({"condition": condition, "model": model_name, "seed": seed, "policy": policy,
                                             "threshold": threshold, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
                                             "metrics": metrics})
                    recall = float((challenge_scores >= threshold).mean())
                    challenge_per_run.append({"condition": condition, "model": model_name, "seed": seed, "policy": policy,
                                              "threshold": threshold, "recall": recall, "false_negative_rate": 1.0 - recall})
                print(json.dumps({"condition": condition, "model": model_name, "seed": seed,
                                  "external_f1_native": external_per_run[-1]["metrics"]["f1"],
                                  "external_fpr_native": external_per_run[-1]["metrics"]["false_positive_rate"],
                                  "challenge_recall_native": challenge_per_run[-1]["recall"]}), flush=True)

    grouped_external = {}; grouped_challenge = {}
    for condition in CONDITIONS:
        grouped_external[condition] = {}; grouped_challenge[condition] = {}
        for model_name in MODELS:
            grouped_external[condition][model_name] = {}; grouped_challenge[condition][model_name] = {}
            for policy in ("fair_original_validation", "native"):
                ext = [row for row in external_per_run if row["condition"] == condition and row["model"] == model_name and row["policy"] == policy]
                cha = [row for row in challenge_per_run if row["condition"] == condition and row["model"] == model_name and row["policy"] == policy]
                grouped_external[condition][model_name][policy] = {metric: mean_sd([float(row["metrics"][metric]) for row in ext]) for metric in METRICS}
                grouped_challenge[condition][model_name][policy] = {metric: mean_sd([float(row[metric]) for row in cha]) for metric in ("recall", "false_negative_rate")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "external_10k.json", {"dataset": str(args.external), "dataset_sha256": report["dataset_sha256"],
               "rows": len(external_rows), "legitimate_rows": int((external_labels == 0).sum()), "phishing_rows": int((external_labels == 1).sum()),
               "grouped": grouped_external, "per_run": external_per_run})
    write_json(args.output_root / "synthetic_challenge.json", {"dataset": str(args.challenge), "dataset_sha256": sha256_file(args.challenge),
               "rows": len(challenge_rows), "metric_scope": "phishing-only: recall/FNR only", "grouped": grouped_challenge,
               "per_run": challenge_per_run})
    print(json.dumps({"external_evaluations": len(external_per_run), "challenge_evaluations": len(challenge_per_run), "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
