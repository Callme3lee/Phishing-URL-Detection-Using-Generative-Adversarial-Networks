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
from .evaluate_external import neural_scores
from .models import create_classifier
from .url_data import encode_url, lexical_features


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
NEURAL = {"cnn", "lstm", "rnn"}
CONDITIONS = {
    "real_only": Path("artifacts/experiments/multisource_baselines_v1"),
    "gan_10pct": Path("artifacts/experiments/ms_hardgan_augmentation_v2/pct_10"),
    "gan_20pct": Path("artifacts/experiments/ms_hardgan_augmentation_v2/pct_20"),
    "gan_30pct": Path("artifacts/experiments/ms_hardgan_augmentation_v2/pct_30"),
    "gan_40pct": Path("artifacts/experiments/ms_hardgan_augmentation_v2/pct_40"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all MS-HardGAN conditions on the frozen phishing-only challenge holdout.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/ms_hardgan_v2_final/challenge_holdout.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/ms_hardgan_augmentation_v2/synthetic_challenge.json"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def mean_sd(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    maximum = int(project["data"]["maximum_encoded_length"])
    tokens = np.stack([encode_url(url, vocabulary, maximum) for url in urls])
    features = np.stack([lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    per_run: list[dict] = []

    for condition, checkpoint_root in CONDITIONS.items():
        for model_name in MODELS:
            for seed in project["seeds"]:
                run_dir = checkpoint_root / model_name / f"seed_{seed}"
                if model_name in NEURAL:
                    checkpoint_path = run_dir / "checkpoint.pt"
                    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    model = create_classifier(model_name, len(saved["vocabulary"]), saved["model_config"]).to(device)
                    model.load_state_dict(saved["state_dict"])
                    scores = neural_scores(model, tokens, args.batch_size, device)
                    threshold = float(saved["threshold"])
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    checkpoint_path = run_dir / "model.joblib"
                    saved = joblib.load(checkpoint_path)
                    model = saved["model"]
                    scores = (model.predict_proba(features)[:, 1] if model_name == "random_forest"
                              else model.decision_function(features)).astype(np.float32)
                    threshold = float(saved["threshold"])
                predictions = scores >= threshold
                recall = float(predictions.mean())
                record = {
                    "condition": condition, "model": model_name, "seed": int(seed),
                    "threshold": threshold, "recall": recall, "false_negative_rate": 1.0 - recall,
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                }
                per_run.append(record)
                print(json.dumps({key: record[key] for key in ("condition", "model", "seed", "recall", "false_negative_rate")}), flush=True)

    grouped: dict[str, dict] = {}
    for condition in CONDITIONS:
        grouped[condition] = {}
        for model_name in MODELS:
            selected = [row for row in per_run if row["condition"] == condition and row["model"] == model_name]
            grouped[condition][model_name] = {
                metric: mean_sd([float(row[metric]) for row in selected])
                for metric in ("recall", "false_negative_rate")
            }
    report = {
        "project_version": 3,
        "protocol": "frozen phishing-only synthetic challenge; excluded from GAN augmentation selection and classifier training",
        "important_metric_scope": "Because every row is phishing, only recall/FNR are defined. Accuracy, precision, F1, ROC-AUC and FPR are intentionally not reported.",
        "dataset": str(args.dataset), "dataset_sha256": sha256_file(args.dataset), "rows": len(rows),
        "conditions": list(CONDITIONS), "models": list(MODELS), "seeds": list(project["seeds"]),
        "grouped": grouped, "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "evaluations": len(per_run)}, indent=2))


if __name__ == "__main__":
    main()
