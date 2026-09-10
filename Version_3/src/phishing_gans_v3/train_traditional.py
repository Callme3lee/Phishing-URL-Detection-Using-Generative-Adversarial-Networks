from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .common import load_json, set_seed, write_json
from .metrics import classification_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version 3 real-only RF/SVM baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--condition", default="real_only")
    parser.add_argument("--models", default="random_forest,linear_svm")
    parser.add_argument("--seeds", default="42")
    return parser.parse_args()


def train(name: str, seed: int, project: dict, data_dir: Path, feature_dir: Path, output_dir: Path,
          condition: str) -> dict:
    metrics_path = output_dir / name / f"seed_{seed}" / "metrics.json"
    if metrics_path.exists():
        result = load_json(metrics_path)
        print(json.dumps({"stage": "skip_completed", "model": name, "seed": seed}), flush=True)
        return result
    set_seed(seed)
    config = project["baseline_models"][name]
    features = {split: np.load(feature_dir / f"{split}_features.npy", mmap_mode="r") for split in ("train", "validation", "test")}
    labels = {split: np.load(data_dir / f"{split}_labels.npy", mmap_mode="r") for split in ("train", "validation", "test")}
    train_labels = np.asarray(labels["train"])
    total = len(train_labels)
    class_weights = {
        0: total / (2 * max(1, int((train_labels == 0).sum()))),
        1: total / (2 * max(1, int((train_labels == 1).sum()))),
    }
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]), max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]), max_features=config["max_features"],
            class_weight=class_weights, random_state=seed, n_jobs=-1,
        )
        score = lambda values: model.predict_proba(values)[:, 1]
    elif name == "linear_svm":
        model = Pipeline((
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=float(config["C"]), class_weight=class_weights,
                                     max_iter=int(config["max_iter"]), random_state=seed, dual="auto")),
        ))
        score = model.decision_function
    else:
        raise ValueError(f"Unknown traditional model: {name}")
    started = time.perf_counter()
    model.fit(features["train"], labels["train"])
    training_seconds = time.perf_counter() - started
    validation_scores = score(features["validation"])
    threshold = select_threshold(labels["validation"], validation_scores, maximum_fpr=0.01)
    started = time.perf_counter()
    test_scores = score(features["test"])
    inference_seconds = time.perf_counter() - started
    metrics = classification_metrics(labels["test"], test_scores, threshold["threshold"])
    run_dir = output_dir / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "threshold": threshold["threshold"], "project_version": 3,
        "master_sha256": project["data"]["master_sha256"], "condition": condition,
        "scheme_removed": True,
    }, run_dir / "model.joblib", compress=3)
    np.savez_compressed(run_dir / "test_predictions.npz", labels=np.asarray(labels["test"]),
                        scores=test_scores, predictions=(test_scores >= threshold["threshold"]).astype(np.uint8))
    payload = {
        **metrics, "project_version": 3, "condition": condition, "model": name, "seed": seed,
        "threshold": threshold["threshold"], "threshold_selection": threshold,
        "training_time_seconds": training_seconds, "test_inference_seconds": inference_seconds,
        "inference_urls_per_second": len(labels["test"]) / inference_seconds,
        "model_size_bytes": (run_dir / "model.joblib").stat().st_size,
        "device": "CPU", "real_training_rows": len(labels["train"]),
        "master_sha256": project["data"]["master_sha256"], "scheme_removed": True,
    }
    write_json(metrics_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        for name in models:
            result = train(name, seed, project, args.data_dir, args.feature_dir, args.output_dir, args.condition)
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
