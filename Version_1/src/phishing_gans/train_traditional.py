from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lexical RF or linear SVM baseline.")
    parser.add_argument("--model", choices=("rf", "svm"), required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features_2m"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=150)
    parser.add_argument("--rf-max-depth", type=int, default=24)
    return parser.parse_args()


def load_split(directory: Path, split: str):
    return (
        np.load(directory / f"{split}_features.npy", mmap_mode="r"),
        np.load(directory / f"{split}_labels.npy", mmap_mode="r"),
    )


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "false_positive_rate": fp / max(fp + tn, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def score_model(model, model_type: str, features: np.ndarray) -> np.ndarray:
    if model_type == "rf":
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/classifier_{args.model}")
    x_train, y_train = load_split(args.feature_dir, "train")
    x_val, y_val = load_split(args.feature_dir, "val")
    x_test, y_test = load_split(args.feature_dir, "test")
    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
            verbose=1,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", dual="auto", max_iter=5000, random_state=args.seed),
        )
    started = time.time()
    model.fit(x_train, y_train)
    training_seconds = time.time() - started
    val_scores = score_model(model, args.model, x_val)
    threshold = best_threshold(y_val, val_scores)
    validation = evaluate(y_val, val_scores, threshold)
    test_scores = score_model(model, args.model, x_test)
    test = evaluate(y_test, test_scores, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib", compress=3)
    result = {
        "model": args.model,
        "training_seconds": training_seconds,
        "validation": validation,
        "test": test,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
