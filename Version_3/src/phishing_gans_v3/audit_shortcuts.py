from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from .common import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure simple remaining Version 3 lexical shortcuts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit/shortcut_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_json(args.feature_dir / "metadata.json")
    verification = load_json(args.data_dir / "verification_report.json")
    x_train = np.load(args.feature_dir / "train_features.npy", mmap_mode="r")
    y_train = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    x_test = np.load(args.feature_dir / "test_features.npy", mmap_mode="r")
    y_test = np.load(args.data_dir / "test_labels.npy", mmap_mode="r")
    results = []
    for index, name in enumerate(metadata["feature_names"]):
        model = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)
        model.fit(np.asarray(x_train[:, index]).reshape(-1, 1), y_train)
        predictions = model.predict(np.asarray(x_test[:, index]).reshape(-1, 1))
        probabilities = model.predict_proba(np.asarray(x_test[:, index]).reshape(-1, 1))[:, 1]
        results.append({
            "feature": name,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "threshold": float(model.tree_.threshold[0]),
        })
    results.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    payload = {
        "project_version": 3,
        "scheme_feature_present": metadata["scheme_feature_present"],
        "hostname_splits_disjoint": all(verification["checks"][key] for key in (
            "train_validation_hosts_disjoint", "train_test_hosts_disjoint", "validation_test_hosts_disjoint"
        )),
        "test_majority_class_accuracy": float(max((y_test == 0).mean(), (y_test == 1).mean())),
        "single_feature_stumps": results,
        "strongest_single_feature": results[0],
        "interpretation": "A stump is diagnostic only. High balanced accuracy indicates a remaining one-feature collection shortcut requiring review.",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
