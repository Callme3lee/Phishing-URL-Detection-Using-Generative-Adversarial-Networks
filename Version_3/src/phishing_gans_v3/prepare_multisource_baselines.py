from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import LEXICAL_FEATURE_NAMES, lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen hybrid multi-source rows for unchanged baseline classifiers."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/experiments/hybrid_cnn_coral_v1/processed"),
    )
    parser.add_argument(
        "--experiment-report", type=Path,
        default=Path("data/experiments/hybrid_cnn_coral_v1/experiment_report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/experiments/multisource_baselines_v1/features"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    source_report = load_json(args.experiment_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        rows = manifest_rows(args.data_dir / f"{split}_manifest.csv")
        labels = np.load(args.data_dir / f"{split}_labels.npy", mmap_mode="r")
        if len(rows) != len(labels):
            raise RuntimeError(f"{split} manifest/label length mismatch")
        target = args.output_dir / f"{split}_features.npy"
        if target.exists() and not args.overwrite:
            matrix = np.load(target, mmap_mode="r")
            if matrix.shape != (len(rows), len(LEXICAL_FEATURE_NAMES)):
                raise RuntimeError(f"Existing feature shape is wrong: {target} {matrix.shape}")
        else:
            matrix = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float32,
                shape=(len(rows), len(LEXICAL_FEATURE_NAMES)),
            )
            for index, row in enumerate(rows):
                matrix[index] = lexical_features(row["url"])
                if index and index % 100000 == 0:
                    print({"stage": "lexical_features", "split": split, "rows": index}, flush=True)
            matrix.flush()
        group_counts = Counter(row.get("source_group", "original_frozen_split") for row in rows)
        counts[split] = {
            "rows": len(rows),
            "legitimate": int((labels == 0).sum()),
            "phishing": int((labels == 1).sum()),
            "source_groups": dict(sorted(group_counts.items())),
            "token_sha256": sha256_file(args.data_dir / f"{split}_tokens.npy"),
            "label_sha256": sha256_file(args.data_dir / f"{split}_labels.npy"),
            "manifest_sha256": sha256_file(args.data_dir / f"{split}_manifest.csv"),
            "feature_sha256": sha256_file(target),
        }
    write_json(args.output_dir.parent / "preparation_report.json", {
        "project_version": 3,
        "condition": "multisource_baselines_v1",
        "purpose": "Train unchanged CNN/LSTM/RNN/RF/SVM baselines on the exact stored hybrid multi-source rows.",
        "source_experiment_report": str(args.experiment_report),
        "source_experiment_report_sha256": sha256_file(args.experiment_report),
        "synthetic_rows": 0,
        "splits": counts,
        "feature_names": LEXICAL_FEATURE_NAMES,
        "feature_count": len(LEXICAL_FEATURE_NAMES),
        "controls": [
            "Classifier architectures and hyperparameters unchanged.",
            "Original frozen validation and test sets unchanged.",
            "Only the real training input changes from original-only to multi-source.",
            "No GAN-generated URLs are included.",
            "Thresholds are selected on the unchanged original validation set with FPR <= 1%.",
        ],
        "upstream_report": source_report,
    })


if __name__ == "__main__":
    main()
