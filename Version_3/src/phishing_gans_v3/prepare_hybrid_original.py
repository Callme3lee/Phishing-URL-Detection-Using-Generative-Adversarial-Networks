from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.preprocessing import RobustScaler

from .common import load_json, sha256_file, write_json
from .url_data import PSL_AWARE_LEXICAL_FEATURE_NAMES, psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HYB-ORIG lexical inputs from the unchanged original splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/hybrid_original_v1"))
    return parser.parse_args()


def read_urls(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["url"] for row in csv.DictReader(handle)]


def extract(urls: list[str], path: Path) -> None:
    matrix = open_memmap(path, mode="w+", dtype=np.float32,
                         shape=(len(urls), len(PSL_AWARE_LEXICAL_FEATURE_NAMES)))
    for index, url in enumerate(urls):
        matrix[index] = psl_aware_lexical_features(url)
        if (index + 1) % 50000 == 0:
            print(json.dumps({"stage": "lexical", "file": path.name, "rows": index + 1}), flush=True)
    matrix.flush()


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "preparation_report.json"
    if report_path.exists():
        raise FileExistsError(f"HYB-ORIG preparation is already frozen at {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_urls = {split: read_urls(args.data_dir / f"{split}_manifest.csv")
                  for split in ("train", "validation", "test")}
    for split, urls in split_urls.items():
        extract(urls, args.output_dir / f"{split}_psl_lexical_raw.npy")
    train_raw = np.load(args.output_dir / "train_psl_lexical_raw.npy", mmap_mode="r")
    scaler = RobustScaler(quantile_range=(10, 90)).fit(train_raw)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    center = np.asarray(scaler.center_, dtype=np.float32)
    np.savez(args.output_dir / "psl_lexical_scaler.npz", center=center, scale=scale)
    for split in split_urls:
        raw = np.load(args.output_dir / f"{split}_psl_lexical_raw.npy", mmap_mode="r")
        scaled = open_memmap(args.output_dir / f"{split}_psl_lexical.npy", mode="w+", dtype=np.float32,
                             shape=raw.shape)
        for start in range(0, len(raw), 50000):
            scaled[start:start + 50000] = (raw[start:start + 50000] - center) / scale
        scaled.flush()
    labels = {split: np.load(args.data_dir / f"{split}_labels.npy", mmap_mode="r")
              for split in split_urls}
    report = {
        "project_version": 3, "condition": "hybrid_original_v1", "display_name": "HYB-ORIG",
        "purpose": "controlled hybrid trained on exactly the same original URLs_New rows as all real-only baselines",
        "train_rows": len(labels["train"]), "validation_rows": len(labels["validation"]),
        "test_rows": len(labels["test"]),
        "train_class_counts": {"legitimate": int((labels["train"] == 0).sum()),
                               "phishing": int((labels["train"] == 1).sum())},
        "manifest_hashes": {split: sha256_file(args.data_dir / f"{split}_manifest.csv") for split in split_urls},
        "feature_count": len(PSL_AWARE_LEXICAL_FEATURE_NAMES),
        "feature_names": list(PSL_AWARE_LEXICAL_FEATURE_NAMES),
        "scaler": "RobustScaler quantile_range=(10,90), fit on the 358,619 original training rows only",
        "excluded": ["outside legitimate coverage", "PhishTank training rows", "OpenPhish training rows",
                     "GAN-generated rows", "CORAL", "source-balanced sampling"],
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
