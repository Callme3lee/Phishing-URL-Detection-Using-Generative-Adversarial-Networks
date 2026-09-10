from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import LEXICAL_FEATURE_NAMES, encode_url, lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize frozen MS-HardGAN-C classifier inputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--pool-dir", type=Path, default=Path("data/augmentations/ms_hardgan_v2"))
    parser.add_argument("--base-data", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--base-features", type=Path, default=Path("data/experiments/multisource_baselines_v1/features"))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/ms_hardgan_augmentation_v2"))
    return parser.parse_args()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args(); project = load_json(args.config); index = load_json(args.pool_dir / "manifest.json")
    if args.output_root.exists():
        raise FileExistsError(f"Frozen experiment root exists: {args.output_root}")
    vocabulary = load_json(args.base_data / "vocabulary.json")
    base_tokens = np.load(args.base_data / "train_tokens.npy", mmap_mode="r")
    base_labels = np.load(args.base_data / "train_labels.npy", mmap_mode="r")
    base_features = np.load(args.base_features / "train_features.npy", mmap_mode="r")
    if len(base_labels) != int(index["base_real_rows"]):
        raise RuntimeError("MS real-only base row count disagrees with augmentation contract")

    reports = []
    for percentage in (10, 20, 30, 40):
        source = args.pool_dir / f"generated_{percentage}pct.csv"
        with source.open("r", encoding="utf-8", newline="") as handle:
            urls = [row["url"] for row in csv.DictReader(handle)]
        expected = int(index["required_rows"][str(percentage)])
        if len(urls) != expected:
            raise RuntimeError(f"{percentage}% pool has {len(urls)} rows, expected {expected}")
        root = args.output_root / f"pct_{percentage:02d}"
        processed, features = root / "processed", root / "features"
        processed.mkdir(parents=True); features.mkdir(parents=True)
        total = len(base_labels) + len(urls)
        tokens = np.lib.format.open_memmap(
            processed / "train_tokens.npy", mode="w+", dtype=np.uint16,
            shape=(total, base_tokens.shape[1]),
        )
        tokens[:len(base_tokens)] = base_tokens
        for start in range(0, len(urls), 10000):
            batch = urls[start:start + 10000]
            tokens[len(base_tokens) + start:len(base_tokens) + start + len(batch)] = np.stack([
                encode_url(url, vocabulary, int(project["data"]["maximum_encoded_length"])) for url in batch
            ])
        tokens.flush(); del tokens
        labels = np.lib.format.open_memmap(
            processed / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,),
        )
        labels[:len(base_labels)] = base_labels; labels[len(base_labels):] = 1; labels.flush(); del labels
        lexical = np.lib.format.open_memmap(
            features / "train_features.npy", mode="w+", dtype=np.float32,
            shape=(total, len(LEXICAL_FEATURE_NAMES)),
        )
        lexical[:len(base_features)] = base_features
        for start in range(0, len(urls), 10000):
            batch = urls[start:start + 10000]
            lexical[len(base_features) + start:len(base_features) + start + len(batch)] = np.stack([
                lexical_features(url) for url in batch
            ])
        lexical.flush(); del lexical
        for split in ("validation", "test"):
            for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
                link(args.base_data / f"{split}_{suffix}", processed / f"{split}_{suffix}")
            link(args.base_features / f"{split}_features.npy", features / f"{split}_features.npy")
        link(args.base_data / "vocabulary.json", processed / "vocabulary.json")
        metadata_source = args.base_features / "metadata.json"
        if metadata_source.exists():
            link(metadata_source, features / "metadata.json")
        else:
            write_json(features / "metadata.json", {
                "feature_names": list(LEXICAL_FEATURE_NAMES),
                "feature_count": len(LEXICAL_FEATURE_NAMES),
                "scheme_removed": True,
            })
        report = {
            "condition": f"ms_hardgan_v2_{percentage}pct", "project_version": 3,
            "generator": "ms_hardgan_v2", "base_real_rows": len(base_labels),
            "base_legitimate_rows": int((base_labels == 0).sum()),
            "base_phishing_rows": int((base_labels == 1).sum()),
            "generated_phishing_rows": len(urls), "final_training_rows": total,
            "generated_fraction": len(urls) / total,
            "source_pool_sha256": sha256_file(source),
            "generator_checkpoint_sha256": index["generator_checkpoint_sha256"],
            "synthetic_challenge_holdout_excluded": True,
            "validation_and_test_unchanged": True,
            "model_architectures_and_hyperparameters_unchanged": True,
        }
        write_json(root / "experiment_report.json", report); reports.append(report)
    write_json(args.output_root / "index.json", {"conditions": reports})
    print(json.dumps({"conditions": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
