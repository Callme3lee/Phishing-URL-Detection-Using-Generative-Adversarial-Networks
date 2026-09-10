from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare feature-equal PSL and hybrid GAN conditions.")
    parser.add_argument("--base", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--augmented-root", type=Path, default=Path("data/experiments/ms_hardgan_augmentation_v2"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/augmentations/ms_hardgan_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1"))
    return parser.parse_args()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_labels = np.load(args.base / "train_labels.npy", mmap_mode="r")
    base_groups = np.load(args.base / "train_groups.npy", mmap_mode="r")
    base_psl = np.load(args.base / "train_psl_lexical.npy", mmap_mode="r")
    scaler = np.load(args.base / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.where(np.asarray(scaler["scale"], dtype=np.float32) == 0, 1.0,
                     np.asarray(scaler["scale"], dtype=np.float32))
    index = {"project_version": 3, "protocol": "feature-equal 35-PSL and hybrid GAN comparison", "conditions": []}

    for percentage in (0, 10, 20, 30, 40):
        name = "real_only" if percentage == 0 else f"gan_{percentage}pct"
        root = args.output_root / name
        processed = root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        source_processed = args.base if percentage == 0 else args.augmented_root / f"pct_{percentage:02d}" / "processed"
        labels = np.load(source_processed / "train_labels.npy", mmap_mode="r")
        generated_rows = len(labels) - len(base_labels)

        link(source_processed / "train_tokens.npy", processed / "train_tokens.npy")
        link(source_processed / "train_labels.npy", processed / "train_labels.npy")
        for split in ("validation", "test"):
            for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
                link(args.base / f"{split}_{suffix}", processed / f"{split}_{suffix}")
            link(args.base / f"{split}_psl_lexical.npy", processed / f"{split}_psl_lexical.npy")
        for suffix in ("tokens.npy", "labels.npy", "groups.npy", "psl_lexical.npy", "manifest.csv"):
            source = args.base / f"domain_validation_{suffix}"
            if source.exists():
                link(source, processed / f"domain_validation_{suffix}")
        for filename in ("vocabulary.json", "psl_lexical_metadata.json", "psl_lexical_scaler.npz"):
            link(args.base / filename, processed / filename)

        psl_path = processed / "train_psl_lexical.npy"
        groups_path = processed / "train_groups.npy"
        if percentage == 0:
            link(args.base / "train_psl_lexical.npy", psl_path)
            link(args.base / "train_groups.npy", groups_path)
        elif not psl_path.exists() or not groups_path.exists():
            matrix = np.lib.format.open_memmap(psl_path, mode="w+", dtype=np.float32,
                                               shape=(len(labels), base_psl.shape[1]))
            matrix[:len(base_psl)] = base_psl
            pool_path = args.pool_root / f"generated_{percentage}pct.csv"
            import csv
            with pool_path.open("r", encoding="utf-8", newline="") as handle:
                urls = [row["url"] for row in csv.DictReader(handle)]
            if len(urls) != generated_rows:
                raise RuntimeError(f"{name}: generated rows differ from frozen pool")
            for start in range(0, len(urls), 10000):
                raw = np.stack([psl_aware_lexical_features(url) for url in urls[start:start + 10000]])
                matrix[len(base_psl) + start:len(base_psl) + start + len(raw)] = (raw - center) / scale
            matrix.flush(); del matrix
            groups = np.lib.format.open_memmap(groups_path, mode="w+", dtype=np.uint8, shape=(len(labels),))
            groups[:len(base_groups)] = base_groups
            groups[len(base_groups):] = 5
            groups.flush(); del groups
        for split in ("train", "validation", "test"):
            link(processed / f"{split}_psl_lexical.npy", processed / f"{split}_features.npy")

        report = {
            "project_version": 3, "condition": name, "real_training_rows": len(base_labels),
            "synthetic_training_rows": generated_rows, "training_rows": len(labels),
            "generated_fraction": generated_rows / len(labels), "feature_count": int(base_psl.shape[1]),
            "feature_contract": "same frozen 35 PSL-aware robust-scaled lexical values for RF, SVM, CNN+PSL, and HYB-MS",
            "synthetic_group": 5 if generated_rows else None,
            "coral_contract": "synthetic group excluded from CORAL pairs; CORAL remains real-source alignment only",
            "validation_test_domain_validation_unchanged": True,
            "train_tokens_sha256": sha256_file(processed / "train_tokens.npy"),
            "train_labels_sha256": sha256_file(processed / "train_labels.npy"),
            "train_psl_sha256": sha256_file(psl_path), "train_groups_sha256": sha256_file(groups_path),
        }
        write_json(root / "experiment_report.json", report)
        index["conditions"].append(report)
    write_json(args.output_root / "index.json", index)
    print(index)


if __name__ == "__main__":
    main()
