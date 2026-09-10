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
    parser = argparse.ArgumentParser(description="Freeze nested Version 3 GAN augmentation datasets.")
    parser.add_argument("--model", choices=("seqgan", "relgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--generated-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--base-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-features", type=Path, default=Path("data/features"))
    parser.add_argument("--coverage-root", type=Path, default=Path("data/experiments/legitimate_coverage_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/gan_augmentation_v1"))
    return parser.parse_args()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def freeze_condition(root: Path, base_data: Path, base_features: Path,
                     synthetic_tokens: np.ndarray, synthetic_features: np.ndarray,
                     count: int, model: str, percentage: int, source_hash: str) -> dict:
    processed = root / "processed"; features = root / "features"
    if (root / "experiment_report.json").exists():
        raise FileExistsError(f"Frozen condition already exists: {root}")
    base_tokens = np.load(base_data / "train_tokens.npy", mmap_mode="r")
    base_labels = np.load(base_data / "train_labels.npy", mmap_mode="r")
    base_lexical = np.load(base_features / "train_features.npy", mmap_mode="r")
    processed.mkdir(parents=True); features.mkdir(parents=True)
    tokens = np.lib.format.open_memmap(processed / "train_tokens.npy", mode="w+", dtype=np.uint16,
                                       shape=(len(base_tokens) + count, base_tokens.shape[1]))
    tokens[:len(base_tokens)] = base_tokens; tokens[len(base_tokens):] = synthetic_tokens[:count]; tokens.flush()
    labels = np.lib.format.open_memmap(processed / "train_labels.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(base_labels) + count,))
    labels[:len(base_labels)] = base_labels; labels[len(base_labels):] = 1; labels.flush()
    lexical = np.lib.format.open_memmap(features / "train_features.npy", mode="w+", dtype=np.float32,
                                        shape=(len(base_lexical) + count, len(LEXICAL_FEATURE_NAMES)))
    lexical[:len(base_lexical)] = base_lexical; lexical[len(base_lexical):] = synthetic_features[:count]; lexical.flush()
    for split in ("validation", "test"):
        for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
            link(base_data / f"{split}_{suffix}", processed / f"{split}_{suffix}")
        link(base_features / f"{split}_features.npy", features / f"{split}_features.npy")
    link(base_data / "vocabulary.json", processed / "vocabulary.json")
    link(base_features / "metadata.json", features / "metadata.json")
    report = {
        "project_version": 3, "generator": model, "requested_percentage": percentage,
        "base_training_rows": len(base_labels), "base_legitimate_rows": int((base_labels == 0).sum()),
        "base_phishing_rows": int((base_labels == 1).sum()), "generated_phishing_rows": count,
        "final_training_rows": len(labels), "actual_generated_fraction": count / len(labels),
        "accepted_pool_sha256": source_hash, "nested_pool": True,
        "original_validation_and_test_unchanged": True,
    }
    write_json(root / "experiment_report.json", report)
    return report


def main() -> None:
    args = parse_args(); project = load_json(args.config); vocabulary = load_json(args.base_data / "vocabulary.json")
    pool = args.generated_root / args.model / "accepted.csv"
    audit = load_json(args.generated_root / args.model / "audit.json")
    with pool.open("r", encoding="utf-8", newline="") as handle:
        accepted = [row["url"] for row in csv.DictReader(handle)]
    base_rows = len(np.load(args.base_data / "train_labels.npy", mmap_mode="r"))
    counts = {int(round(f * 100)): int(round(base_rows * f / (1 - f)))
              for f in project["augmentation"]["generated_fractions"]}
    maximum = max(counts.values())
    if len(accepted) < maximum or not audit["supports_all_frozen_fractions"]:
        raise RuntimeError(f"{args.model} has {len(accepted)} accepted URLs but {maximum} are required")
    rng = np.random.default_rng(int(project["seed"]))
    selected = np.asarray(accepted, dtype=object)[rng.permutation(len(accepted))[:maximum]].tolist()
    generated_dir = args.generated_root / args.model
    (generated_dir / "frozen_nested_urls.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    tokens = np.stack([encode_url(url, vocabulary, int(project["data"]["maximum_encoded_length"]))
                       for url in selected]).astype(np.uint16)
    features = np.stack([lexical_features(url) for url in selected]).astype(np.float32)
    np.save(generated_dir / "frozen_nested_tokens.npy", tokens)
    np.save(generated_dir / "frozen_nested_features.npy", features)
    source_hash = sha256_file(generated_dir / "frozen_nested_urls.txt")
    reports = []
    bases = {
        "urls_new": (args.base_data, args.base_features),
        "coverage_plus": (args.coverage_root / "processed", args.coverage_root / "features"),
    }
    for base_name, (base_data, base_features) in bases.items():
        for percentage, count in counts.items():
            root = args.output_root / args.model / base_name / f"pct_{percentage:02d}"
            reports.append(freeze_condition(root, base_data, base_features, tokens, features, count,
                                            args.model, percentage, source_hash))
    write_json(args.output_root / args.model / "index.json", {"model": args.model, "conditions": reports})
    print(json.dumps({"model": args.model, "conditions": reports}, indent=2))


if __name__ == "__main__":
    main()
