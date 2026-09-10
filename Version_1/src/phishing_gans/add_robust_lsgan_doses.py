from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .common import load_json, save_json
from .prepare_robust_augmentation import (
    generated_candidates,
    quality_order,
    token_hash,
    write_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add lower-dose quality-filtered LSGAN conditions to the robust experiment."
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/robust_augmentation")
    )
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument(
        "--lsgan-tokens",
        type=Path,
        default=Path("artifacts/corpus_lsgan_multisource/filtered_tokens.npy"),
    )
    parser.add_argument("--percentages", type=int, nargs="+", default=[10, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.experiment_root / "manifest.json")
    metadata = load_json(args.base_dir / "metadata.json")
    real_dir = args.experiment_root / "data" / "real_only"
    real_hashes = set()
    for split in ("train", "val", "test"):
        rows = np.load(real_dir / f"{split}_tokens.npy", mmap_mode="r")
        real_hashes.update(token_hash(row) for row in rows)
    train_tokens = np.load(real_dir / "train_tokens.npy", mmap_mode="r")
    train_labels = np.load(real_dir / "train_labels.npy", mmap_mode="r")
    malicious = np.asarray(train_tokens[np.flatnonzero(train_labels == 1)])
    lsgan = generated_candidates(args.lsgan_tokens, real_hashes)
    order = quality_order(lsgan, malicious)
    real_count = len(train_labels)
    reports_by_name = {item["name"]: item for item in manifest["conditions"]}
    for percentage in args.percentages:
        if not 0 < percentage < 50:
            raise ValueError(f"Invalid percentage: {percentage}")
        generated_count = math.ceil(real_count * percentage / (100 - percentage))
        name = f"lsgan_quality_{percentage}"
        report = write_condition(
            args.experiment_root,
            name,
            lsgan[order[:generated_count]],
            metadata,
            manifest["sources"],
        )
        report["generated_target_percent"] = percentage
        reports_by_name[name] = report
        print(f"{name}: {generated_count} generated, {real_count + generated_count} total")
    manifest["conditions"] = list(reports_by_name.values())
    manifest["additional_quality_lsgan_percentages"] = args.percentages
    save_json(args.experiment_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
