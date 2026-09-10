from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .common import encode_url, iter_csv, load_json, save_json
from .lexical_features import FEATURE_NAMES, extract_lexical_features
from .prepare_data import clean_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode controlled augmented CSVs while preserving fixed validation/test sets."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("augmented data sets")
    )
    parser.add_argument(
        "--base-data-dir", type=Path, default=Path("artifacts/data_2m")
    )
    parser.add_argument(
        "--base-feature-dir", type=Path, default=Path("artifacts/features_2m")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/augmented_experiments"),
    )
    return parser.parse_args()


def replace_hard_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.link(source, destination)


def condition_reports(manifest: dict) -> list[dict]:
    reference = dict(manifest["reference"])
    reference["gan_condition"] = "real_only"
    reference["target_generated_percentage"] = 0
    return [reference, *manifest["datasets"]]


def prepare_condition(args: argparse.Namespace, base_metadata: dict, report: dict) -> None:
    source = Path(report["file"])
    name = source.stem
    data_dir = args.output_root / "data" / name
    feature_dir = args.output_root / "features" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    total = int(report["total_rows"])
    max_length = int(base_metadata["max_length"])
    tokens = np.lib.format.open_memmap(
        data_dir / "train_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, max_length),
    )
    labels = np.lib.format.open_memmap(
        data_dir / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    features = np.lib.format.open_memmap(
        feature_dir / "train_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, len(FEATURE_NAMES)),
    )
    feature_labels = np.lib.format.open_memmap(
        feature_dir / "train_labels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total,),
    )

    written = 0
    counts = {0: 0, 1: 0}
    for row in iter_csv(source):
        item = clean_row(row)
        if item is None:
            raise ValueError(f"Invalid row {written + 2} in {source}")
        url, label = item
        if written >= total:
            raise RuntimeError(f"{source} contains more than {total} rows")
        tokens[written] = encode_url(url, base_metadata["vocab"], max_length)
        labels[written] = label
        features[written] = extract_lexical_features(url)
        feature_labels[written] = label
        counts[label] += 1
        written += 1
    if written != total:
        raise RuntimeError(f"{source}: wrote {written} rows, expected {total}")
    tokens.flush()
    labels.flush()
    features.flush()
    feature_labels.flush()

    for split in ("val", "test"):
        replace_hard_link(
            args.base_data_dir / f"{split}_tokens.npy",
            data_dir / f"{split}_tokens.npy",
        )
        replace_hard_link(
            args.base_data_dir / f"{split}_labels.npy",
            data_dir / f"{split}_labels.npy",
        )
        replace_hard_link(
            args.base_feature_dir / f"{split}_features.npy",
            feature_dir / f"{split}_features.npy",
        )
        replace_hard_link(
            args.base_feature_dir / f"{split}_labels.npy",
            feature_dir / f"{split}_labels.npy",
        )

    split_counts = {
        "train": {
            "total": total,
            "label_0": counts[0],
            "label_1": counts[1],
        },
        "val": base_metadata["counts"]["val"],
        "test": base_metadata["counts"]["test"],
    }
    data_metadata = dict(base_metadata)
    data_metadata.update(
        {
            "source": str(source),
            "experiment_condition": name,
            "counts": split_counts,
            "fixed_validation_and_test_from": str(args.base_data_dir),
        }
    )
    save_json(data_dir / "metadata.json", data_metadata)
    save_json(
        feature_dir / "metadata.json",
        {
            "source": str(source),
            "experiment_condition": name,
            "feature_names": list(FEATURE_NAMES),
            "counts": {split: values["total"] for split, values in split_counts.items()},
            "fixed_validation_and_test_from": str(args.base_feature_dir),
        },
    )
    print(f"{name}: encoded {written} training rows ({counts[0]} benign, {counts[1]} malicious)")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.input_dir / "manifest.json")
    base_metadata = load_json(args.base_data_dir / "metadata.json")
    for report in condition_reports(manifest):
        prepare_condition(args, base_metadata, report)
    print(f"Prepared all conditions in {args.output_root}")


if __name__ == "__main__":
    main()
