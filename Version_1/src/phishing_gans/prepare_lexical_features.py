from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import hostname_group, iter_csv, load_json, save_json, split_name
from .lexical_features import FEATURE_NAMES, extract_lexical_features
from .prepare_data import clean_row


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract lexical features for RF/SVM.")
    parser.add_argument("--input", type=Path, default=Path("Data sets/urls_dataset2M.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/features_2m"))
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_json(args.data_dir / "metadata.json")
    expected = {
        split: int(metadata["counts"][split]["total"]) for split in SPLITS
    }
    if args.max_rows is not None:
        # Count the smoke-test rows using the identical cleaning and split logic.
        expected = {split: 0 for split in SPLITS}
        seen: set[str] = set()
        for row_number, row in enumerate(iter_csv(args.input), start=1):
            if row_number > args.max_rows:
                break
            item = clean_row(row)
            if item is None or item[0] in seen:
                continue
            seen.add(item[0])
            split = split_name(
                hostname_group(item[0]),
                metadata["seed"],
                metadata["train_ratio"],
                metadata["val_ratio"],
            )
            expected[split] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = {
        split: np.lib.format.open_memmap(
            args.output_dir / f"{split}_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(expected[split], len(FEATURE_NAMES)),
        )
        for split in SPLITS
    }
    labels = {
        split: np.lib.format.open_memmap(
            args.output_dir / f"{split}_labels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(expected[split],),
        )
        for split in SPLITS
    }
    offsets = {split: 0 for split in SPLITS}
    seen: set[str] = set()
    for row_number, row in enumerate(iter_csv(args.input), start=1):
        if args.max_rows is not None and row_number > args.max_rows:
            break
        item = clean_row(row)
        if item is None:
            continue
        url, label = item
        if url in seen:
            continue
        seen.add(url)
        split = split_name(
            hostname_group(url),
            metadata["seed"],
            metadata["train_ratio"],
            metadata["val_ratio"],
        )
        position = offsets[split]
        features[split][position] = extract_lexical_features(url)
        labels[split][position] = label
        offsets[split] += 1

    for split in SPLITS:
        features[split].flush()
        labels[split].flush()
        if offsets[split] != expected[split]:
            raise RuntimeError(
                f"{split}: wrote {offsets[split]} rows, expected {expected[split]}"
            )
    save_json(
        args.output_dir / "metadata.json",
        {
            "source": str(args.input),
            "feature_names": list(FEATURE_NAMES),
            "counts": expected,
            "split_metadata": str(args.data_dir / "metadata.json"),
        },
    )
    print(f"Prepared {len(FEATURE_NAMES)} lexical features in {args.output_dir}")
    print(expected)


if __name__ == "__main__":
    main()
