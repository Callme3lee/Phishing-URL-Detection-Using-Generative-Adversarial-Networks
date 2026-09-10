from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import hostname_group, load_json, save_json, split_name


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hostname-separated multi-source malicious URL corpus."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument(
        "--external-dir",
        type=Path,
        action="append",
        default=None,
        help="Encoded external directory containing test_tokens.npy and test_labels.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gan_data_multisource"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--copy-chunk-size", type=int, default=50_000)
    return parser.parse_args()


def decode_row(row: np.ndarray, inverse: dict[int, str]) -> str:
    return "".join(inverse.get(int(token), "?") for token in row if token != 0)


def safe_hostname_group(url: str) -> str:
    try:
        return hostname_group(url)
    except ValueError:
        return url.strip().lower()[:80]


def copy_selected(
    destination: np.memmap,
    offset: int,
    source: np.ndarray,
    indices: np.ndarray,
    chunk_size: int,
) -> int:
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        destination[offset : offset + len(selected)] = source[selected]
        offset += len(selected)
    return offset


def main() -> None:
    args = parse_args()
    external_dirs = args.external_dir or [
        Path("artifacts/external_phishing_test"),
        Path("artifacts/external_new_data_urls"),
    ]
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("Split ratios must be between zero and one.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Split ratios must leave a held-out test portion.")

    base_metadata = load_json(args.base_dir / "metadata.json")
    inverse = {
        int(value): key
        for key, value in base_metadata["vocab"].items()
        if len(key) == 1
    }
    base_indices: dict[str, np.ndarray] = {}
    counts = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        labels = np.load(args.base_dir / f"{split}_labels.npy", mmap_mode="r")
        indices = np.flatnonzero(labels == 1)
        base_indices[split] = indices
        counts[split]["base_2m"] = len(indices)
        counts[split]["total"] = len(indices)

    selections: dict[Path, dict[str, np.ndarray]] = {}
    seen_external: set[bytes] = set()
    cross_source_duplicates = 0
    for directory in external_dirs:
        tokens = np.load(directory / "test_tokens.npy", mmap_mode="r")
        labels = np.load(directory / "test_labels.npy", mmap_mode="r")
        selected = {split: [] for split in SPLITS}
        source_name = directory.name
        for index in np.flatnonzero(labels == 1):
            row = tokens[index]
            key = row.tobytes()
            if key in seen_external:
                cross_source_duplicates += 1
                continue
            seen_external.add(key)
            url = decode_row(row, inverse)
            split = split_name(
                safe_hostname_group(url),
                args.seed,
                args.train_ratio,
                args.val_ratio,
            )
            selected[split].append(int(index))
        selections[directory] = {
            split: np.asarray(indices, dtype=np.int64)
            for split, indices in selected.items()
        }
        for split in SPLITS:
            count = len(selections[directory][split])
            counts[split][source_name] = count
            counts[split]["total"] += count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_tokens = {}
    output_labels = {}
    for split in SPLITS:
        total = counts[split]["total"]
        output_tokens[split] = np.lib.format.open_memmap(
            args.output_dir / f"{split}_tokens.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, base_metadata["max_length"]),
        )
        output_labels[split] = np.lib.format.open_memmap(
            args.output_dir / f"{split}_labels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total,),
        )
        output_labels[split][:] = 1

    for split in SPLITS:
        offset = 0
        base_tokens = np.load(
            args.base_dir / f"{split}_tokens.npy", mmap_mode="r"
        )
        offset = copy_selected(
            output_tokens[split],
            offset,
            base_tokens,
            base_indices[split],
            args.copy_chunk_size,
        )
        for directory in external_dirs:
            tokens = np.load(directory / "test_tokens.npy", mmap_mode="r")
            offset = copy_selected(
                output_tokens[split],
                offset,
                tokens,
                selections[directory][split],
                args.copy_chunk_size,
            )
        if offset != counts[split]["total"]:
            raise RuntimeError(f"Expected {counts[split]['total']} {split} rows, wrote {offset}.")
        output_tokens[split].flush()
        output_labels[split].flush()

    metadata = {
        "purpose": "multi-source malicious-only GAN training corpus",
        "base_source": str(args.base_dir),
        "external_sources": [str(path) for path in external_dirs],
        "label_map": {"malicious": 1},
        "max_length": base_metadata["max_length"],
        "vocab": base_metadata["vocab"],
        "seed": args.seed,
        "split_method": "blake2b hash of normalized hostname",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "cross_external_encoded_duplicates_removed": cross_source_duplicates,
        "counts": {split: dict(counts[split]) for split in SPLITS},
    }
    save_json(args.output_dir / "metadata.json", metadata)
    print(f"Prepared {args.output_dir}")
    for split in SPLITS:
        print(f"{split}: {dict(counts[split])}")
    print(f"cross-source duplicates removed: {cross_source_duplicates}")


if __name__ == "__main__":
    main()
