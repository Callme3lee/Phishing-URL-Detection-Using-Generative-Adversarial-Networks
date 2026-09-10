from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import build_vocab, encode_url, hostname_group, iter_csv, save_json, split_name


LABEL_MAP = {"benign": 0, "malicious": 1}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare URL data as memory-mapped character tensors."
    )
    parser.add_argument("--input", type=Path, default=Path("Data sets/urls_dataset2M.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for a quick pipeline smoke test.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.max_length < 16 or args.max_length > 512:
        raise ValueError("--max-length must be between 16 and 512")
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("Split ratios must be between zero and one")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Train and validation ratios must leave room for testing")


def clean_row(row: dict[str, str]) -> tuple[str, int] | None:
    url = (row.get("url") or "").strip()
    raw_label = (row.get("label") or "").strip().lower()
    if not url or raw_label not in LABEL_MAP:
        return None
    return url, LABEL_MAP[raw_label]


def scan(args: argparse.Namespace) -> tuple[dict[str, Counter], set[str], int]:
    counts = {name: Counter() for name in SPLITS}
    characters: set[str] = set()
    invalid = 0
    seen: set[str] = set()
    for row_number, row in enumerate(iter_csv(args.input), start=1):
        if args.max_rows is not None and row_number > args.max_rows:
            break
        item = clean_row(row)
        if item is None:
            invalid += 1
            continue
        url, label = item
        if url in seen:
            counts["train"]["duplicates_skipped"] += 1
            continue
        seen.add(url)
        group = hostname_group(url)
        split = split_name(group, args.seed, args.train_ratio, args.val_ratio)
        counts[split]["total"] += 1
        counts[split][f"label_{label}"] += 1
        if split == "train":
            characters.update(url[: args.max_length])
    return counts, characters, invalid


def write_arrays(args: argparse.Namespace, counts: dict[str, Counter], vocab: dict[str, int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    offsets = {name: 0 for name in SPLITS}
    for name in SPLITS:
        total = counts[name]["total"]
        arrays[name] = (
            np.lib.format.open_memmap(
                args.output_dir / f"{name}_tokens.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(total, args.max_length),
            ),
            np.lib.format.open_memmap(
                args.output_dir / f"{name}_labels.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(total,),
            ),
        )

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
            hostname_group(url), args.seed, args.train_ratio, args.val_ratio
        )
        position = offsets[split]
        arrays[split][0][position] = encode_url(url, vocab, args.max_length)
        arrays[split][1][position] = label
        offsets[split] += 1

    for tokens, labels in arrays.values():
        tokens.flush()
        labels.flush()


def main() -> None:
    args = parse_args()
    validate_args(args)
    counts, characters, invalid = scan(args)
    vocab = build_vocab(characters)
    write_arrays(args, counts, vocab)
    metadata = {
        "source": str(args.input),
        "label_map": LABEL_MAP,
        "max_length": args.max_length,
        "seed": args.seed,
        "split_method": "blake2b hash of normalized hostname",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "invalid_rows_skipped": invalid,
        "counts": {name: dict(value) for name, value in counts.items()},
        "vocab": vocab,
    }
    save_json(args.output_dir / "metadata.json", metadata)
    print(f"Prepared data in {args.output_dir}")
    for name in SPLITS:
        print(f"{name}: {dict(counts[name])}")
    print(f"vocabulary_size: {len(vocab)}")


if __name__ == "__main__":
    main()
