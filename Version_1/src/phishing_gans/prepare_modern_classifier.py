from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import encode_url, load_json


def load_urls(path: Path, label: int, source: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [
            {"url": row["url"], "label": label, "source": source}
            for row in csv.DictReader(handle)
            if row.get("url")
        ]


def write_split(output_dir: Path, split: str, rows: list[dict], metadata: dict) -> Counter:
    tokens = np.lib.format.open_memmap(
        output_dir / f"{split}_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), metadata["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        output_dir / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    counts: Counter = Counter()
    with (output_dir / f"{split}_urls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "label", "source"])
        writer.writeheader()
        for index, row in enumerate(rows):
            tokens[index] = encode_url(row["url"], metadata["vocab"], metadata["max_length"])
            labels[index] = row["label"]
            writer.writerow(row)
            counts[f"label_{row['label']}"] += 1
            counts[f"source::{row['source']}"] += 1
    tokens.flush()
    labels.flush()
    counts["total"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced modern classifier adaptation dataset.")
    parser.add_argument("--modern-data-dir", type=Path, required=True)
    parser.add_argument("--base-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.generated_count <= 0:
        raise ValueError("--generated-count must be positive")
    # If G is 10% of the final augmented set, the shared real-only base is 9G.
    real_total = 9 * args.generated_count
    if real_total % 2:
        raise ValueError("The generated count must produce an even 9x real base.")
    per_class_train = real_total // 2
    rng = np.random.default_rng(args.seed)
    metadata = load_json(args.base_data_dir / "metadata.json")

    malicious_train = load_urls(
        args.modern_data_dir / "train_urls.csv", 1, "modern_phishing_real"
    )
    malicious_val = load_urls(
        args.modern_data_dir / "val_urls.csv", 1, "modern_phishing_real"
    )
    malicious_test = load_urls(
        args.modern_data_dir / "test_urls.csv", 1, "modern_phishing_real"
    )
    if len(malicious_train) < per_class_train:
        raise RuntimeError(f"Need {per_class_train} malicious train rows, found {len(malicious_train)}")
    train_indices = rng.choice(len(malicious_train), size=per_class_train, replace=False)
    selected_malicious_train = [malicious_train[int(index)] for index in train_indices]

    benign = load_urls(
        args.modern_data_dir / "modern_benign_hard_negatives.csv",
        0,
        "tranco_unfamiliar_hard_negative",
    )
    rng.shuffle(benign)
    required_benign = per_class_train + len(malicious_val) + len(malicious_test)
    if len(benign) < required_benign:
        raise RuntimeError(f"Need {required_benign} benign rows, found {len(benign)}")
    train_benign = benign[:per_class_train]
    val_start = per_class_train
    val_benign = benign[val_start : val_start + len(malicious_val)]
    test_start = val_start + len(malicious_val)
    test_benign = benign[test_start : test_start + len(malicious_test)]

    splits = {
        "train": selected_malicious_train + train_benign,
        "val": malicious_val + val_benign,
        "test": malicious_test + test_benign,
    }
    for rows in splits.values():
        rng.shuffle(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        split: dict(write_split(args.output_dir, split, rows, metadata))
        for split, rows in splits.items()
    }
    output_metadata = {
        "purpose": "modern real-only base shared by real-only and exact-10%-LSGAN classifier adaptation",
        "vocab": metadata["vocab"],
        "max_length": metadata["max_length"],
        "counts": counts,
        "seed": args.seed,
        "generated_count_for_exact_10_percent": args.generated_count,
        "real_train_count": real_total,
        "augmented_final_count": real_total + args.generated_count,
        "generated_fraction": args.generated_count / (real_total + args.generated_count),
        "modern_data_dir": str(args.modern_data_dir),
        "policy": "balanced real classes; temporally held-out phishing; disjoint unfamiliar benign domains",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(output_metadata, indent=2))


if __name__ == "__main__":
    main()
