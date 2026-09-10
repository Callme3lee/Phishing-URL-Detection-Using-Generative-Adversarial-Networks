from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import encode_url, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a leakage-screened external URL test set.")
    parser.add_argument("--external", type=Path, default=Path("Data sets/URL dataset.csv"))
    parser.add_argument("--source", type=Path, default=Path("Data sets/urls_dataset2M.csv"))
    parser.add_argument("--training-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/external_phishing_test"))
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--benign-label", default="legitimate")
    parser.add_argument("--malicious-label", default="phishing")
    return parser.parse_args()


def url_hash(url: str) -> int:
    normalized = url.strip().lower().encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(normalized, digest_size=8).digest(), "big")


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    args = parse_args()
    label_map = {
        args.benign_label.strip().lower(): 0,
        args.malicious_label.strip().lower(): 1,
    }
    if len(label_map) != 2:
        raise ValueError("Benign and malicious source labels must be different.")
    training_metadata = load_json(args.training_data_dir / "metadata.json")
    source_hashes: set[int] = set()
    for row in iter_rows(args.source):
        url = (row.get("url") or "").strip()
        if url:
            source_hashes.add(url_hash(url))

    counts = Counter({"invalid": 0, "duplicate_external": 0, "overlap_removed": 0})
    seen_external: set[int] = set()
    for row in iter_rows(args.external):
        url = (row.get(args.url_column) or "").strip()
        label_text = (row.get(args.label_column) or "").strip().lower()
        if not url or label_text not in label_map:
            counts["invalid"] += 1
            continue
        key = url_hash(url)
        if key in seen_external:
            counts["duplicate_external"] += 1
            continue
        seen_external.add(key)
        if key in source_hashes:
            counts["overlap_removed"] += 1
            continue
        label = label_map[label_text]
        counts["accepted"] += 1
        counts[f"label_{label}"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        args.output_dir / "test_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(counts["accepted"], training_metadata["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        args.output_dir / "test_labels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(counts["accepted"],),
    )
    seen_external.clear()
    position = 0
    for row in iter_rows(args.external):
        url = (row.get(args.url_column) or "").strip()
        label_text = (row.get(args.label_column) or "").strip().lower()
        if not url or label_text not in label_map:
            continue
        key = url_hash(url)
        if key in seen_external:
            continue
        seen_external.add(key)
        if key in source_hashes:
            continue
        tokens[position] = encode_url(
            url, training_metadata["vocab"], training_metadata["max_length"]
        )
        labels[position] = label_map[label_text]
        position += 1
    tokens.flush()
    labels.flush()
    report = {
        "external_source": str(args.external),
        "overlap_source": str(args.source),
        "overlap_method": "case-insensitive normalized URL BLAKE2b-64 hash",
        "source_columns": {
            "url": args.url_column,
            "label": args.label_column,
        },
        "label_map": label_map,
        "counts": dict(counts),
        "encoding_metadata": str(args.training_data_dir / "metadata.json"),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
