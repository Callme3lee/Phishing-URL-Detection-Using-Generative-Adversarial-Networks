from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_json, save_json, split_name
from .lexical_features import FEATURE_NAMES, extract_lexical_features
from .prepare_external import iter_rows, url_hash
from .prepare_multisource_gan_data import decode_row, safe_hostname_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the GAN-unseen hostname-held-out portion of an external source."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--encoded-dir", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("Data sets/urls_dataset2M.csv")
    )
    parser.add_argument(
        "--training-data-dir", type=Path, default=Path("artifacts/data_2m")
    )
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--benign-label", required=True)
    parser.add_argument("--malicious-label", required=True)
    parser.add_argument(
        "--prior-malicious-encoded-dir",
        type=Path,
        action="append",
        default=None,
        help="Earlier external sources used to reproduce cross-source encoded deduplication.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_heldout"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = {
        args.benign_label.strip().lower(): 0,
        args.malicious_label.strip().lower(): 1,
    }
    if len(label_map) != 2:
        raise ValueError("Benign and malicious source labels must differ.")
    metadata = load_json(args.training_data_dir / "metadata.json")
    inverse = {
        int(value): key
        for key, value in metadata["vocab"].items()
        if len(key) == 1
    }
    external_tokens = np.load(args.encoded_dir / "test_tokens.npy", mmap_mode="r")
    external_labels = np.load(args.encoded_dir / "test_labels.npy", mmap_mode="r")
    seen_malicious_tokens: set[bytes] = set()
    for directory in args.prior_malicious_encoded_dir or []:
        prior_tokens = np.load(directory / "test_tokens.npy", mmap_mode="r")
        prior_labels = np.load(directory / "test_labels.npy", mmap_mode="r")
        seen_malicious_tokens.update(
            prior_tokens[index].tobytes() for index in np.flatnonzero(prior_labels == 1)
        )

    source_hashes: set[int] = set()
    for row in iter_rows(args.source):
        url = (row.get("url") or "").strip()
        if url:
            source_hashes.add(url_hash(url))

    seen_external: set[int] = set()
    selected_indices = []
    selected_urls = []
    selected_labels = []
    position = 0
    skipped = Counter()
    for row in iter_rows(args.external):
        url = (row.get(args.url_column) or "").strip()
        label_text = (row.get(args.label_column) or "").strip().lower()
        if not url or label_text not in label_map:
            skipped["invalid"] += 1
            continue
        key = url_hash(url)
        if key in seen_external:
            skipped["duplicate_external"] += 1
            continue
        seen_external.add(key)
        if key in source_hashes:
            skipped["overlap_source"] += 1
            continue
        label = label_map[label_text]
        if position >= len(external_tokens):
            raise RuntimeError("Raw accepted rows exceed the encoded external dataset.")
        if int(external_labels[position]) != label:
            raise RuntimeError(f"Label misalignment at encoded position {position}.")
        if label == 1:
            encoded_key = external_tokens[position].tobytes()
            if encoded_key in seen_malicious_tokens:
                skipped["encoded_malicious_duplicate"] += 1
                position += 1
                continue
            seen_malicious_tokens.add(encoded_key)
        decoded = decode_row(external_tokens[position], inverse)
        split = split_name(safe_hostname_group(decoded), args.seed, 0.8, 0.1)
        if split == "test":
            selected_indices.append(position)
            selected_urls.append(url)
            selected_labels.append(label)
        position += 1
    if position != len(external_tokens):
        raise RuntimeError(
            f"Aligned {position} accepted rows, encoded dataset has {len(external_tokens)}."
        )

    indices = np.asarray(selected_indices, dtype=np.int64)
    labels_array = np.asarray(selected_labels, dtype=np.uint8)
    data_dir = args.output_root / "data" / args.name
    feature_dir = args.output_root / "features" / args.name
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "test_tokens.npy", np.asarray(external_tokens[indices]))
    np.save(data_dir / "test_labels.npy", labels_array)
    features = np.lib.format.open_memmap(
        feature_dir / "test_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_urls), len(FEATURE_NAMES)),
    )
    for index, url in enumerate(selected_urls):
        features[index] = extract_lexical_features(url)
    features.flush()
    np.save(feature_dir / "test_labels.npy", labels_array)

    counts = Counter(int(value) for value in labels_array)
    report = {
        "name": args.name,
        "external_source": str(args.external),
        "encoded_source": str(args.encoded_dir),
        "selection": "test bucket from blake2b hash of decoded normalized hostname",
        "seed": args.seed,
        "train_ratio_used_by_gan_corpus": 0.8,
        "validation_ratio_used_by_gan_corpus": 0.1,
        "heldout_ratio": 0.1,
        "label_map": label_map,
        "counts": {
            "total": len(labels_array),
            "label_0": counts[0],
            "label_1": counts[1],
        },
        "raw_alignment": {
            "accepted_encoded_rows": position,
            "skipped": dict(skipped),
        },
    }
    save_json(data_dir / "metadata.json", report)
    save_json(
        feature_dir / "metadata.json",
        {**report, "feature_names": list(FEATURE_NAMES)},
    )
    print(
        f"{args.name}: {len(labels_array)} GAN-unseen held-out URLs "
        f"({counts[0]} benign, {counts[1]} malicious)"
    )


if __name__ == "__main__":
    main()
