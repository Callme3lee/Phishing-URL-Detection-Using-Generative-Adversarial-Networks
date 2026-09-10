from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a controlled SeqGAN 10% condition for the existing HYB-MS recipe."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/experiments/fair_hybrid_gan_v1/real_only/processed"),
    )
    parser.add_argument("--accepted", type=Path, default=Path("data/generated/seqgan/accepted.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/generated/seqgan/audit.json"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/experiments/generator_hybrid_comparison_v1/seqgan"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-rows", type=int, default=40410)
    parser.add_argument("--experiment", default="generator_hybrid_comparison_v1")
    return parser.parse_args()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    condition = args.output_root / "gan_10pct"
    processed = condition / "processed"
    if condition.exists():
        raise FileExistsError(f"Frozen condition already exists: {condition}")
    processed.mkdir(parents=True)

    base_manifest = read_csv(args.base / "train_manifest.csv")
    base_urls = {canonicalize_url(row["url"])[0] for row in base_manifest}
    base_hosts = {canonicalize_url(row["url"])[1] for row in base_manifest}
    candidates = []
    rejected_overlap = 0
    rejected_invalid = 0
    seen_urls: set[str] = set()
    for row in read_csv(args.accepted):
        try:
            url, host, _ = canonicalize_url(row["url"])
        except ValueError:
            rejected_invalid += 1
            continue
        if url in base_urls or host in base_hosts:
            rejected_overlap += 1
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        candidates.append({**row, "url": url, "hostname": host})
    candidates.sort(
        key=lambda row: hashlib.blake2b(
            f"{args.seed}:{row['url']}".encode("utf-8"), digest_size=16
        ).digest()
    )
    if len(candidates) < args.synthetic_rows:
        raise RuntimeError(f"Need {args.synthetic_rows} clean SeqGAN URLs; found {len(candidates)}")
    selected = candidates[: args.synthetic_rows]

    selected_path = args.output_root / "selected_seqgan_10pct.csv"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    base_tokens = np.load(args.base / "train_tokens.npy", mmap_mode="r")
    base_labels = np.load(args.base / "train_labels.npy", mmap_mode="r")
    base_groups = np.load(args.base / "train_groups.npy", mmap_mode="r")
    base_features = np.load(args.base / "train_psl_lexical.npy", mmap_mode="r")
    vocabulary = load_json(args.base / "vocabulary.json")
    scaler = np.load(args.base / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.where(np.asarray(scaler["scale"], dtype=np.float32) == 0, 1.0,
                     np.asarray(scaler["scale"], dtype=np.float32))
    total = len(base_labels) + len(selected)

    tokens = np.lib.format.open_memmap(
        processed / "train_tokens.npy", mode="w+", dtype=np.uint16,
        shape=(total, base_tokens.shape[1]),
    )
    labels = np.lib.format.open_memmap(
        processed / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    groups = np.lib.format.open_memmap(
        processed / "train_groups.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    features = np.lib.format.open_memmap(
        processed / "train_psl_lexical.npy", mode="w+", dtype=np.float32,
        shape=(total, base_features.shape[1]),
    )
    tokens[: len(base_tokens)] = base_tokens
    labels[: len(base_labels)] = base_labels
    groups[: len(base_groups)] = base_groups
    features[: len(base_features)] = base_features
    for offset, row in enumerate(selected, start=len(base_labels)):
        url = row["url"]
        tokens[offset] = encode_url(url, vocabulary, base_tokens.shape[1])
        labels[offset] = 1
        groups[offset] = 5
        features[offset] = (psl_aware_lexical_features(url) - center) / scale
    tokens.flush(); labels.flush(); groups.flush(); features.flush()

    for split in ("validation", "test", "domain_validation"):
        for suffix in ("tokens.npy", "labels.npy", "manifest.csv", "psl_lexical.npy"):
            source = args.base / f"{split}_{suffix}"
            if source.exists():
                link(source, processed / source.name)
        group_source = args.base / f"{split}_groups.npy"
        if group_source.exists():
            link(group_source, processed / group_source.name)
    for name in ("vocabulary.json", "psl_lexical_metadata.json", "psl_lexical_scaler.npz"):
        link(args.base / name, processed / name)
    for split in ("train", "validation", "test"):
        link(processed / f"{split}_psl_lexical.npy", processed / f"{split}_features.npy")

    fields = list(base_manifest[0])
    manifest = list(base_manifest)
    for index, row in enumerate(selected, start=len(base_manifest)):
        manifest.append({
            "row_index": index,
            "url": row["url"],
            "label": 1,
            "canonical_hostname": row["hostname"],
            "source_group": "synthetic_phishing",
            "source_id": row.get("raw_sha256", ""),
            "source_timestamp": "",
            "raw_url": row["url"],
        })
    with (processed / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest)

    audit = load_json(args.audit)
    report = {
        "project_version": 3,
        "experiment": args.experiment,
        "condition": "gan_10pct",
        "generator": "seqgan",
        "generator_checkpoint": audit["checkpoint"],
        "generator_checkpoint_sha256": audit["checkpoint_sha256"],
        "selection": f"BLAKE2b deterministic ordering with seed {args.seed} after full MS overlap rejection",
        "real_training_rows": len(base_labels),
        "synthetic_training_rows": len(selected),
        "training_rows": total,
        "generated_fraction": len(selected) / total,
        "eligible_clean_unique_rows": len(candidates),
        "rejected_full_ms_url_or_hostname_overlap": rejected_overlap,
        "rejected_invalid": rejected_invalid,
        "same_hyb_ms_recipe_as_ms_hardgan": True,
        "validation_test_domain_validation_unchanged": True,
        "synthetic_group": 5,
        "coral_contract": "synthetic group excluded from CORAL pairs; CORAL remains real-source alignment only",
        "selected_pool": str(selected_path),
        "selected_pool_sha256": sha256_file(selected_path),
        "train_tokens_sha256": sha256_file(processed / "train_tokens.npy"),
        "train_labels_sha256": sha256_file(processed / "train_labels.npy"),
        "train_psl_sha256": sha256_file(processed / "train_psl_lexical.npy"),
        "train_groups_sha256": sha256_file(processed / "train_groups.npy"),
    }
    write_json(condition / "experiment_report.json", report)
    write_json(args.output_root / "index.json", {"experiment": report["experiment"], "conditions": [report]})
    print(report)


if __name__ == "__main__":
    main()
