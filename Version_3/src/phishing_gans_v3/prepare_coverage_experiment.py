from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import LEXICAL_FEATURE_NAMES, canonicalize_url, encode_url, lexical_features


STRUCTURES = ("root", "shallow", "deep", "query")
REVIEW_SEED = 20260811
REVIEW_PER_STRUCTURE = 20
MANUAL_REJECTED_DOMAINS = {
    "anuytzc.xyz": "opaque random-looking domain; legitimacy cannot be supported by URL-only review",
    "danatoto.com": "gambling domain; excluded conservatively from proxy-labelled benign training",
    "ebun.top": "opaque domain on a high-abuse-risk TLD; legitimacy cannot be supported by URL-only review",
    "exloader.net": "game-cheat loader content may distribute unsafe software",
    "thatpervert.com": "adult-content domain excluded conservatively from proxy-labelled benign training",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Version 3 legitimate-coverage experiment data.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--base-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-features", type=Path, default=Path("data/features"))
    parser.add_argument(
        "--coverage-train", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/audit/train.csv"),
    )
    parser.add_argument(
        "--coverage-verification", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/audit/verification_report.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/experiments/legitimate_coverage_v1"),
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    processed = args.output_root / "processed"
    features = args.output_root / "features"
    report_path = args.output_root / "experiment_report.json"
    if report_path.exists():
        raise FileExistsError(f"Coverage experiment is already frozen at {args.output_root}")
    project = load_json(args.config)
    verification = load_json(args.coverage_verification)
    if not verification.get("passed"):
        raise RuntimeError("Coverage candidate audit did not pass")
    if sha256_file(args.coverage_train) != load_json(
        args.coverage_verification.parent / "profile.json"
    )["split_files"]["train"]["sha256"]:
        raise RuntimeError("Coverage training manifest hash differs from its audit profile")

    candidates = load_csv(args.coverage_train)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row["structure_category"]].append(row)
    rng = random.Random(REVIEW_SEED)
    reviewed = []
    for structure in STRUCTURES:
        for row in rng.sample(grouped[structure], REVIEW_PER_STRUCTURE):
            domain = row["tranco_domain"]
            rejected = domain in MANUAL_REJECTED_DOMAINS
            reviewed.append({
                "structure_category": structure,
                "tranco_domain": domain,
                "tranco_rank": row["tranco_rank"],
                "raw_url": row["raw_url"],
                "review_scope": "URL text and recorded provenance only; page content not independently adjudicated",
                "decision": "exclude_entire_domain" if rejected else "accept_for_proxy_labelled_development",
                "reason": MANUAL_REJECTED_DOMAINS.get(domain, "plausible popular-domain URL; no structural anomaly observed"),
            })
    reviewed_domains = {row["tranco_domain"] for row in reviewed}
    if set(MANUAL_REJECTED_DOMAINS) - reviewed_domains:
        raise RuntimeError("A manually rejected domain is absent from the frozen review sample")
    accepted = [row for row in candidates if row["tranco_domain"] not in MANUAL_REJECTED_DOMAINS]
    excluded = [row for row in candidates if row["tranco_domain"] in MANUAL_REJECTED_DOMAINS]
    if not accepted:
        raise RuntimeError("No accepted coverage rows")

    vocabulary = load_json(args.base_data / "vocabulary.json")
    maximum = int(project["data"]["maximum_encoded_length"])
    accepted_tokens = np.stack([
        encode_url(canonicalize_url(row["raw_url"])[0], vocabulary, maximum) for row in accepted
    ]).astype(np.uint16)
    accepted_features = np.stack([
        lexical_features(canonicalize_url(row["raw_url"])[0]) for row in accepted
    ]).astype(np.float32)
    base_train_tokens = np.load(args.base_data / "train_tokens.npy", mmap_mode="r")
    base_train_labels = np.load(args.base_data / "train_labels.npy", mmap_mode="r")
    base_train_features = np.load(args.base_features / "train_features.npy", mmap_mode="r")
    processed.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)

    train_tokens = np.lib.format.open_memmap(
        processed / "train_tokens.npy", mode="w+", dtype=np.uint16,
        shape=(len(base_train_tokens) + len(accepted), maximum),
    )
    train_tokens[:len(base_train_tokens)] = base_train_tokens
    train_tokens[len(base_train_tokens):] = accepted_tokens
    train_tokens.flush()
    train_labels = np.lib.format.open_memmap(
        processed / "train_labels.npy", mode="w+", dtype=np.uint8,
        shape=(len(base_train_labels) + len(accepted),),
    )
    train_labels[:len(base_train_labels)] = base_train_labels
    train_labels[len(base_train_labels):] = 0
    train_labels.flush()
    train_features = np.lib.format.open_memmap(
        features / "train_features.npy", mode="w+", dtype=np.float32,
        shape=(len(base_train_features) + len(accepted), len(LEXICAL_FEATURE_NAMES)),
    )
    train_features[:len(base_train_features)] = base_train_features
    train_features[len(base_train_features):] = accepted_features
    train_features.flush()

    for split in ("validation", "test"):
        for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
            shutil.copy2(args.base_data / f"{split}_{suffix}", processed / f"{split}_{suffix}")
        shutil.copy2(args.base_features / f"{split}_features.npy", features / f"{split}_features.npy")
    shutil.copy2(args.base_data / "vocabulary.json", processed / "vocabulary.json")
    shutil.copy2(args.base_features / "metadata.json", features / "metadata.json")

    base_manifest = load_csv(args.base_data / "train_manifest.csv")
    manifest_rows: list[dict[str, object]] = [dict(row) for row in base_manifest]
    for offset, row in enumerate(accepted, start=len(base_manifest)):
        normalized, hostname, _ = canonicalize_url(row["raw_url"])
        manifest_rows.append({
            "row_index": offset, "url": normalized, "label": 0,
            "canonical_hostname": hostname,
            "source_row": f"coverage:{row['tranco_list_id']}:{row['tranco_rank']}:{row['common_crawl_digest']}",
        })
    write_csv(processed / "train_manifest.csv", manifest_rows, list(base_manifest[0].keys()))
    write_csv(args.output_root / "manual_review_sample.csv", reviewed, list(reviewed[0].keys()))
    write_csv(args.output_root / "excluded_rows.csv", excluded, list(candidates[0].keys()))

    base_report = load_json(args.base_data / "preparation_report.json")
    split_report = {}
    for split in ("train", "validation", "test"):
        labels = np.load(processed / f"{split}_labels.npy", mmap_mode="r")
        split_report[split] = {
            "total": len(labels),
            "label_0": int((labels == 0).sum()),
            "label_1": int((labels == 1).sum()),
            "manifest_sha256": sha256_file(processed / f"{split}_manifest.csv"),
            "tokens_sha256": sha256_file(processed / f"{split}_tokens.npy"),
            "labels_sha256": sha256_file(processed / f"{split}_labels.npy"),
            "features_sha256": sha256_file(features / f"{split}_features.npy"),
        }
    preparation_report = {
        **base_report,
        "purpose": "legitimate structural-coverage experiment; original validation/test unchanged",
        "splits": split_report,
        "coverage_rows_added": len(accepted),
    }
    write_json(processed / "preparation_report.json", preparation_report)

    report = {
        "project_version": 3,
        "condition": "legitimate_coverage_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_master_sha256": project["data"]["master_sha256"],
        "base_train_rows": len(base_train_labels),
        "accepted_coverage_rows": len(accepted),
        "final_train_rows": len(train_labels),
        "coverage_fraction_of_final_train": len(accepted) / len(train_labels),
        "accepted_structure_counts": dict(Counter(row["structure_category"] for row in accepted)),
        "review": {
            "seed": REVIEW_SEED,
            "rows": len(reviewed),
            "per_structure": REVIEW_PER_STRUCTURE,
            "scope": "URL text and provenance only",
            "accepted_in_sample": sum(row["decision"].startswith("accept") for row in reviewed),
            "excluded_domains": MANUAL_REJECTED_DOMAINS,
            "excluded_rows": len(excluded),
            "sample_path": str(args.output_root / "manual_review_sample.csv"),
            "sample_sha256": sha256_file(args.output_root / "manual_review_sample.csv"),
        },
        "contracts": {
            "coverage_source_split": "training manifest only",
            "coverage_rows_repeated": False,
            "original_validation_unchanged": split_report["validation"]["tokens_sha256"] == sha256_file(args.base_data / "validation_tokens.npy"),
            "original_test_unchanged": split_report["test"]["tokens_sha256"] == sha256_file(args.base_data / "test_tokens.npy"),
            "threshold_policy_unchanged": project["threshold_policy"],
            "hyperparameters_unchanged": True,
        },
        "splits": split_report,
        "limitations": [
            "The new benign rows use proxy labels, not independent page-level safety adjudication.",
            "Only a pre-registered stratified sample received URL-only human review.",
            "Each accepted row is appended once; no weighting or oversampling is used.",
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
