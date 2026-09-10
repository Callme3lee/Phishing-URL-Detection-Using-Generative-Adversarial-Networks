from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from phishing_gans_v3.common import load_json, sha256_file, write_json
from phishing_gans_v3.url_data import (
    LEXICAL_FEATURE_NAMES,
    canonicalize_url,
    encode_url,
    encoded_digest,
    lexical_features,
    split_for_hostname,
)


STRUCTURES = ("root", "shallow", "deep", "query")
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and profile the frozen Version 3 legitimate coverage candidates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/benign_candidates.csv"),
    )
    parser.add_argument(
        "--collection-report",
        type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/collection_report.json"),
    )
    parser.add_argument(
        "--openphish",
        type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/raw/openphish_feed.txt"),
    )
    parser.add_argument("--master-database", type=Path, default=Path("data/processed/master.sqlite"))
    parser.add_argument("--vocabulary", type=Path, default=Path("data/processed/vocabulary.json"))
    parser.add_argument("--external-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/audit"),
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        return list(csv.DictReader(handle))


def load_external(path: Path) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hosts: set[str] = set()
    for csv_path in path.glob("*/external_urls.csv"):
        for row in load_csv(csv_path):
            raw = row.get("url") or row.get("normalized_url") or row.get("raw_url") or ""
            try:
                normalized, hostname, _ = canonicalize_url(raw)
            except ValueError:
                continue
            urls.add(normalized)
            hosts.add(hostname)
    return urls, hosts


def load_openphish(path: Path) -> tuple[set[str], set[str], int]:
    urls: set[str] = set()
    hosts: set[str] = set()
    invalid = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            normalized, hostname, _ = canonicalize_url(raw)
        except ValueError:
            invalid += 1
            continue
        urls.add(normalized)
        hosts.add(hostname)
    return urls, hosts, invalid


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    report = load_json(args.collection_report)
    vocabulary = load_json(args.vocabulary)
    rows = load_csv(args.dataset)
    if not rows:
        raise RuntimeError("Candidate dataset is empty")

    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            failures.append(name)

    dataset_hash = sha256_file(args.dataset)
    check("dataset_hash_matches_collection_report", dataset_hash == report["dataset_sha256"])
    check("collection_marked_complete", report.get("complete") is True)
    check("row_count_matches_collection_report", len(rows) == int(report["rows"]))
    check("all_labels_are_legitimate_zero", all(row["label"] == "0" for row in rows))
    check(
        "all_labels_are_explicit_proxies",
        all(row["label_semantics"] == "popular-domain legitimate proxy" for row in rows),
    )

    maximum = int(project["data"]["maximum_encoded_length"])
    seed = int(project["seed"])
    normalized_urls: list[str] = []
    hostnames: list[str] = []
    encoded_hashes: list[bytes] = []
    canonical_errors: list[dict[str, str]] = []
    split_domains: dict[str, set[str]] = defaultdict(set)
    structure_counts = Counter()
    feature_rows: list[np.ndarray] = []

    for row in rows:
        try:
            normalized, hostname, _ = canonicalize_url(row["raw_url"])
        except ValueError as error:
            canonical_errors.append({"raw_url": row["raw_url"], "error": str(error)})
            continue
        normalized_urls.append(normalized)
        hostnames.append(hostname)
        encoded_hashes.append(encoded_digest(encode_url(normalized, vocabulary, maximum)))
        structure_counts[row["structure_category"]] += 1
        split_domains[row["tranco_domain"]].add(row["proposed_split"])
        feature_rows.append(lexical_features(normalized))

        expected_split = split_for_hostname(row["tranco_domain"], seed, 0.8, 0.1)
        check(f"canonical_url::{len(normalized_urls)}", normalized == row["url"])
        check(f"canonical_hostname::{len(normalized_urls)}", hostname == row["canonical_hostname"])
        check(f"split_hash::{len(normalized_urls)}", expected_split == row["proposed_split"])

    check("all_rows_canonicalize", not canonical_errors and len(normalized_urls) == len(rows))
    check("normalized_urls_are_unique", len(set(normalized_urls)) == len(rows))
    check("encoded_inputs_are_unique", len(set(encoded_hashes)) == len(rows))
    check("all_structure_categories_known", set(structure_counts).issubset(STRUCTURES))
    check(
        "structure_buckets_are_balanced",
        len(set(structure_counts.get(name, 0) for name in STRUCTURES)) == 1,
    )
    check("tranco_domains_do_not_cross_splits", all(len(value) == 1 for value in split_domains.values()))

    database = sqlite3.connect(args.master_database)
    master_urls = {value for value, in database.execute("SELECT normalized_url FROM urls")}
    master_hosts = {value for value, in database.execute("SELECT DISTINCT hostname FROM urls")}
    master_hashes = {bytes(value) for value, in database.execute("SELECT encoded_hash FROM encoded_rows")}
    database.close()

    external_urls, external_hosts = load_external(args.external_root)
    openphish_urls, openphish_hosts, openphish_invalid = load_openphish(args.openphish)
    candidate_urls = set(normalized_urls)
    candidate_hosts = set(hostnames)
    candidate_hashes = set(encoded_hashes)
    overlaps = {
        "version_3_exact_urls": sorted(candidate_urls & master_urls),
        "version_3_encoded_inputs": len(candidate_hashes & master_hashes),
        "version_3_hostnames": sorted(candidate_hosts & master_hosts),
        "frozen_external_exact_urls": sorted(candidate_urls & external_urls),
        "frozen_external_hostnames": sorted(candidate_hosts & external_hosts),
        "openphish_exact_urls": sorted(candidate_urls & openphish_urls),
        "openphish_hostnames": sorted(candidate_hosts & openphish_hosts),
    }
    for name, values in overlaps.items():
        count = values if isinstance(values, int) else len(values)
        check(f"no_{name}", count == 0)

    split_counts = Counter(row["proposed_split"] for row in rows)
    check("only_expected_splits", set(split_counts).issubset(SPLITS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    split_files: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        output_path = args.output_dir / f"{split}.csv"
        split_rows = [row for row in rows if row["proposed_split"] == split]
        write_csv(output_path, split_rows, fieldnames)
        split_files[split] = {
            "path": str(output_path),
            "rows": len(split_rows),
            "sha256": sha256_file(output_path),
            "unique_tranco_domains": len({row["tranco_domain"] for row in split_rows}),
            "unique_hostnames": len({row["canonical_hostname"] for row in split_rows}),
            "structure_counts": dict(Counter(row["structure_category"] for row in split_rows)),
        }

    feature_matrix = np.stack(feature_rows)
    category_profiles: dict[str, object] = {}
    for category in STRUCTURES:
        indices = [index for index, row in enumerate(rows) if row["structure_category"] == category]
        category_profiles[category] = {
            name: numeric_summary(feature_matrix[indices, feature_index])
            for feature_index, name in enumerate(LEXICAL_FEATURE_NAMES)
        }
    profile = {
        "dataset_sha256": dataset_hash,
        "rows": len(rows),
        "unique_urls": len(candidate_urls),
        "unique_encoded_inputs": len(candidate_hashes),
        "unique_hostnames": len(candidate_hosts),
        "unique_tranco_domains": len(split_domains),
        "structure_counts": {name: structure_counts[name] for name in STRUCTURES},
        "split_counts": {name: split_counts[name] for name in SPLITS},
        "split_files": split_files,
        "lexical_features_by_structure": category_profiles,
    }
    write_json(args.output_dir / "profile.json", profile)

    compact_checks = {key: value for key, value in checks.items() if "::" not in key}
    row_check_failures = [key for key in failures if "::" in key]
    verification = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "checks": compact_checks,
        "row_level_check_count": len(checks) - len(compact_checks),
        "row_level_failure_count": len(row_check_failures),
        "row_level_failure_examples": row_check_failures[:20],
        "canonicalization_errors": canonical_errors[:20],
        "overlaps": overlaps,
        "openphish_screen": {
            "path": str(args.openphish),
            "sha256": sha256_file(args.openphish),
            "valid_unique_urls": len(openphish_urls),
            "valid_unique_hostnames": len(openphish_hosts),
            "invalid_rows": openphish_invalid,
            "reference": "https://www.openphish.com/phishing_feeds.html",
            "label_scope": "current OpenPhish community-feed snapshot only",
        },
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    write_json(args.output_dir / "verification_report.json", verification)
    print(json.dumps({
        "passed": verification["passed"],
        "rows": len(rows),
        "structure_counts": profile["structure_counts"],
        "split_counts": profile["split_counts"],
        "overlap_counts": {
            key: value if isinstance(value, int) else len(value) for key, value in overlaps.items()
        },
        "output_dir": str(args.output_dir),
    }, indent=2))
    if failures:
        raise RuntimeError(f"Benign coverage audit failed {len(failures)} checks; see verification_report.json")


if __name__ == "__main__":
    main()
