from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import canonicalize_url, encode_url, encoded_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a small post-hybrid calibration set.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid"))
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "calibration_urls.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"Calibration freeze already exists at {args.output_dir}")
    project = load_json(args.config)
    maximum = int(project["data"]["maximum_encoded_length"])
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    forbidden_urls: set[str] = set()
    forbidden_hosts: set[str] = set()
    forbidden_hashes: set[bytes] = set()

    database = sqlite3.connect("data/processed/master.sqlite")
    forbidden_urls.update(row[0] for row in database.execute("SELECT normalized_url FROM urls"))
    forbidden_hosts.update(row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls"))
    forbidden_hashes.update(bytes(row[0]) for row in database.execute("SELECT encoded_hash FROM encoded_rows"))
    database.close()

    identity_files = [
        Path("data/experiments/hybrid_cnn_coral_v1/processed/train_manifest.csv"),
        Path("data/experiments/hybrid_cnn_coral_v1/processed/domain_validation_manifest.csv"),
        Path("data/development/benign_coverage_2026-08-11/benign_candidates.csv"),
        *sorted(Path("data/external").glob("*/external_urls.csv")),
        *sorted(Path("../Version_2/data/external").glob("*/external_urls.csv")),
    ]
    for path in identity_files:
        if not path.exists():
            continue
        for row in csv_rows(path):
            raw = row.get("raw_url") or row.get("url") or ""
            try:
                normalized, hostname, _ = canonicalize_url(raw)
            except ValueError:
                continue
            forbidden_urls.add(normalized); forbidden_hosts.add(hostname)
            forbidden_hashes.add(encoded_digest(encode_url(normalized, vocabulary, maximum)))

    phishtank_path = args.raw_dir / "phishtank_online_valid.csv.gz"
    tranco_path = args.raw_dir / "tranco_top1m.csv.zip"
    openphish_path = args.raw_dir / "openphish_feed.txt"
    counters: Counter = Counter()
    selected: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()
    seen_hashes: set[bytes] = set()

    def accept(raw_url: str, label: int, source: str, source_id: str,
               source_timestamp: str = "", rank: int | None = None) -> bool:
        counters[f"{source}::candidates"] += 1
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url, int(project["data"]["maximum_raw_url_length"]))
        except ValueError as error:
            counters[f"{source}::invalid::{error}"] += 1
            return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if scheme not in set(project["data"]["allowed_schemes"]):
            counters[f"{source}::unsupported_scheme"] += 1; return False
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters[f"{source}::prior_project_identity"] += 1; return False
        if normalized in seen_urls or hostname in seen_hosts or digest in seen_hashes:
            counters[f"{source}::calibration_duplicate"] += 1; return False
        seen_urls.add(normalized); seen_hosts.add(hostname); seen_hashes.add(digest)
        selected.append({
            "url": normalized, "label": label, "source": source, "source_id": source_id,
            "source_timestamp": source_timestamp, "source_rank": "" if rank is None else rank,
            "canonical_hostname": hostname, "raw_url": raw_url,
        })
        return True

    with gzip.open(phishtank_path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishing = list(csv.DictReader(handle))
    phishing.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    for row in phishing:
        if sum(int(item["label"]) == 1 for item in selected) >= args.per_class:
            break
        if row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes":
            accept(row.get("url", ""), 1, "PhishTank verified-online calibration",
                   row.get("phish_id", ""), row.get("verification_time", ""))

    with zipfile.ZipFile(tranco_path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            domains = [(int(rank), domain.strip()) for rank, domain in csv.reader(
                line.decode("utf-8", errors="replace") for line in handle
            )]
    random.Random(args.seed).shuffle(domains)
    for rank, domain in domains:
        if sum(int(item["label"]) == 0 for item in selected) >= args.per_class:
            break
        accept(f"https://{domain}/", 0, "Tranco root calibration proxy", str(rank), rank=rank)

    counts = Counter(int(row["label"]) for row in selected)
    if counts != Counter({0: args.per_class, 1: args.per_class}):
        raise RuntimeError(f"Insufficient disjoint calibration rows: {dict(counts)}")
    selected.sort(key=lambda row: (int(row["label"]), int(row["source_rank"] or 0), str(row["url"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    phishing_times = [str(row["source_timestamp"]) for row in selected if int(row["label"]) == 1]
    previous_openphish = Path("data/development/benign_coverage_2026-08-11/raw/openphish_feed.txt")
    previous_tranco = Path("data/development/benign_coverage_2026-08-11/raw/tranco_Q2XX4_top1m.csv.zip")
    report = {
        "project_version": 3, "purpose": "threshold calibration only; never training or final confirmation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(selected),
        "legitimate_rows": counts[0], "phishing_rows": counts[1], "dataset": str(output_csv),
        "dataset_sha256": sha256_file(output_csv), "seed": args.seed,
        "unique_urls": len(seen_urls), "unique_hostnames": len(seen_hosts), "unique_encoded_rows": len(seen_hashes),
        "overlap_contract": "zero URL, hostname, or 128-token encoding overlap with master, hybrid train/domain-validation, development coverage, or prior external sets",
        "measured_overlap": {"urls": 0, "hostnames": 0, "encoded_rows": 0},
        "sources": {
            "phishing": {"name": "PhishTank verified-online", "path": str(phishtank_path),
                          "sha256": sha256_file(phishtank_path), "feed_rows": len(phishing),
                          "selected_verification_range": {"earliest": min(phishing_times), "latest": max(phishing_times)}},
            "legitimate": {"name": "Tranco root-domain proxy", "path": str(tranco_path),
                            "sha256": sha256_file(tranco_path), "list_rows": len(domains)},
            "openphish_audit": {"path": str(openphish_path), "sha256": sha256_file(openphish_path)},
        },
        "freshness_audit": {
            "openphish_changed_from_development": sha256_file(openphish_path) != sha256_file(previous_openphish),
            "tranco_changed_from_development": sha256_file(tranco_path) != sha256_file(previous_tranco),
            "temporal_claim": "post-model calibration freeze, but not a genuinely later independent source snapshot",
        },
        "filter_counters": dict(counters),
        "limitations": [
            "Tranco membership is a legitimate proxy, not page-level adjudication.",
            "Legitimate rows are roots while phishing rows can contain paths and queries.",
            "The OpenPhish and Tranco source files had not changed since development; no later-snapshot claim is made.",
            "Balanced calibration prevalence does not represent browser deployment prevalence.",
        ],
    }
    write_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
