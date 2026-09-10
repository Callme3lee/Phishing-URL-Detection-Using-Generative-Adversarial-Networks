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
    parser = argparse.ArgumentParser(description="Freeze a large identity-disjoint external URL evaluation set.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/master.sqlite"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/2026-08-12-large-frozen/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/2026-08-12-large-frozen"))
    parser.add_argument("--per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--calibration-report", type=Path,
        default=Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"),
    )
    return parser.parse_args()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "external_urls.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"Large external snapshot is already frozen at {args.output_dir}")

    project = load_json(args.config)
    data_config = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum_length = int(data_config["maximum_encoded_length"])
    calibration_report = load_json(args.calibration_report)
    calibration_frozen_at = datetime.fromisoformat(calibration_report["frozen_at_utc"])

    forbidden_urls: set[str] = set()
    forbidden_hosts: set[str] = set()
    forbidden_hashes: set[bytes] = set()
    database = sqlite3.connect(args.database)
    forbidden_urls.update(row[0] for row in database.execute("SELECT normalized_url FROM urls"))
    forbidden_hosts.update(row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls"))
    forbidden_hashes.update(bytes(row[0]) for row in database.execute("SELECT encoded_hash FROM encoded_rows"))
    database.close()

    identity_files = [
        Path("data/experiments/hybrid_cnn_coral_v1/processed/train_manifest.csv"),
        Path("data/experiments/hybrid_cnn_coral_v1/processed/domain_validation_manifest.csv"),
        Path("data/development/benign_coverage_2026-08-11/benign_candidates.csv"),
        Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv"),
        *sorted(Path("data/external").glob("*/external_urls.csv")),
        *sorted(Path("../Version_2/data/external").glob("*/external_urls.csv")),
    ]
    for path in identity_files:
        if not path.exists() or path.resolve() == output_csv.resolve():
            continue
        for row in csv_rows(path):
            try:
                normalized, hostname, _ = canonicalize_url(row.get("raw_url") or row.get("url") or "")
            except ValueError:
                continue
            forbidden_urls.add(normalized)
            forbidden_hosts.add(hostname)
            forbidden_hashes.add(encoded_digest(encode_url(normalized, vocabulary, maximum_length)))

    counters: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()
    seen_hashes: set[bytes] = set()

    def accept(raw_url: str, label: int, source: str, source_id: str,
               source_timestamp: str = "", rank: int | None = None) -> bool:
        counters[f"{source}::candidates"] += 1
        try:
            normalized, hostname, scheme = canonicalize_url(
                raw_url, int(data_config["maximum_raw_url_length"])
            )
        except ValueError as error:
            counters[f"{source}::invalid::{error}"] += 1
            return False
        if scheme not in set(data_config["allowed_schemes"]):
            counters[f"{source}::unsupported_scheme"] += 1
            return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum_length))
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters[f"{source}::prior_project_identity"] += 1
            return False
        if normalized in seen_urls or hostname in seen_hosts or digest in seen_hashes:
            counters[f"{source}::snapshot_duplicate"] += 1
            return False
        seen_urls.add(normalized)
        seen_hosts.add(hostname)
        seen_hashes.add(digest)
        verified_after_calibration = False
        if source_timestamp:
            try:
                verified_after_calibration = datetime.fromisoformat(source_timestamp) > calibration_frozen_at
            except ValueError:
                pass
        selected.append({
            "url": normalized, "label": label, "source": source, "source_id": source_id,
            "source_timestamp": source_timestamp, "verified_after_calibration": int(verified_after_calibration),
            "source_rank": "" if rank is None else rank, "canonical_hostname": hostname, "raw_url": raw_url,
        })
        return True

    openphish_path = args.raw_dir / "openphish_feed.txt"
    openphish_lines = [
        line.strip() for line in openphish_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    for index, raw_url in enumerate(openphish_lines, start=1):
        accept(raw_url, 1, "OpenPhish current community feed", str(index))

    phishtank_path = args.raw_dir / "phishtank_online_valid.csv.gz"
    with gzip.open(phishtank_path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishtank_rows = list(csv.DictReader(handle))
    phishtank_rows.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    for row in phishtank_rows:
        if sum(int(item["label"]) == 1 for item in selected) >= args.per_class:
            break
        if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
            counters["PhishTank verified-online::not_verified_online"] += 1
            continue
        accept(
            row.get("url", ""), 1, "PhishTank verified-online", row.get("phish_id", ""),
            row.get("verification_time", ""),
        )

    phishing_hosts = set(seen_hosts)
    tranco_path = args.raw_dir / "tranco_top1m.csv.zip"
    with zipfile.ZipFile(tranco_path) as archive:
        with archive.open(archive.namelist()[0]) as binary:
            tranco_rows = [
                (int(rank), domain.strip()) for rank, domain in csv.reader(
                    line.decode("utf-8", errors="replace") for line in binary
                )
            ]
    random.Random(args.seed).shuffle(tranco_rows)
    for rank, domain in tranco_rows:
        if sum(int(item["label"]) == 0 for item in selected) >= args.per_class:
            break
        try:
            _, hostname, _ = canonicalize_url(f"https://{domain}/")
        except ValueError:
            continue
        if hostname in phishing_hosts:
            counters["Tranco root proxy::current_phishing_hostname"] += 1
            continue
        accept(f"https://{domain}/", 0, "Tranco current root proxy", str(rank), rank=rank)

    counts = Counter(int(row["label"]) for row in selected)
    if counts != Counter({0: args.per_class, 1: args.per_class}):
        raise RuntimeError(f"Insufficient disjoint rows: legitimate={counts[0]}, phishing={counts[1]}")
    selected.sort(key=lambda row: (int(row["label"]), str(row["source"]), str(row["url"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    source_counts = Counter(str(row["source"]) for row in selected)
    later_count = sum(int(row["verified_after_calibration"]) for row in selected)
    report = {
        "project_version": 3,
        "purpose": "large frozen external evaluation only; never training, calibration, threshold, or model selection",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_frozen_at_utc": calibration_report["frozen_at_utc"],
        "rows": len(selected), "legitimate_rows": counts[0], "phishing_rows": counts[1],
        "source_counts": dict(source_counts), "verified_after_calibration_phishing_rows": later_count,
        "unique_urls": len(seen_urls), "unique_hostnames": len(seen_hosts),
        "unique_encoded_rows": len(seen_hashes),
        "measured_overlap": {"urls": 0, "hostnames": 0, "encoded_rows": 0},
        "overlap_contract": "zero URL, hostname, or 128-token identity overlap with all training, development, calibration, and prior external identities",
        "dataset": str(output_csv), "dataset_sha256": sha256_file(output_csv), "seed": args.seed,
        "sources": {
            "openphish": {"path": str(openphish_path), "sha256": sha256_file(openphish_path), "feed_rows": len(openphish_lines)},
            "phishtank": {"path": str(phishtank_path), "sha256": sha256_file(phishtank_path), "feed_rows": len(phishtank_rows)},
            "tranco": {"path": str(tranco_path), "sha256": sha256_file(tranco_path), "list_rows": len(tranco_rows)},
        },
        "filter_counters": dict(counters),
        "limitations": [
            "Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.",
            "Legitimate rows are roots while phishing rows often include paths and queries.",
            "The balanced class prevalence does not represent browser traffic.",
            "Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.",
            "The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.",
        ],
    }
    write_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
