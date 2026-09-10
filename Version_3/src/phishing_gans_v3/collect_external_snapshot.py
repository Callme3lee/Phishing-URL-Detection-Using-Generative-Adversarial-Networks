from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import canonicalize_url, encode_url, encoded_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a Version 3 PhishTank/Top-Domain external snapshot.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--phishtank", type=Path, default=Path("data/external/2026-08-11/raw/phishtank_online_valid.csv.gz"))
    parser.add_argument("--top-domains", type=Path, default=Path("../Version_2/data/raw/top10milliondomains.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/2026-08-11"))
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--top-rank-pool", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def prior_external_identity() -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hosts: set[str] = set()
    roots = (Path("data/external"), Path("../Version_2/data/external"))
    for root in roots:
        for path in sorted(root.glob("*/external_urls.csv")):
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw = row.get("raw_url") or row.get("url") or ""
                    try:
                        normalized, hostname, _ = canonicalize_url(raw)
                    except ValueError:
                        continue
                    urls.add(normalized)
                    hosts.add(hostname)
    return urls, hosts


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "external_urls.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"External snapshot is already frozen at {args.output_dir}")
    project = load_json(args.config)
    config = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(config["maximum_encoded_length"])
    database = sqlite3.connect("data/processed/master.sqlite")
    master_hosts = {row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls")}
    prior_urls, prior_hosts = prior_external_identity()
    counters: Counter = Counter()
    selected: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hashes: set[bytes] = set()
    seen_hosts: set[str] = set()

    def accept(raw_url: str, label: int, source: str, source_id: str,
               source_timestamp: str, source_target: str, rank: int | None) -> bool:
        counters[f"{source}::candidates"] += 1
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url, int(config["maximum_raw_url_length"]))
        except ValueError as error:
            counters[f"{source}::invalid::{error}"] += 1
            return False
        if scheme not in set(config["allowed_schemes"]):
            counters[f"{source}::unsupported_scheme"] += 1
            return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if normalized in seen_urls or digest in seen_hashes:
            counters[f"{source}::snapshot_url_or_encoding_duplicate"] += 1
            return False
        if hostname in seen_hosts:
            counters[f"{source}::snapshot_hostname_duplicate"] += 1
            return False
        if normalized in prior_urls or hostname in prior_hosts:
            counters[f"{source}::prior_external_overlap"] += 1
            return False
        if database.execute("SELECT 1 FROM urls WHERE normalized_url=?", (normalized,)).fetchone():
            counters[f"{source}::master_exact_overlap"] += 1
            return False
        if database.execute("SELECT 1 FROM encoded_rows WHERE encoded_hash=?", (digest,)).fetchone():
            counters[f"{source}::master_encoded_overlap"] += 1
            return False
        if hostname in master_hosts:
            counters[f"{source}::master_hostname_overlap"] += 1
            return False
        seen_urls.add(normalized)
        seen_hashes.add(digest)
        seen_hosts.add(hostname)
        selected.append({
            "url": normalized, "label": label, "source": source, "source_id": source_id,
            "source_timestamp": source_timestamp, "source_target": source_target,
            "source_rank": "" if rank is None else rank, "canonical_hostname": hostname,
            "raw_url": raw_url,
        })
        return True

    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishing_rows = list(csv.DictReader(handle))
    phishing_rows.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    for row in phishing_rows:
        if sum(int(item["label"]) == 1 for item in selected) >= args.per_class:
            break
        if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
            counters["PhishTank::not_verified_online"] += 1
            continue
        accept(row.get("url", ""), 1, "PhishTank verified-online", row.get("phish_id", ""),
               row.get("verification_time", ""), row.get("target", ""), None)

    candidates: list[tuple[int, str]] = []
    with args.top_domains.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            rank = int(row["Rank"])
            if rank > args.top_rank_pool:
                break
            candidates.append((rank, row["Domain"].strip()))
    random.Random(args.seed).shuffle(candidates)
    for rank, domain in candidates:
        if sum(int(item["label"]) == 0 for item in selected) >= args.per_class:
            break
        accept(f"https://{domain}/", 0, "Top-Domain legitimate proxy", str(rank), "", "", rank)

    counts = Counter(int(row["label"]) for row in selected)
    if counts[0] != args.per_class or counts[1] != args.per_class:
        raise RuntimeError(f"Could not build balanced snapshot: legitimate={counts[0]}, phishing={counts[1]}")
    selected.sort(key=lambda row: (int(row["label"]), int(row["source_rank"] or 0), str(row["url"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    selected_phishing_times = [str(row["source_timestamp"]) for row in selected if int(row["label"]) == 1]
    report = {
        "project_version": 3,
        "purpose": "frozen external evaluation only; never training, threshold selection, or model selection",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(selected), "legitimate_rows": counts[0], "phishing_rows": counts[1],
        "unique_urls": len(seen_urls), "unique_encoded_rows": len(seen_hashes),
        "unique_hostnames": len(seen_hosts), "master_exact_overlap": 0,
        "master_encoded_overlap": 0, "master_hostname_overlap": 0,
        "prior_external_overlap": 0, "seed": args.seed,
        "dataset": str(output_csv), "dataset_sha256": sha256_file(output_csv),
        "sources": {
            "phishing": {
                "name": "PhishTank verified-online hourly feed",
                "reference": "https://phishtank.org/developer_info.php",
                "path": str(args.phishtank), "sha256": sha256_file(args.phishtank),
                "feed_rows": len(phishing_rows),
                "latest_feed_verification_time": max(row.get("verification_time", "") for row in phishing_rows),
                "selected_verification_time_range": {"earliest": min(selected_phishing_times), "latest": max(selected_phishing_times)},
            },
            "legitimate": {
                "name": "local Top-Domain ranked-list root proxies",
                "path": str(args.top_domains), "sha256": sha256_file(args.top_domains),
                "candidate_rank_pool": args.top_rank_pool, "candidate_rows_loaded": len(candidates),
                "note": "This is a newly sampled evaluation snapshot from the local ranked list, not a newly downloaded ranking file.",
            },
        },
        "filter_counters": dict(counters),
        "canonicalization": config["canonicalization"],
        "limitations": [
            "Top-Domain membership is a legitimacy proxy rather than a manual safety audit.",
            "Legitimate examples are root domains, while phishing examples may contain deep paths and queries.",
            "The balanced 500-row snapshot does not estimate deployment prevalence.",
            "PhishTank is one phishing source and cannot represent every campaign or region.",
        ],
    }
    write_json(output_report, report)
    database.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
