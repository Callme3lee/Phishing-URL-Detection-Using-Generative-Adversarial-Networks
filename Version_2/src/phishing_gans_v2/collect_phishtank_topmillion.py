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
from .url_data import encode_url, encoded_digest, normalize_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a balanced PhishTank/Top-Million external snapshot.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--phishtank", type=Path, default=Path("data/external/2026-08-11/raw/phishtank_online_valid.csv.gz"))
    parser.add_argument("--top-million", type=Path, default=Path("data/raw/top10milliondomains.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/2026-08-11"))
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--top-rank-pool", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); project = load_json(args.config); data = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(data["maximum_encoded_length"])
    connection = sqlite3.connect("data/processed/master.sqlite")
    master_hostnames = {row[0] for row in connection.execute("SELECT DISTINCT hostname FROM urls")}

    prior_urls: set[str] = set(); prior_hosts: set[str] = set()
    for path in sorted(Path("data/external").glob("*/external_urls.csv")):
        if path.parent == args.output_dir: continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                prior_urls.add(row["url"])
                prior_hosts.add(row.get("canonical_hostname", ""))

    counters: Counter = Counter(); selected: list[dict] = []
    seen_urls: set[str] = set(); seen_hashes: set[bytes] = set(); seen_hosts: set[str] = set()

    def accept(raw_url: str, label: int, source: str, source_id: str, timestamp: str, target: str, rank: int | None) -> bool:
        counters[f"{source}_candidates"] += 1
        try:
            normalized, hostname = normalize_url(raw_url, int(data["maximum_raw_url_length"]))
        except ValueError:
            counters[f"{source}_invalid"] += 1; return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if normalized in seen_urls or digest in seen_hashes:
            counters[f"{source}_snapshot_duplicate"] += 1; return False
        if hostname in seen_hosts:
            counters[f"{source}_snapshot_hostname_duplicate"] += 1; return False
        if normalized in prior_urls or hostname in prior_hosts:
            counters[f"{source}_prior_external_overlap"] += 1; return False
        if connection.execute("SELECT 1 FROM urls WHERE normalized_url=?", (normalized,)).fetchone():
            counters[f"{source}_master_exact_overlap"] += 1; return False
        if connection.execute("SELECT 1 FROM encoded_rows WHERE encoded_hash=?", (digest,)).fetchone():
            counters[f"{source}_master_encoded_overlap"] += 1; return False
        if hostname in master_hostnames:
            counters[f"{source}_master_hostname_overlap"] += 1; return False
        seen_urls.add(normalized); seen_hashes.add(digest); seen_hosts.add(hostname)
        selected.append({
            "url": normalized,
            "label": label,
            "source": source,
            "source_id": source_id,
            "source_timestamp": timestamp,
            "source_target": target,
            "source_rank": "" if rank is None else rank,
            "canonical_hostname": hostname,
        })
        return True

    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishtank_rows = list(csv.DictReader(handle))
    phishtank_rows.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    raw_verified_online = sum(
        row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes"
        for row in phishtank_rows
    )
    for row in phishtank_rows:
        if sum(item["label"] == 1 for item in selected) >= args.per_class: break
        if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
            counters["PhishTank_not_verified_online"] += 1; continue
        accept(
            row["url"], 1, "PhishTank verified online", row.get("phish_id", ""),
            row.get("verification_time", ""), row.get("target", ""), None,
        )

    top_rows: list[tuple[int, str]] = []
    with args.top_million.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            rank = int(row["Rank"])
            if rank > args.top_rank_pool: break
            top_rows.append((rank, row["Domain"].strip()))
    random.Random(args.seed).shuffle(top_rows)
    for rank, domain in top_rows:
        if sum(item["label"] == 0 for item in selected) >= args.per_class: break
        accept(
            f"https://{domain}/", 0, "Top Million legitimate proxy", str(rank),
            "", "", rank,
        )

    counts = Counter(int(row["label"]) for row in selected)
    if counts[0] != args.per_class or counts[1] != args.per_class:
        raise RuntimeError(f"Could not freeze balanced snapshot: legitimate={counts[0]}, phishing={counts[1]}")
    selected.sort(key=lambda row: (int(row["label"]), int(row["source_rank"] or 0), row["url"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "external_urls.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0])); writer.writeheader(); writer.writerows(selected)

    report = {
        "project_version": 2,
        "purpose": "frozen evaluation only; never use for training, threshold selection, or model selection",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "rows": len(selected),
        "legitimate_rows": counts[0],
        "phishing_rows": counts[1],
        "unique_urls": len(seen_urls),
        "unique_encoded_rows": len(seen_hashes),
        "unique_hostnames": len(seen_hosts),
        "master_exact_overlap": 0,
        "master_encoded_overlap": 0,
        "master_hostname_overlap": 0,
        "prior_external_url_or_hostname_overlap": 0,
        "source_files": {
            "phishtank": {
                "path": str(args.phishtank), "sha256": sha256_file(args.phishtank),
                "feed_rows": len(phishtank_rows), "verified_online_rows": raw_verified_online,
                "reference": "https://phishtank.org/developer_info.php",
            },
            "top_million": {
                "path": str(args.top_million), "sha256": sha256_file(args.top_million),
                "candidate_rank_pool": args.top_rank_pool, "candidate_rows_loaded": len(top_rows),
            },
        },
        "filter_counters": dict(counters),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "label_note": "PhishTank rows are feed-verified online phishes; Top Million root URLs are legitimate proxies.",
        "limitations": [
            "Top Million membership is a legitimacy proxy, not a manual safety audit.",
            "Root-domain legitimate URLs are structurally easier than arbitrary benign deep links.",
            "The snapshot is small and source-specific; it is not a population-wide performance estimate.",
        ],
    }
    write_json(args.output_dir / "collection_report.json", report)
    connection.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
