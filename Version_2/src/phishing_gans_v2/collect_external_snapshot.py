from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import encode_url, encoded_digest, normalize_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small, balanced external URL snapshot.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--openphish", type=Path, required=True)
    parser.add_argument("--tranco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/2026-08-09"))
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--tranco-pool", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); project = load_json(args.config); data = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(data["maximum_encoded_length"])
    connection = sqlite3.connect("data/processed/master.sqlite")
    master_hostnames = {row[0] for row in connection.execute("SELECT DISTINCT hostname FROM urls")}
    counters = Counter(); selected: list[dict] = []; seen_urls = set(); seen_hashes = set(); phishing_hosts = set()

    def accept(raw_url: str, label: int, source: str, reference: str, rank: int | None = None) -> bool:
        counters[f"{source}_candidates"] += 1
        try:
            normalized, hostname = normalize_url(raw_url, int(data["maximum_raw_url_length"]))
        except ValueError:
            counters[f"{source}_invalid"] += 1; return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if normalized in seen_urls or digest in seen_hashes:
            counters[f"{source}_external_duplicate"] += 1; return False
        if connection.execute("SELECT 1 FROM urls WHERE normalized_url=?", (normalized,)).fetchone():
            counters[f"{source}_master_exact_overlap"] += 1; return False
        if connection.execute("SELECT 1 FROM encoded_rows WHERE encoded_hash=?", (digest,)).fetchone():
            counters[f"{source}_master_encoded_overlap"] += 1; return False
        if hostname in master_hostnames:
            counters[f"{source}_master_hostname_overlap"] += 1; return False
        if label == 0 and hostname in phishing_hosts:
            counters[f"{source}_phishing_hostname_conflict"] += 1; return False
        seen_urls.add(normalized); seen_hashes.add(digest)
        if label == 1: phishing_hosts.add(hostname)
        selected.append({
            "url": normalized, "label": label, "source": source,
            "source_reference": reference, "source_rank": "" if rank is None else rank,
            "canonical_hostname": hostname,
            "master_hostname_overlap": 0,
        })
        return True

    phishing = [line.strip() for line in args.openphish.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    for url in phishing:
        if sum(row["label"] == 1 for row in selected) >= args.per_class: break
        accept(url, 1, "OpenPhish Community Feed", "https://openphish.com/feed.txt")

    with zipfile.ZipFile(args.tranco) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as binary:
            rows = list(csv.reader(io.TextIOWrapper(binary, encoding="utf-8")))[:args.tranco_pool]
    random.Random(args.seed).shuffle(rows)
    for row in rows:
        if sum(item["label"] == 0 for item in selected) >= args.per_class: break
        if len(row) < 2: continue
        accept(f"https://{row[1].strip()}/", 0, "Tranco Top Sites", "https://tranco-list.eu/", int(row[0]))

    counts = Counter(row["label"] for row in selected)
    if counts[0] != args.per_class or counts[1] != args.per_class:
        raise RuntimeError(f"Could not build balanced snapshot: benign={counts[0]}, phishing={counts[1]}")
    selected.sort(key=lambda row: (row["label"], row["source_rank"] or 0, row["url"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "external_urls.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys()); writer.writeheader(); writer.writerows(selected)
    report = {
        "project_version": 2, "purpose": "one-time external evaluation only",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "per_class": args.per_class, "total": len(selected), "label_0": counts[0], "label_1": counts[1],
        "master_exact_overlap": 0, "master_encoded_overlap": 0,
        "master_hostname_overlap": sum(row["master_hostname_overlap"] for row in selected),
        "source_files": {
            "openphish": {"path": str(args.openphish), "sha256": sha256_file(args.openphish), "raw_rows": len(phishing)},
            "tranco": {"path": str(args.tranco), "sha256": sha256_file(args.tranco), "candidate_pool": args.tranco_pool},
        },
        "filter_counters": dict(counters), "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "label_note": "OpenPhish feed entries are phishing; Tranco top-site root URLs are treated as legitimate proxies.",
    }
    write_json(args.output_dir / "collection_report.json", report); connection.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
