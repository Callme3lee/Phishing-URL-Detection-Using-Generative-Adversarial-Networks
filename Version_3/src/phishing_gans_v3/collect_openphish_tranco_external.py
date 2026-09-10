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
    parser = argparse.ArgumentParser(description="Freeze an overlap-free OpenPhish/Tranco external set.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/master.sqlite"))
    parser.add_argument(
        "--openphish", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/raw/openphish_feed.txt"),
    )
    parser.add_argument(
        "--tranco", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/raw/tranco_Q2XX4_top1m.csv.zip"),
    )
    parser.add_argument(
        "--phishtank", type=Path,
        default=Path("data/external/2026-08-11/raw/phishtank_online_valid.csv.gz"),
    )
    parser.add_argument(
        "--coverage", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/benign_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/2026-08-11-openphish-tranco"),
    )
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def prior_external_identity(output_dir: Path) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hosts: set[str] = set()
    for root in (Path("data/external"), Path("../Version_2/data/external")):
        for path in sorted(root.glob("*/external_urls.csv")):
            if path.parent.resolve() == output_dir.resolve():
                continue
            for row in read_csv(path):
                try:
                    normalized, hostname, _ = canonicalize_url(row.get("raw_url") or row.get("url") or "")
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
    database = sqlite3.connect(args.database)
    master_urls = {row[0] for row in database.execute("SELECT normalized_url FROM urls")}
    master_hosts = {row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls")}
    master_hashes = {bytes(row[0]) for row in database.execute("SELECT encoded_hash FROM encoded_rows")}
    prior_urls, prior_hosts = prior_external_identity(args.output_dir)
    coverage_rows = read_csv(args.coverage)
    coverage_urls = {row["url"] for row in coverage_rows}
    coverage_hosts = {row["canonical_hostname"] for row in coverage_rows}

    phishtank_hosts: set[str] = set()
    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                _, hostname, _ = canonicalize_url(row.get("url", ""))
                phishtank_hosts.add(hostname)
            except ValueError:
                pass

    counters: Counter = Counter()
    selected: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()
    seen_hashes: set[bytes] = set()

    def accept(raw_url: str, label: int, source: str, source_id: str, rank: int | None = None) -> bool:
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
        reasons = {
            "snapshot_duplicate": normalized in seen_urls or hostname in seen_hosts or digest in seen_hashes,
            "master_overlap": normalized in master_urls or hostname in master_hosts or digest in master_hashes,
            "prior_external_overlap": normalized in prior_urls or hostname in prior_hosts,
            "coverage_overlap": normalized in coverage_urls or hostname in coverage_hosts,
        }
        for reason, matched in reasons.items():
            if matched:
                counters[f"{source}::{reason}"] += 1
                return False
        if label == 0 and hostname in phishtank_hosts:
            counters[f"{source}::phishtank_hostname"] += 1
            return False
        seen_urls.add(normalized)
        seen_hosts.add(hostname)
        seen_hashes.add(digest)
        selected.append({
            "url": normalized, "label": label, "source": source, "source_id": source_id,
            "source_timestamp": "", "source_target": "", "source_rank": "" if rank is None else rank,
            "canonical_hostname": hostname, "raw_url": raw_url,
        })
        return True

    phishing_lines = [line.strip() for line in args.openphish.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    for index, raw_url in enumerate(phishing_lines, start=1):
        if sum(int(row["label"]) == 1 for row in selected) >= args.per_class:
            break
        accept(raw_url, 1, "OpenPhish community snapshot", str(index))

    with zipfile.ZipFile(args.tranco) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as binary:
            domains = [(int(rank), domain.strip()) for rank, domain in csv.reader(
                line.decode("utf-8", errors="replace") for line in binary
            )]
    random.Random(args.seed).shuffle(domains)
    openphish_hosts = set(seen_hosts)
    for rank, domain in domains:
        if sum(int(row["label"]) == 0 for row in selected) >= args.per_class:
            break
        try:
            _, hostname, _ = canonicalize_url(f"https://{domain}/")
        except ValueError:
            continue
        if hostname in openphish_hosts:
            counters["Tranco root proxy::openphish_hostname"] += 1
            continue
        accept(f"https://{domain}/", 0, "Tranco Q2XX4 root proxy", str(rank), rank)

    counts = Counter(int(row["label"]) for row in selected)
    if counts[0] != args.per_class or counts[1] != args.per_class:
        raise RuntimeError(f"Could not build balanced snapshot: legitimate={counts[0]}, phishing={counts[1]}")
    selected.sort(key=lambda row: (int(row["label"]), int(row["source_rank"] or 0), str(row["url"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "project_version": 3,
        "purpose": "second frozen external evaluation only; never training, threshold selection, or model selection",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(selected), "legitimate_rows": counts[0], "phishing_rows": counts[1],
        "unique_urls": len(seen_urls), "unique_encoded_rows": len(seen_hashes), "unique_hostnames": len(seen_hosts),
        "master_exact_overlap": 0, "master_encoded_overlap": 0, "master_hostname_overlap": 0,
        "prior_external_overlap": 0, "development_coverage_overlap": 0,
        "seed": args.seed, "dataset": str(output_csv), "dataset_sha256": sha256_file(output_csv),
        "sources": {
            "phishing": {
                "name": "OpenPhish free community snapshot", "reference": "https://www.openphish.com/phishing_feeds.html",
                "path": str(args.openphish), "sha256": sha256_file(args.openphish), "feed_rows": len(phishing_lines),
            },
            "legitimate": {
                "name": "Tranco Q2XX4 root proxies", "reference": "https://tranco-list.eu/list/Q2XX4",
                "path": str(args.tranco), "sha256": sha256_file(args.tranco), "list_rows": len(domains),
            },
        },
        "filter_counters": dict(counters),
        "canonicalization": config["canonicalization"],
        "limitations": [
            "Tranco membership is a legitimate proxy, not page-level manual adjudication.",
            "OpenPhish Community is a limited feed without the metadata available in paid tiers.",
            "Legitimate samples are roots while phishing samples can include paths and queries.",
            "A balanced 400-row set does not estimate deployment prevalence.",
            "This is a source-transfer test collected on the same date, not a later temporal evaluation.",
        ],
    }
    write_json(output_report, report)
    database.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
