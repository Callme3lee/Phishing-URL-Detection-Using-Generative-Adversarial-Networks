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
    parser = argparse.ArgumentParser(description="Freeze the final browser-prevalence confirmation cohort.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/master.sqlite"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/external/2026-08-13-browser-prevalence-final/raw"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/external/2026-08-13-browser-prevalence-final"))
    parser.add_argument("--legitimate", type=int, default=50_000)
    parser.add_argument("--phishing", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_813_2)
    parser.add_argument(
        "--phishing-source",
        choices=("combined", "phishtank-only"),
        default="combined",
        help="Use the original combined feed policy or exclude OpenPhish completely.",
    )
    parser.add_argument("--purpose", default="one-shot final browser-prevalence confirmation; never training, calibration, threshold selection, or model selection")
    return parser.parse_args()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "external_urls.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"Final cohort already frozen at {args.output_dir}")
    if args.legitimate <= args.phishing or args.phishing < 100:
        raise ValueError("Browser cohort must be legitimate-dominant with enough phishing rows to estimate recall.")

    rng = random.Random(args.seed)
    project = load_json(args.config)
    data_config = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum_length = int(data_config["maximum_encoded_length"])

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
    identity_files = [p for p in identity_files if p.resolve() != output_csv.resolve()]
    prior_identity_files = []
    for path in identity_files:
        if not path.exists():
            continue
        prior_identity_files.append({"path": str(path), "sha256": sha256_file(path)})
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
    selected_counts: Counter[int] = Counter()

    def accept(raw_url: str, label: int, source: str, source_id: str,
               source_timestamp: str = "", source_target: str = "", rank: int | None = None) -> bool:
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
        selected.append({
            "url": normalized, "label": label, "source": source, "source_id": source_id,
            "source_timestamp": source_timestamp, "source_target": source_target,
            "source_rank": "" if rank is None else rank, "canonical_hostname": hostname,
            "raw_url": raw_url,
        })
        selected_counts[label] += 1
        return True

    openphish_path = args.raw_dir / "openphish_feed.txt"
    openphish_lines: list[str] = []
    if args.phishing_source == "combined":
        openphish_lines = [line.strip() for line in openphish_path.read_text(
            encoding="utf-8", errors="replace").splitlines() if line.strip()]
        rng.shuffle(openphish_lines)
        for index, raw_url in enumerate(openphish_lines, start=1):
            if selected_counts[1] >= args.phishing:
                break
            accept(raw_url, 1, "OpenPhish current community feed", str(index))

    phishtank_path = args.raw_dir / "phishtank_online_valid.csv.gz"
    with gzip.open(phishtank_path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishtank_rows = [row for row in csv.DictReader(handle)
                          if row.get("verified", "").lower() == "yes"
                          and row.get("online", "").lower() == "yes"]
    # Prefer the most recently verified rows while preserving seeded order inside a timestamp.
    rng.shuffle(phishtank_rows)
    phishtank_rows.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    for row in phishtank_rows:
        if selected_counts[1] >= args.phishing:
            break
        accept(row.get("url", ""), 1, "PhishTank verified-online", row.get("phish_id", ""),
               row.get("verification_time", ""), row.get("target", ""))

    phishing_hosts = set(seen_hosts)
    tranco_path = args.raw_dir / "tranco_top1m.csv.zip"
    with zipfile.ZipFile(tranco_path) as archive:
        with archive.open(archive.namelist()[0]) as binary:
            tranco_rows = [(int(rank), domain.strip()) for rank, domain in csv.reader(
                line.decode("utf-8", errors="replace") for line in binary)]
    # Weighted browser traffic cannot be inferred from a rank list. Random sampling avoids a top-rank-only cohort.
    rng.shuffle(tranco_rows)
    for rank, domain in tranco_rows:
        if selected_counts[0] >= args.legitimate:
            break
        try:
            _, hostname, _ = canonicalize_url(f"https://{domain}/")
        except ValueError:
            continue
        if hostname in phishing_hosts:
            counters["Tranco current root proxy::current_phishing_hostname"] += 1
            continue
        accept(f"https://{domain}/", 0, "Tranco current root proxy", str(rank), rank=rank)

    counts = Counter(int(row["label"]) for row in selected)
    if counts != Counter({0: args.legitimate, 1: args.phishing}):
        raise RuntimeError(f"Insufficient disjoint rows: legitimate={counts[0]}/{args.legitimate}, "
                           f"phishing={counts[1]}/{args.phishing}")
    rng.shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)

    phishing_times = [str(row["source_timestamp"]) for row in selected
                      if int(row["label"]) == 1 and row["source_timestamp"]]
    total = args.legitimate + args.phishing
    report = {
        "project_version": 3,
        "purpose": args.purpose,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_locked_before_inference": True,
        "rows": total, "legitimate_rows": counts[0], "phishing_rows": counts[1],
        "phishing_prevalence": counts[1] / total,
        "seed": args.seed,
        "source_counts": dict(Counter(str(row["source"]) for row in selected)),
        "unique_urls": len(seen_urls), "unique_hostnames": len(seen_hosts),
        "unique_encoded_rows": len(seen_hashes),
        "measured_overlap": {"urls": 0, "hostnames": 0, "encoded_rows": 0},
        "overlap_contract": "zero URL, hostname, or 128-token identity overlap with training, development, calibration, and every earlier external cohort",
        "prior_identity_files": prior_identity_files,
        "dataset": str(output_csv), "dataset_sha256": sha256_file(output_csv),
        "phishing_source_policy": args.phishing_source,
        "sources": {
            "phishtank": {"reference": "https://phishtank.org/developer_info.php",
                          "path": str(phishtank_path), "sha256": sha256_file(phishtank_path),
                          "verified_online_feed_rows": len(phishtank_rows),
                          "selected_verification_time_range": {
                              "earliest": min(phishing_times) if phishing_times else None,
                              "latest": max(phishing_times) if phishing_times else None}},
            "tranco": {"reference": "https://tranco-list.eu/", "path": str(tranco_path),
                       "sha256": sha256_file(tranco_path), "list_rows": len(tranco_rows)},
        },
        "filter_counters": dict(counters),
        "limitations": [
            "Tranco rank membership is a legitimate-domain proxy, not manual page-level adjudication.",
            "Root-domain legitimate URLs underrepresent deep legitimate paths, parameters, and redirects.",
            f"The {100 * counts[1] / total:.2f}% prevalence is a declared deployment simulation, not a measurement of one person's browsing history.",
            ("PhishTank does not cover every phishing campaign, language, or region."
             if args.phishing_source == "phishtank-only"
             else "PhishTank and the limited OpenPhish feed do not cover every campaign, language, or region."),
            "Source families have appeared in earlier project work, although every URL, hostname, and encoded identity here is unseen.",
        ],
    }
    if args.phishing_source == "combined":
        report["sources"]["openphish"] = {
            "reference": "https://openphish.com/phishing_feeds.html",
            "path": str(openphish_path),
            "sha256": sha256_file(openphish_path),
            "feed_rows": len(openphish_lines),
        }
    write_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
