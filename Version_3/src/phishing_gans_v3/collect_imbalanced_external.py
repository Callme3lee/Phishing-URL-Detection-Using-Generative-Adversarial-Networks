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
    parser = argparse.ArgumentParser(
        description="Freeze a reproducibly imbalanced, identity-disjoint external URL snapshot."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/master.sqlite"))
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("data/external/2026-08-13-imbalanced-30k/raw"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/2026-08-13-imbalanced-30k"),
    )
    parser.add_argument("--total", type=int, default=30_000)
    parser.add_argument("--minimum-phishing-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-phishing-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20_260_813)
    return parser.parse_args()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "external_urls.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"External snapshot is already frozen at {args.output_dir}")
    if not 0 < args.minimum_phishing_fraction < args.maximum_phishing_fraction < 0.5:
        raise ValueError("Phishing fraction bounds must be ordered, positive, and below 0.5")

    rng = random.Random(args.seed)
    minimum_phishing = round(args.total * args.minimum_phishing_fraction)
    maximum_phishing = round(args.total * args.maximum_phishing_fraction)
    phishing_target = rng.randint(minimum_phishing, maximum_phishing)
    legitimate_target = args.total - phishing_target

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
    identity_files = [path for path in identity_files if path.resolve() != output_csv.resolve()]
    for path in identity_files:
        if not path.exists():
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
    selected_counts: Counter[int] = Counter()

    def accept(
        raw_url: str, label: int, source: str, source_id: str,
        source_timestamp: str = "", source_target: str = "", rank: int | None = None,
    ) -> bool:
        nonlocal selected_counts
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
            "url": normalized,
            "label": label,
            "source": source,
            "source_id": source_id,
            "source_timestamp": source_timestamp,
            "source_target": source_target,
            "source_rank": "" if rank is None else rank,
            "canonical_hostname": hostname,
            "raw_url": raw_url,
        })
        selected_counts[label] += 1
        return True

    openphish_path = args.raw_dir / "openphish_feed.txt"
    openphish_lines = [
        line.strip() for line in openphish_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    rng.shuffle(openphish_lines)
    for index, raw_url in enumerate(openphish_lines, start=1):
        accept(raw_url, 1, "OpenPhish current community feed", str(index))

    phishtank_path = args.raw_dir / "phishtank_online_valid.csv.gz"
    with gzip.open(phishtank_path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishtank_rows = list(csv.DictReader(handle))
    phishtank_rows = [
        row for row in phishtank_rows
        if row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes"
    ]
    rng.shuffle(phishtank_rows)
    for row in phishtank_rows:
        if selected_counts[1] >= phishing_target:
            break
        accept(
            row.get("url", ""), 1, "PhishTank verified-online", row.get("phish_id", ""),
            row.get("verification_time", ""), row.get("target", ""),
        )

    phishing_hosts = set(seen_hosts)
    tranco_path = args.raw_dir / "tranco_top1m.csv.zip"
    with zipfile.ZipFile(tranco_path) as archive:
        with archive.open(archive.namelist()[0]) as binary:
            tranco_rows = [
                (int(rank), domain.strip())
                for rank, domain in csv.reader(
                    line.decode("utf-8", errors="replace") for line in binary
                )
            ]
    rng.shuffle(tranco_rows)
    for rank, domain in tranco_rows:
        if selected_counts[0] >= legitimate_target:
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
    expected = Counter({0: legitimate_target, 1: phishing_target})
    if counts != expected:
        raise RuntimeError(
            f"Insufficient disjoint rows: legitimate={counts[0]}/{legitimate_target}, "
            f"phishing={counts[1]}/{phishing_target}"
        )
    rng.shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    phishing_times = [
        str(row["source_timestamp"]) for row in selected
        if int(row["label"]) == 1 and row["source_timestamp"]
    ]
    report = {
        "project_version": 3,
        "purpose": "frozen imbalanced external evaluation only; never training, calibration, threshold, or model selection",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_count_selection": {
            "method": "uniform seeded integer draw before model evaluation",
            "seed": args.seed,
            "total": args.total,
            "minimum_phishing_fraction": args.minimum_phishing_fraction,
            "maximum_phishing_fraction": args.maximum_phishing_fraction,
            "selected_phishing_rows": phishing_target,
            "selected_legitimate_rows": legitimate_target,
            "selected_phishing_prevalence": phishing_target / args.total,
        },
        "rows": len(selected),
        "legitimate_rows": counts[0],
        "phishing_rows": counts[1],
        "source_counts": dict(Counter(str(row["source"]) for row in selected)),
        "unique_urls": len(seen_urls),
        "unique_hostnames": len(seen_hosts),
        "unique_encoded_rows": len(seen_hashes),
        "measured_overlap": {"urls": 0, "hostnames": 0, "encoded_rows": 0},
        "overlap_contract": "zero URL, hostname, or 128-token identity overlap with training, development, calibration, and all prior external identities",
        "dataset": str(output_csv),
        "dataset_sha256": sha256_file(output_csv),
        "sources": {
            "openphish": {
                "reference": "https://openphish.com/phishing_feeds.html",
                "path": str(openphish_path), "sha256": sha256_file(openphish_path),
                "feed_rows": len(openphish_lines),
            },
            "phishtank": {
                "reference": "https://phishtank.org/developer_info.php",
                "path": str(phishtank_path), "sha256": sha256_file(phishtank_path),
                "verified_online_feed_rows": len(phishtank_rows),
                "selected_verification_time_range": {
                    "earliest": min(phishing_times), "latest": max(phishing_times),
                },
            },
            "tranco": {
                "reference": "https://tranco-list.eu/",
                "path": str(tranco_path), "sha256": sha256_file(tranco_path),
                "list_rows": len(tranco_rows),
            },
        },
        "filter_counters": dict(counters),
        "limitations": [
            "Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.",
            "Legitimate examples are root domains while phishing examples often contain paths and queries.",
            "The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.",
            "PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.",
            "The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.",
        ],
    }
    write_json(output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
