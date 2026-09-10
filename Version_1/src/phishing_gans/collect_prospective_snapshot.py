from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .prepare_later_collected import normalized_hostname, normalized_url, sha256
from .prepare_post_training_snapshot import host_key, parse_time


FIELDS = (
    "url",
    "hostname",
    "observed_at",
    "source",
    "source_id",
    "first_collected_at",
    "snapshot_sha256",
)
API_URL = "https://api.phishstats.info/api/phishing"


def load_ledger(path: Path) -> tuple[list[dict], set[int], set[str]]:
    if not path.exists():
        return [], set(), set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        rows,
        {host_key(row["hostname"]) for row in rows},
        {normalized_url(row["url"]) for row in rows},
    )


def write_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def accept(
    record: dict,
    source: str,
    cutoff: datetime,
    exclusions: set[int],
    seen_hosts: set[int],
    seen_urls: set[str],
    collected_at: str,
    snapshot_hash: str,
    counts: Counter,
) -> dict | None:
    try:
        observed = parse_time(str(record.get("date") or record.get("observed_at") or ""))
    except ValueError:
        counts["invalid_time"] += 1
        return None
    if observed <= cutoff:
        counts["not_after_cutoff"] += 1
        return None
    url = str(record.get("url") or "").strip()
    normalized = normalized_url(url)
    host = normalized_hostname(url)
    if not normalized or not host:
        counts["invalid_url"] += 1
        return None
    key = host_key(host)
    if key in exclusions:
        counts["excluded_prior_hostname"] += 1
        return None
    if key in seen_hosts:
        counts["duplicate_or_accumulated_hostname"] += 1
        return None
    if normalized in seen_urls:
        counts["duplicate_or_accumulated_url"] += 1
        return None
    seen_hosts.add(key)
    seen_urls.add(normalized)
    return {
        "url": url,
        "hostname": host,
        "observed_at": observed.isoformat(),
        "source": source,
        "source_id": record.get("id") or record.get("source_id") or "",
        "first_collected_at": collected_at,
        "snapshot_sha256": snapshot_hash,
    }


def fetch_pages(
    raw_dir: Path, start_page: int, pages: int, size: int
) -> tuple[list[tuple[Path, list]], list[dict]]:
    snapshots = []
    manifests = []
    for page in range(start_page, start_page + pages):
        query = urllib.parse.urlencode({"_sort": "-date", "_p": page, "_size": size})
        request = urllib.request.Request(
            f"{API_URL}?{query}",
            headers={"User-Agent": "GraduationResearch-ProspectiveCollector/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            headers = dict(response.headers.items())
        path = raw_dir / f"phishstats_page_{page:02d}.json"
        path.write_bytes(payload)
        records = json.loads(payload)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list from {API_URL}, received {type(records).__name__}")
        digest = hashlib.sha256(payload).hexdigest()
        snapshots.append((path, records))
        manifests.append(
            {"page": page, "path": str(path), "sha256": digest, "records": len(records), "headers": headers}
        )
    return snapshots, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Accumulate an evaluation-only prospective phishing ledger.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--import-evaluation-csv", type=Path, action="append", default=None)
    parser.add_argument("--import-api-json", type=Path, action="append", default=None)
    parser.add_argument("--import-phishtank-bz2", type=Path, action="append", default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    if args.target < 100:
        raise ValueError("Prospective target must be at least 100 examples per class.")
    if not 1 <= args.pages <= 20 or args.start_page < 1 or not 1 <= args.page_size <= 100:
        raise ValueError("Use 1-10 pages and a page size of 1-100.")

    cutoff = parse_time(args.cutoff)
    collected = datetime.now(timezone.utc)
    collected_at = collected.isoformat()
    stamp = collected.strftime("%Y-%m-%dT%H%M%SZ")
    exclusions = {int(value) for value in np.load(args.exclusions)}
    ledger_path = args.root / "candidates.csv"
    rows, seen_hosts, seen_urls = load_ledger(ledger_path)
    counts: Counter = Counter()

    for path in args.import_evaluation_csv or []:
        snapshot_hash = sha256(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                if str(record.get("label")) != "1":
                    continue
                counts["import_rows"] += 1
                row = accept(
                    record,
                    "phishstats_post_training_import",
                    cutoff,
                    exclusions,
                    seen_hosts,
                    seen_urls,
                    collected_at,
                    snapshot_hash,
                    counts,
                )
                if row:
                    rows.append(row)
                    counts["import_accepted"] += 1

    for path in args.import_api_json or []:
        snapshot_hash = sha256(path)
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected {path} to contain a JSON list.")
        for record in records:
            counts["import_api_rows"] += 1
            row = accept(
                record,
                "phishstats",
                cutoff,
                exclusions,
                seen_hosts,
                seen_urls,
                collected_at,
                snapshot_hash,
                counts,
            )
            if row:
                rows.append(row)
                counts["import_api_accepted"] += 1
            if len(rows) >= args.target:
                break
        if len(rows) >= args.target:
            break

    for path in args.import_phishtank_bz2 or []:
        snapshot_hash = sha256(path)
        with bz2.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for record in csv.DictReader(handle):
                counts["import_phishtank_rows"] += 1
                if (record.get("verified") or "").lower() != "yes" or (record.get("online") or "").lower() != "yes":
                    counts["import_phishtank_not_verified_online"] += 1
                    continue
                normalized_record = {
                    "url": record.get("url"),
                    "date": record.get("submission_time"),
                    "id": record.get("phish_id"),
                }
                row = accept(
                    normalized_record,
                    "phishtank_verified_online",
                    cutoff,
                    exclusions,
                    seen_hosts,
                    seen_urls,
                    collected_at,
                    snapshot_hash,
                    counts,
                )
                if row:
                    rows.append(row)
                    counts["import_phishtank_accepted"] += 1
                if len(rows) >= args.target:
                    break
        if len(rows) >= args.target:
            break

    manifest = []
    if not args.skip_fetch:
        raw_dir = args.root / "raw" / stamp
        raw_dir.mkdir(parents=True, exist_ok=False)
        snapshots, manifest = fetch_pages(
            raw_dir, args.start_page, args.pages, args.page_size
        )
        for path, records in snapshots:
            digest = sha256(path)
            for record in records:
                counts["api_rows"] += 1
                row = accept(
                    record,
                    "phishstats",
                    cutoff,
                    exclusions,
                    seen_hosts,
                    seen_urls,
                    collected_at,
                    digest,
                    counts,
                )
                if row:
                    rows.append(row)
                    counts["api_accepted"] += 1
        (raw_dir / "manifest.json").write_text(
            json.dumps({"collected_at_utc": collected_at, "source": API_URL, "pages": manifest, "counts": dict(counts)}, indent=2),
            encoding="utf-8",
        )

    rows.sort(key=lambda row: (row["observed_at"], row["hostname"], row["url"]))
    write_ledger(ledger_path, rows)
    status = {
        "updated_at_utc": collected_at,
        "policy": "evaluation-only; exact-hostname unseen; no URL visits; never use for training/calibration/model selection",
        "cutoff_utc": cutoff.isoformat(),
        "target_phishing_hostnames": args.target,
        "current_phishing_hostnames": len(rows),
        "remaining": max(args.target - len(rows), 0),
        "ready_to_freeze": len(rows) >= args.target,
        "evaluated": False,
        "exclusions": str(args.exclusions),
        "exclusions_sha256": sha256(args.exclusions),
        "ledger": str(ledger_path),
        "ledger_sha256": sha256(ledger_path),
        "last_collection_counts": dict(counts),
        "recommended_collection_interval": "once per day; PhishStats documentation says its data refresh cycle is approximately 90 minutes",
    }
    (args.root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
