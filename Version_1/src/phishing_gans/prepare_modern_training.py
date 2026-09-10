from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import encode_url, load_json
from .prepare_later_collected import iter_csv_urls, normalized_hostname, normalized_url, sha256


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).astimezone(timezone.utc)


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def file_sha256_optional(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def load_excluded_hosts(prior_sources: list[Path], frozen_csv: Path) -> tuple[set[str], Counter]:
    hosts: set[str] = set()
    counts: Counter = Counter()
    for source in prior_sources:
        source_count = 0
        for url in iter_csv_urls(source):
            source_count += 1
            host = normalized_hostname(url)
            if host:
                hosts.add(host)
            else:
                counts["invalid_prior_hostname"] += 1
        counts[f"prior_rows::{source}"] = source_count
    with frozen_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            host = normalized_hostname(row.get("hostname") or row.get("url") or "")
            if host:
                hosts.add(host)
                counts["frozen_rows"] += 1
    counts["excluded_unique_hostnames"] = len(hosts)
    return hosts, counts


def add_candidate(
    candidates: dict[str, dict],
    url: str,
    observed_at: datetime,
    source: str,
    excluded_hosts: set[str],
    counts: Counter,
) -> None:
    normalized = normalized_url(url)
    host = normalized_hostname(url)
    if not normalized or not host:
        counts[f"{source}_invalid"] += 1
        return
    if host in excluded_hosts:
        counts[f"{source}_excluded_hostname"] += 1
        return
    current = candidates.get(normalized)
    if current is None:
        candidates[normalized] = {
            "url": url.strip(),
            "hostname": host,
            "first_seen": observed_at,
            "last_seen": observed_at,
            "sources": {source},
        }
    else:
        current["first_seen"] = min(current["first_seen"], observed_at)
        current["last_seen"] = max(current["last_seen"], observed_at)
        current["sources"].add(source)
        counts["cross_snapshot_or_source_duplicates"] += 1


def load_openphish_history(
    repo: Path,
    start: datetime,
    end: datetime,
    excluded_hosts: set[str],
    candidates: dict[str, dict],
    counts: Counter,
) -> dict:
    log = git_output(
        repo,
        "log",
        "--reverse",
        f"--since={start.isoformat()}",
        f"--until={end.isoformat()}",
        "--format=%H%x09%aI",
        "--",
        "feed.txt",
    )
    commits = []
    for line in log.splitlines():
        commit, timestamp = line.split("\t", 1)
        observed_at = parse_time(timestamp)
        feed = git_output(repo, "show", f"{commit}:feed.txt")
        row_count = 0
        for url in feed.splitlines():
            if not url.strip():
                continue
            row_count += 1
            counts["openphish_history_rows"] += 1
            add_candidate(
                candidates, url, observed_at, "openphish", excluded_hosts, counts
            )
        commits.append({"commit": commit, "timestamp": observed_at.isoformat(), "rows": row_count})
    counts["openphish_commits"] = len(commits)
    return {
        "repo": str(repo),
        "head": git_output(repo, "rev-parse", "HEAD").strip(),
        "commits": commits,
    }


def load_phishtank(
    path: Path,
    start: datetime,
    end: datetime,
    excluded_hosts: set[str],
    candidates: dict[str, dict],
    counts: Counter,
) -> None:
    with bz2.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["phishtank_rows"] += 1
            if (row.get("verified") or "").lower() != "yes" or (row.get("online") or "").lower() != "yes":
                counts["phishtank_not_verified_online"] += 1
                continue
            try:
                observed_at = parse_time(row.get("submission_time") or "")
            except ValueError:
                counts["phishtank_invalid_time"] += 1
                continue
            if not start <= observed_at <= end:
                counts["phishtank_outside_window"] += 1
                continue
            counts["phishtank_in_window"] += 1
            add_candidate(
                candidates,
                row.get("url") or "",
                observed_at,
                "phishtank",
                excluded_hosts,
                counts,
            )


def assign_splits(candidates: dict[str, dict], val_start: datetime, test_start: datetime) -> dict[str, list[dict]]:
    host_first_seen: dict[str, datetime] = {}
    for row in candidates.values():
        host = row["hostname"]
        host_first_seen[host] = min(host_first_seen.get(host, row["first_seen"]), row["first_seen"])
    splits = {"train": [], "val": [], "test": []}
    for row in candidates.values():
        group_time = host_first_seen[row["hostname"]]
        split = "test" if group_time >= test_start else "val" if group_time >= val_start else "train"
        output = dict(row)
        output["first_seen"] = row["first_seen"].isoformat()
        output["last_seen"] = row["last_seen"].isoformat()
        output["sources"] = "+".join(sorted(row["sources"]))
        splits[split].append(output)
    for rows in splits.values():
        rows.sort(key=lambda row: (row["first_seen"], row["url"]))
    return splits


def write_encoded_splits(output_dir: Path, splits: dict[str, list[dict]], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        tokens = np.lib.format.open_memmap(
            output_dir / f"{split}_tokens.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows), metadata["max_length"]),
        )
        labels = np.lib.format.open_memmap(
            output_dir / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
        )
        labels[:] = 1
        with (output_dir / f"{split}_urls.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["url", "hostname", "first_seen", "last_seen", "sources"],
            )
            writer.writeheader()
            for index, row in enumerate(rows):
                tokens[index] = encode_url(row["url"], metadata["vocab"], metadata["max_length"])
                writer.writerow(row)
        tokens.flush()
        labels.flush()


def write_hard_negatives(
    tranco_zip: Path,
    output: Path,
    excluded_hosts: set[str],
    malicious_hosts: set[str],
    limit: int,
) -> dict:
    rows = []
    seen = set()
    rejected = Counter()
    with zipfile.ZipFile(tranco_zip) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as binary_handle:
            lines = (line.decode("utf-8", errors="replace") for line in binary_handle)
            for raw in csv.reader(lines):
                if len(raw) < 2:
                    rejected["invalid"] += 1
                    continue
                host = normalized_hostname(raw[1])
                if not host:
                    rejected["invalid"] += 1
                    continue
                if host in excluded_hosts:
                    rejected["old_or_frozen_hostname"] += 1
                    continue
                if host in malicious_hosts:
                    rejected["modern_malicious_hostname"] += 1
                    continue
                if host in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(host)
                rows.append({"url": f"https://{host}/", "hostname": host, "rank": int(raw[0]), "label": 0})
                if len(rows) == limit:
                    break
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "hostname", "rank", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return {"accepted": len(rows), "rejected": dict(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare temporally split modern URL training data.")
    parser.add_argument("--openphish-repo", type=Path, required=True)
    parser.add_argument("--phishtank", type=Path, required=True)
    parser.add_argument("--tranco-zip", type=Path, required=True)
    parser.add_argument("--frozen-csv", type=Path, required=True)
    parser.add_argument("--prior-source", type=Path, action="append", required=True)
    parser.add_argument("--base-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-08-01T23:59:59+00:00")
    parser.add_argument("--validation-start", default="2026-07-25T00:00:00+00:00")
    parser.add_argument("--test-start", default="2026-07-29T00:00:00+00:00")
    parser.add_argument("--hard-negative-count", type=int, default=50_000)
    args = parser.parse_args()

    start, end = parse_time(args.start), parse_time(args.end)
    val_start, test_start = parse_time(args.validation_start), parse_time(args.test_start)
    if not start < val_start < test_start <= end:
        raise ValueError("Require start < validation-start < test-start <= end")

    base_metadata = load_json(args.base_data_dir / "metadata.json")
    excluded_hosts, counts = load_excluded_hosts(args.prior_source, args.frozen_csv)
    candidates: dict[str, dict] = {}
    provenance = load_openphish_history(
        args.openphish_repo, start, end, excluded_hosts, candidates, counts
    )
    load_phishtank(args.phishtank, start, end, excluded_hosts, candidates, counts)
    splits = assign_splits(candidates, val_start, test_start)
    write_encoded_splits(args.output_dir, splits, base_metadata)

    malicious_hosts = {row["hostname"] for rows in splits.values() for row in rows}
    hard_negative_report = write_hard_negatives(
        args.tranco_zip,
        args.output_dir / "modern_benign_hard_negatives.csv",
        excluded_hosts,
        malicious_hosts,
        args.hard_negative_count,
    )
    split_sources = {}
    for split, rows in splits.items():
        source_counts = Counter(row["sources"] for row in rows)
        split_sources[split] = dict(source_counts)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "modern malicious GAN adaptation corpus plus separate modern benign hard negatives",
        "class_semantics": {"0": "benign", "1": "phishing"},
        "time_window": {
            "start": start.isoformat(),
            "validation_start": val_start.isoformat(),
            "test_start": test_start.isoformat(),
            "end": end.isoformat(),
        },
        "split_policy": "canonical hostname grouped by its earliest observation; temporal cutoffs prevent hostname leakage",
        "frozen_test_policy": "every canonical hostname from the 31 July frozen evaluation is excluded",
        "old_project_policy": "every canonical hostname from all three prior project CSVs is excluded",
        "counts": dict(counts),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "split_source_counts": split_sources,
        "unique_modern_malicious_hostnames": len(malicious_hosts),
        "hard_negatives": hard_negative_report,
        "vocab": base_metadata["vocab"],
        "max_length": base_metadata["max_length"],
        "snapshots": {
            "openphish": provenance,
            "phishtank": {"path": str(args.phishtank), "sha256": sha256(args.phishtank)},
            "tranco": {"path": str(args.tranco_zip), "sha256": sha256(args.tranco_zip)},
            "frozen_exclusion_csv": {"path": str(args.frozen_csv), "sha256": sha256(args.frozen_csv)},
        },
        "important_note": "The 31 July result motivated this work, so a newly collected future snapshot is required for an unbiased final claim.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("time_window", "split_counts", "split_source_counts", "unique_modern_malicious_hostnames", "hard_negatives")}, indent=2))


if __name__ == "__main__":
    main()
