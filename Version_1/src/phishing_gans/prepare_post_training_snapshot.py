from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import encode_url, load_json
from .prepare_later_collected import normalized_hostname, normalized_url, sha256


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).astimezone(timezone.utc)


def host_key(host: str) -> int:
    return int.from_bytes(hashlib.blake2b(host.encode("ascii"), digest_size=8).digest(), "big")


def load_used_hosts(paths: list[Path], url_lists: list[Path]) -> tuple[set[int], Counter]:
    used: set[int] = set()
    counts: Counter = Counter()
    for path in paths:
        rows = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                host = normalized_hostname(row.get("hostname") or row.get("url") or "")
                if host:
                    used.add(host_key(host))
                else:
                    counts["invalid_used_hostname"] += 1
        counts[f"used_rows::{path}"] = rows
    for path in url_lists:
        rows = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for url in handle:
                rows += 1
                host = normalized_hostname(url)
                if host:
                    used.add(host_key(host))
                else:
                    counts["invalid_used_hostname"] += 1
        counts[f"used_rows::{path}"] = rows
    counts["unique_used_hostname_hashes"] = len(used)
    return used, counts


def load_phishstats(
    paths: list[Path], cutoff: datetime, used_hosts: set[int]
) -> tuple[list[dict], Counter]:
    accepted = []
    counts: Counter = Counter()
    seen_urls: set[str] = set()
    seen_hosts: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected {path} to contain a JSON list.")
        for record in payload:
            counts["phishstats_rows"] += 1
            try:
                observed = parse_time(str(record.get("date") or ""))
            except ValueError:
                counts["phishstats_invalid_time"] += 1
                continue
            if observed <= cutoff:
                counts["phishstats_not_after_cutoff"] += 1
                continue
            url = str(record.get("url") or "").strip()
            normalized = normalized_url(url)
            host = normalized_hostname(url)
            if not normalized or not host:
                counts["phishstats_invalid_url"] += 1
                continue
            key = host_key(host)
            if key in used_hosts:
                counts["phishstats_used_hostname"] += 1
                continue
            if normalized in seen_urls:
                counts["phishstats_duplicate_url"] += 1
                continue
            if key in seen_hosts:
                counts["phishstats_duplicate_hostname"] += 1
                continue
            seen_urls.add(normalized)
            seen_hosts.add(key)
            accepted.append(
                {
                    "url": url,
                    "hostname": host,
                    "label": 1,
                    "source": "phishstats_post_training",
                    "observed_at": observed.isoformat(),
                    "source_id": record.get("id"),
                    "rank": "",
                }
            )
    counts["phishstats_accepted"] = len(accepted)
    return accepted, counts


def load_benign(
    path: Path, target: int, used_hosts: set[int], malicious: list[dict]
) -> tuple[list[dict], Counter]:
    forbidden = set(used_hosts)
    forbidden.update(host_key(row["hostname"]) for row in malicious)
    accepted = []
    counts: Counter = Counter()
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as binary_handle:
            lines = (line.decode("utf-8", errors="replace") for line in binary_handle)
            for raw in csv.reader(lines):
                counts["tranco_rows_examined"] += 1
                if len(raw) < 2:
                    counts["tranco_invalid"] += 1
                    continue
                host = normalized_hostname(raw[1])
                if not host:
                    counts["tranco_invalid"] += 1
                    continue
                key = host_key(host)
                if key in forbidden:
                    counts["tranco_used_hostname"] += 1
                    continue
                forbidden.add(key)
                accepted.append(
                    {
                        "url": f"https://{host}/",
                        "hostname": host,
                        "label": 0,
                        "source": "tranco_unused_control",
                        "observed_at": "",
                        "source_id": "",
                        "rank": int(raw[0]),
                    }
                )
                if len(accepted) == target:
                    break
    if len(accepted) != target:
        raise RuntimeError(f"Found {len(accepted)} benign controls; needed {target}.")
    counts["tranco_accepted"] = len(accepted)
    return accepted, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a genuinely post-training URL snapshot.")
    parser.add_argument("--phishstats-json", type=Path, action="append", required=True)
    parser.add_argument("--tranco-zip", type=Path, required=True)
    parser.add_argument("--used-csv", type=Path, action="append", required=True)
    parser.add_argument("--used-url-list", type=Path, action="append", default=None)
    parser.add_argument("--training-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--phishing-cutoff", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cutoff = parse_time(args.phishing_cutoff)
    encoding = load_json(args.training_data_dir / "metadata.json")
    used_hosts, counts = load_used_hosts(args.used_csv, args.used_url_list or [])
    phishing, source_counts = load_phishstats(args.phishstats_json, cutoff, used_hosts)
    if not phishing:
        raise RuntimeError("No genuinely post-cutoff unseen phishing URLs remained.")
    benign, benign_counts = load_benign(args.tranco_zip, len(phishing), used_hosts, phishing)
    counts.update(source_counts)
    counts.update(benign_counts)

    rows = []
    for benign_row, phishing_row in zip(benign, phishing):
        rows.extend((benign_row, phishing_row))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        args.output_dir / "test_tokens.npy", mode="w+", dtype=np.uint8,
        shape=(len(rows), encoding["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        args.output_dir / "test_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    with (args.output_dir / "frozen_urls.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["url", "hostname", "label", "source", "observed_at", "source_id", "rank"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            tokens[index] = encode_url(row["url"], encoding["vocab"], encoding["max_length"])
            labels[index] = row["label"]
            writer.writerow(row)
    tokens.flush()
    labels.flush()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "post-training frozen external snapshot; no training, calibration, threshold or checkpoint selection",
        "phishing_cutoff_utc": cutoff.isoformat(),
        "final_examples": len(rows),
        "final_class_counts": {"benign": len(benign), "phishing": len(phishing)},
        "counts": dict(counts),
        "used_hostname_scope": [str(path) for path in args.used_csv],
        "used_url_list_scope": [str(path) for path in (args.used_url_list or [])],
        "source_snapshots": {
            "phishstats": {
                "files": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in args.phishstats_json
                ],
                "url": "https://api.phishstats.info/api/phishing",
            },
            "tranco": {"path": str(args.tranco_zip), "sha256": sha256(args.tranco_zip), "url": "https://tranco-list.eu/top-1m.csv.zip"},
        },
        "important_limitations": [
            "The independent phishing source is genuinely timestamped after the training-source cutoff.",
            "The benign controls are unused hostnames from the current Tranco daily file, not newly registered domains.",
            "PhishStats is an external intelligence source and may contain label noise.",
            "The snapshot is small and confidence intervals across only three training seeds do not capture sample-level uncertainty.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
