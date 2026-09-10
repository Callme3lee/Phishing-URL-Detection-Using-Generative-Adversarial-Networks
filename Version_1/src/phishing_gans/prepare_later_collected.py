from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .common import encode_url, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen, hostname-leakage-screened later-collected URL test set."
    )
    parser.add_argument("--openphish", type=Path, required=True)
    parser.add_argument("--tranco-zip", type=Path, required=True)
    parser.add_argument("--prior-source", type=Path, action="append", required=True)
    parser.add_argument("--training-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hostname(url: str) -> str:
    text = url.strip()
    if not text:
        return ""
    candidate = text if "://" in text else f"http://{text}"
    try:
        host = (urlsplit(candidate).hostname or "").strip(".").lower()
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def normalized_url(url: str) -> str:
    text = url.strip()
    candidate = text if "://" in text else f"http://{text}"
    try:
        parts = urlsplit(candidate)
        host = normalized_hostname(candidate)
        if not host:
            return ""
        port = f":{parts.port}" if parts.port else ""
        path = parts.path or "/"
        return urlunsplit((parts.scheme.lower(), host + port, path, parts.query, ""))
    except ValueError:
        return ""


def iter_csv_urls(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}
        url_column = lookup.get("url")
        if url_column is None:
            raise ValueError(f"No URL column found in {path}: {reader.fieldnames}")
        for row in reader:
            yield (row.get(url_column) or "").strip()


def load_prior_hosts(paths: list[Path]) -> tuple[set[str], Counter]:
    hosts: set[str] = set()
    counts: Counter = Counter()
    for path in paths:
        for url in iter_csv_urls(path):
            counts["prior_rows"] += 1
            host = normalized_hostname(url)
            if not host:
                counts["prior_invalid_hostname"] += 1
                continue
            hosts.add(host)
        counts[f"prior_rows::{path}"] = counts["prior_rows"] - sum(
            value for key, value in counts.items() if key.startswith("prior_rows::")
        )
    counts["unique_prior_hostnames"] = len(hosts)
    return hosts, counts


def load_openphish(path: Path, prior_hosts: set[str]) -> tuple[list[dict], Counter]:
    accepted: list[dict] = []
    counts: Counter = Counter()
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            counts["openphish_rows"] += 1
            url = line.strip()
            normalized = normalized_url(url)
            host = normalized_hostname(url)
            if not normalized or not host:
                counts["openphish_invalid"] += 1
                continue
            if host in prior_hosts:
                counts["openphish_prior_hostname_overlap"] += 1
                continue
            if normalized in seen_urls:
                counts["openphish_duplicate_url"] += 1
                continue
            if host in seen_hosts:
                counts["openphish_duplicate_hostname"] += 1
                continue
            seen_urls.add(normalized)
            seen_hosts.add(host)
            accepted.append({"url": url, "label": 1, "source": "openphish", "hostname": host})
    counts["openphish_accepted"] = len(accepted)
    return accepted, counts


def load_matched_tranco(
    path: Path, prior_hosts: set[str], forbidden_hosts: set[str], target: int
) -> tuple[list[dict], Counter]:
    accepted: list[dict] = []
    counts: Counter = Counter()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in Tranco archive, found: {names}")
        with archive.open(names[0]) as binary_handle:
            lines = (line.decode("utf-8", errors="replace") for line in binary_handle)
            for row in csv.reader(lines):
                counts["tranco_rows_examined"] += 1
                if len(row) < 2:
                    counts["tranco_invalid"] += 1
                    continue
                rank, domain = row[0].strip(), row[1].strip()
                host = normalized_hostname(domain)
                if not host:
                    counts["tranco_invalid"] += 1
                    continue
                if host in prior_hosts:
                    counts["tranco_prior_hostname_overlap"] += 1
                    continue
                if host in forbidden_hosts:
                    counts["tranco_cross_class_hostname_overlap"] += 1
                    continue
                accepted.append(
                    {
                        "url": f"https://{host}/",
                        "label": 0,
                        "source": "tranco",
                        "hostname": host,
                        "rank": int(rank),
                    }
                )
                forbidden_hosts.add(host)
                if len(accepted) == target:
                    break
    if len(accepted) != target:
        raise RuntimeError(f"Only found {len(accepted)} eligible Tranco rows; needed {target}.")
    counts["tranco_accepted"] = len(accepted)
    return accepted, counts


def main() -> None:
    args = parse_args()
    metadata = load_json(args.training_data_dir / "metadata.json")
    prior_hosts, counts = load_prior_hosts(args.prior_source)
    phishing, phishing_counts = load_openphish(args.openphish, prior_hosts)
    if not phishing:
        raise RuntimeError("No OpenPhish URLs remained after leakage screening.")
    benign, benign_counts = load_matched_tranco(
        args.tranco_zip,
        prior_hosts,
        {row["hostname"] for row in phishing},
        len(phishing),
    )
    counts.update(phishing_counts)
    counts.update(benign_counts)

    # Interleave classes deterministically. No random sample selection is performed.
    rows: list[dict] = []
    for benign_row, phishing_row in zip(benign, phishing):
        rows.extend((benign_row, phishing_row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        args.output_dir / "test_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), metadata["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        args.output_dir / "test_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    with (args.output_dir / "frozen_urls.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "label", "source", "hostname", "rank"])
        writer.writeheader()
        for index, row in enumerate(rows):
            tokens[index] = encode_url(row["url"], metadata["vocab"], metadata["max_length"])
            labels[index] = row["label"]
            writer.writerow(row)
    tokens.flush()
    labels.flush()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "frozen later-collected external test; never used for training, calibration, threshold selection, or model selection",
        "class_semantics": {"0": "benign", "1": "phishing"},
        "counts": dict(counts),
        "final_examples": len(rows),
        "final_class_counts": {"benign": len(benign), "phishing": len(phishing)},
        "leakage_screen": {
            "scope": "canonical exact hostname against every row in all listed prior project datasets",
            "canonicalization": "lowercase IDNA hostname, trailing dot removed, leading www removed",
            "prior_sources": [str(path) for path in args.prior_source],
        },
        "source_snapshots": {
            "openphish": {
                "path": str(args.openphish),
                "sha256": sha256(args.openphish),
                "url": "https://openphish.com/feed.txt",
            },
            "tranco": {
                "path": str(args.tranco_zip),
                "sha256": sha256(args.tranco_zip),
                "url": "https://tranco-list.eu/top-1m.csv.zip",
            },
        },
        "encoded_with": str(args.training_data_dir / "metadata.json"),
        "important_limitations": [
            "Collection is demonstrably later than model training, but the feeds do not prove that every domain was first created after training.",
            "OpenPhish community labels are feed intelligence and may contain label noise.",
            "Tranco membership is used as a benign proxy and is not an absolute safety guarantee.",
            "Tranco examples are canonical domain-root URLs, whereas OpenPhish examples can contain full paths and queries.",
            "The community feed is small, so confidence intervals will be wider than for the main benchmark.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
