from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tldextract
from sklearn.preprocessing import RobustScaler

from .common import load_json, sha256_file, write_json
from .url_data import (
    PSL_AWARE_LEXICAL_FEATURE_NAMES,
    canonicalize_url,
    encode_url,
    encoded_digest,
    psl_aware_lexical_features,
)


GROUPS = {
    "urls_new_legitimate": 0,
    "outside_legitimate": 1,
    "urls_new_phishing": 2,
    "phishtank_phishing": 3,
    "openphish_phishing": 4,
}
REJECTED_COVERAGE_DOMAINS = {"anuytzc.xyz", "danatoto.com", "ebun.top", "exloader.net", "thatpervert.com"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze multi-source training data for the hybrid domain model.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--base-data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--coverage-root", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/audit"),
    )
    parser.add_argument(
        "--phishtank", type=Path,
        default=Path("data/external/2026-08-11/raw/phishtank_online_valid.csv.gz"),
    )
    parser.add_argument(
        "--openphish", type=Path,
        default=Path("data/development/benign_coverage_2026-08-11/raw/openphish_feed.txt"),
    )
    parser.add_argument("--external-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/experiments/hybrid_cnn_coral_v1"),
    )
    parser.add_argument("--phishtank-train", type=int, default=2000)
    parser.add_argument("--phishtank-validation", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=16).digest()


def external_identity(root: Path, vocabulary: dict[str, int], maximum: int) -> tuple[set[str], set[str], set[bytes]]:
    urls: set[str] = set()
    hosts: set[str] = set()
    hashes: set[bytes] = set()
    for path in root.glob("*/external_urls.csv"):
        for row in load_csv(path):
            try:
                normalized, hostname, _ = canonicalize_url(row.get("raw_url") or row.get("url") or "")
            except ValueError:
                continue
            urls.add(normalized)
            hosts.add(hostname)
            hashes.add(encoded_digest(encode_url(normalized, vocabulary, maximum)))
    return urls, hosts, hashes


def materialize_lexical(urls: list[str], output: Path) -> np.memmap:
    matrix = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(urls), len(PSL_AWARE_LEXICAL_FEATURE_NAMES)),
    )
    for index, url in enumerate(urls):
        matrix[index] = psl_aware_lexical_features(url)
        if index and index % 100000 == 0:
            print(json.dumps({"stage": "psl_features", "path": str(output), "rows": index}), flush=True)
    matrix.flush()
    return matrix


def main() -> None:
    args = parse_args()
    processed = args.output_root / "processed"
    report_path = args.output_root / "experiment_report.json"
    if report_path.exists():
        raise FileExistsError(f"Hybrid domain dataset is already frozen at {args.output_root}")
    project = load_json(args.config)
    vocabulary = load_json(args.base_data / "vocabulary.json")
    maximum = int(project["data"]["maximum_encoded_length"])
    base_manifest = load_csv(args.base_data / "train_manifest.csv")
    base_tokens = np.load(args.base_data / "train_tokens.npy", mmap_mode="r")
    base_labels = np.load(args.base_data / "train_labels.npy", mmap_mode="r")
    if len(base_manifest) != len(base_labels):
        raise RuntimeError("Base manifest and labels differ in length")

    database = sqlite3.connect(args.base_data / "master.sqlite")
    master_urls = {row[0] for row in database.execute("SELECT normalized_url FROM urls")}
    master_hosts = {row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls")}
    master_hashes = {bytes(row[0]) for row in database.execute("SELECT encoded_hash FROM encoded_rows")}
    external_urls, external_hosts, external_hashes = external_identity(args.external_root, vocabulary, maximum)
    blocked_urls = master_urls | external_urls
    blocked_hosts = master_hosts | external_hosts
    blocked_hashes = master_hashes | external_hashes

    coverage_train = [
        row for row in load_csv(args.coverage_root / "train.csv")
        if row["tranco_domain"] not in REJECTED_COVERAGE_DOMAINS
    ]
    coverage_validation = [
        row for row in load_csv(args.coverage_root / "validation.csv")
        if row["tranco_domain"] not in REJECTED_COVERAGE_DOMAINS
    ]
    for row in coverage_train + coverage_validation:
        if row["url"] in blocked_urls or row["canonical_hostname"] in blocked_hosts:
            raise RuntimeError("Coverage split overlaps the master or a frozen external hostname")

    candidates: dict[str, list[dict[str, object]]] = {"openphish": [], "phishtank": []}
    seen_source_urls: set[str] = set()
    seen_source_hosts: set[str] = set()
    seen_source_hashes: set[bytes] = set()

    def accept_source(raw_url: str, source: str, source_id: str, timestamp: str) -> None:
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url, int(project["data"]["maximum_raw_url_length"]))
        except ValueError:
            return
        if scheme not in set(project["data"]["allowed_schemes"]):
            return
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if (
            normalized in blocked_urls or hostname in blocked_hosts or digest in blocked_hashes
            or normalized in seen_source_urls or hostname in seen_source_hosts or digest in seen_source_hashes
        ):
            return
        seen_source_urls.add(normalized)
        seen_source_hosts.add(hostname)
        seen_source_hashes.add(digest)
        candidates[source].append({
            "url": normalized, "label": 1, "canonical_hostname": hostname,
            "source_group": f"{source}_phishing", "source_id": source_id,
            "source_timestamp": timestamp, "raw_url": raw_url,
        })

    for index, raw_url in enumerate(args.openphish.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if raw_url.strip():
            accept_source(raw_url.strip(), "openphish", str(index), "")
    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phishtank_rows = list(csv.DictReader(handle))
    phishtank_rows.sort(key=lambda row: row.get("verification_time", ""), reverse=True)
    for row in phishtank_rows:
        if row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes":
            accept_source(row.get("url", ""), "phishtank", row.get("phish_id", ""), row.get("verification_time", ""))

    openphish_sorted = sorted(candidates["openphish"], key=lambda row: stable_key(args.seed, str(row["url"])))
    openphish_validation_count = max(1, len(openphish_sorted) // 4)
    openphish_validation = openphish_sorted[:openphish_validation_count]
    openphish_train = openphish_sorted[openphish_validation_count:]
    phishtank_sorted = sorted(candidates["phishtank"], key=lambda row: stable_key(args.seed, str(row["url"])))
    required = args.phishtank_train + args.phishtank_validation
    if len(phishtank_sorted) < required:
        raise RuntimeError(f"Only {len(phishtank_sorted)} eligible PhishTank rows; require {required}")
    phishtank_validation = phishtank_sorted[:args.phishtank_validation]
    phishtank_train = phishtank_sorted[args.phishtank_validation:required]

    appended_train: list[dict[str, object]] = []
    for row in coverage_train:
        appended_train.append({
            "url": row["url"], "label": 0, "canonical_hostname": row["canonical_hostname"],
            "source_group": "outside_legitimate", "source_id": f"{row['tranco_list_id']}:{row['tranco_rank']}",
            "source_timestamp": row["crawl_timestamp"], "raw_url": row["raw_url"],
        })
    appended_train.extend(phishtank_train)
    appended_train.extend(openphish_train)
    domain_validation: list[dict[str, object]] = []
    for row in coverage_validation:
        domain_validation.append({
            "url": row["url"], "label": 0, "canonical_hostname": row["canonical_hostname"],
            "source_group": "outside_legitimate", "source_id": f"{row['tranco_list_id']}:{row['tranco_rank']}",
            "source_timestamp": row["crawl_timestamp"], "raw_url": row["raw_url"],
        })
    domain_validation.extend(phishtank_validation)
    domain_validation.extend(openphish_validation)

    processed.mkdir(parents=True, exist_ok=True)
    train_total = len(base_labels) + len(appended_train)
    train_tokens = np.lib.format.open_memmap(
        processed / "train_tokens.npy", mode="w+", dtype=np.uint16, shape=(train_total, maximum)
    )
    train_labels = np.lib.format.open_memmap(
        processed / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(train_total,)
    )
    train_groups = np.lib.format.open_memmap(
        processed / "train_groups.npy", mode="w+", dtype=np.uint8, shape=(train_total,)
    )
    train_tokens[:len(base_tokens)] = base_tokens
    train_labels[:len(base_labels)] = base_labels
    train_groups[:len(base_labels)] = np.where(base_labels == 0, GROUPS["urls_new_legitimate"], GROUPS["urls_new_phishing"])
    train_manifest: list[dict[str, object]] = []
    train_urls: list[str] = []
    for index, row in enumerate(base_manifest):
        group = "urls_new_legitimate" if int(row["label"]) == 0 else "urls_new_phishing"
        train_manifest.append({
            "row_index": index, "url": row["url"], "label": row["label"],
            "canonical_hostname": row["canonical_hostname"], "source_group": group,
            "source_id": row["source_row"], "source_timestamp": "", "raw_url": row["url"],
        })
        train_urls.append(row["url"])
    for offset, row in enumerate(appended_train, start=len(base_labels)):
        train_tokens[offset] = encode_url(str(row["url"]), vocabulary, maximum)
        train_labels[offset] = int(row["label"])
        train_groups[offset] = GROUPS[str(row["source_group"])]
        train_manifest.append({"row_index": offset, **row})
        train_urls.append(str(row["url"]))
    train_tokens.flush(); train_labels.flush(); train_groups.flush()
    manifest_fields = ["row_index", "url", "label", "canonical_hostname", "source_group", "source_id", "source_timestamp", "raw_url"]
    write_csv(processed / "train_manifest.csv", train_manifest, manifest_fields)

    domain_tokens = np.stack([encode_url(str(row["url"]), vocabulary, maximum) for row in domain_validation]).astype(np.uint16)
    domain_labels = np.asarray([int(row["label"]) for row in domain_validation], dtype=np.uint8)
    domain_groups = np.asarray([GROUPS[str(row["source_group"])] for row in domain_validation], dtype=np.uint8)
    np.save(processed / "domain_validation_tokens.npy", domain_tokens)
    np.save(processed / "domain_validation_labels.npy", domain_labels)
    np.save(processed / "domain_validation_groups.npy", domain_groups)
    write_csv(
        processed / "domain_validation_manifest.csv",
        [{"row_index": index, **row} for index, row in enumerate(domain_validation)], manifest_fields,
    )

    for split in ("validation", "test"):
        for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
            source = args.base_data / f"{split}_{suffix}"
            target = processed / f"{split}_{suffix}"
            target.write_bytes(source.read_bytes())
    (processed / "vocabulary.json").write_bytes((args.base_data / "vocabulary.json").read_bytes())

    raw_train = materialize_lexical(train_urls, processed / "train_psl_lexical_raw.npy")
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(np.asarray(raw_train))
    np.savez_compressed(processed / "psl_lexical_scaler.npz", center=scaler.center_, scale=scaler.scale_)

    def transform_urls(urls: list[str], output: Path, raw: np.ndarray | None = None) -> None:
        raw_values = raw if raw is not None else np.stack([psl_aware_lexical_features(url) for url in urls])
        result = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.float32,
            shape=(len(urls), len(PSL_AWARE_LEXICAL_FEATURE_NAMES)),
        )
        chunk = 50000
        for start in range(0, len(urls), chunk):
            stop = min(len(urls), start + chunk)
            result[start:stop] = scaler.transform(np.asarray(raw_values[start:stop])).astype(np.float32)
        result.flush()

    transform_urls(train_urls, processed / "train_psl_lexical.npy", raw_train)
    for split in ("validation", "test"):
        rows = load_csv(processed / f"{split}_manifest.csv")
        transform_urls([row["url"] for row in rows], processed / f"{split}_psl_lexical.npy")
    transform_urls([str(row["url"]) for row in domain_validation], processed / "domain_validation_psl_lexical.npy")
    write_json(processed / "psl_lexical_metadata.json", {
        "feature_names": PSL_AWARE_LEXICAL_FEATURE_NAMES,
        "feature_count": len(PSL_AWARE_LEXICAL_FEATURE_NAMES),
        "scaler": "RobustScaler quantile_range=(10,90), fit on hybrid training rows only",
        "tldextract_version": tldextract.__version__,
        "suffix_snapshot": "tldextract bundled public/private suffix snapshot; no network lookup",
        "exact_suffix_or_hostname_identity_included": False,
    })

    files = [
        "train_tokens.npy", "train_labels.npy", "train_groups.npy", "train_psl_lexical.npy",
        "validation_tokens.npy", "validation_labels.npy", "validation_psl_lexical.npy",
        "test_tokens.npy", "test_labels.npy", "test_psl_lexical.npy",
        "domain_validation_tokens.npy", "domain_validation_labels.npy", "domain_validation_groups.npy",
        "domain_validation_psl_lexical.npy", "train_manifest.csv", "domain_validation_manifest.csv",
        "psl_lexical_scaler.npz", "vocabulary.json",
    ]
    report = {
        "project_version": 3, "condition": "hybrid_cnn_coral_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_master_sha256": project["data"]["master_sha256"],
        "external_exclusion": {
            "urls": len(external_urls), "hostnames": len(external_hosts), "encoded_inputs": len(external_hashes),
            "contract": "every frozen external URL, hostname, and encoding excluded from all added training/development rows",
        },
        "train_rows": train_total,
        "train_group_counts": dict(Counter(str(row["source_group"]) for row in train_manifest)),
        "domain_validation_rows": len(domain_validation),
        "domain_validation_group_counts": dict(Counter(str(row["source_group"]) for row in domain_validation)),
        "source_pool": {
            "eligible_openphish_after_exclusions": len(openphish_sorted),
            "eligible_phishtank_after_exclusions": len(phishtank_sorted),
            "openphish_feed_sha256": sha256_file(args.openphish),
            "phishtank_feed_sha256": sha256_file(args.phishtank),
        },
        "group_ids": GROUPS,
        "files": {name: {"sha256": sha256_file(processed / name)} for name in files},
        "limitations": [
            "PhishTank/OpenPhish development rows come from the same dated feed snapshots as already evaluated sources, but all external rows and hostnames are excluded.",
            "Because the frozen external results motivated this design, post-training use of those sets is exploratory rather than confirmatory.",
            "OpenPhish Community has few eligible hostname-disjoint rows and will be sampled with replacement during source-balanced training.",
            "Proxy-labelled legitimate coverage is not page-level manual adjudication.",
        ],
    }
    write_json(report_path, report)
    database.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
