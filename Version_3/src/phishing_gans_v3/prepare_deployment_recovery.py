from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest, psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare separate recovery-development rows after the deployment gate failure.")
    parser.add_argument("--base", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1/gan_10pct/processed"))
    parser.add_argument("--benign", type=Path, default=Path("data/development/deployment_recovery_2026-08-15/benign_candidates.csv"))
    parser.add_argument("--phishtank", type=Path, default=Path("data/deployment/2026-08-15-precision/raw/phishtank_online_valid.csv.gz"))
    parser.add_argument("--calibration", type=Path, default=Path("data/deployment/2026-08-15-precision/calibration_urls.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/deployment_recovery_v1/gan_10pct"))
    parser.add_argument("--phishing-rows", type=int, default=5_000)
    parser.add_argument("--phishing-cutoff", default="2026-08-15T00:00:00+00:00")
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def registered(host: str) -> str:
    item = PSL_EXTRACTOR(host)
    return ".".join(value for value in (item.domain, item.suffix) if value) or host


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args(); output = args.output_root / "processed"
    report_path = args.output_root / "experiment_report.json"
    if report_path.exists():
        raise FileExistsError(f"Recovery condition already frozen at {args.output_root}")
    vocabulary = load_json(args.base / "vocabulary.json"); maximum = 128
    manifest_path = args.base / "train_manifest.csv"
    if manifest_path.exists():
        base_manifest = read_csv(manifest_path)
    else:
        base_manifest = read_csv(Path("data/experiments/hybrid_cnn_coral_v1/processed/train_manifest.csv"))
        generated = read_csv(Path("data/augmentations/ms_hardgan_v2/generated_10pct.csv"))
        for index, row in enumerate(generated, start=len(base_manifest)):
            base_manifest.append({
                "row_index": str(index), "url": row["url"], "label": "1",
                "canonical_hostname": row["hostname"], "source_group": "synthetic_phishing",
                "source_id": row["raw_sha256"], "source_timestamp": "", "raw_url": row["url"],
            })
    base_urls = {row["url"] for row in base_manifest}
    base_hosts = {row["canonical_hostname"] for row in base_manifest}
    base_domains = {registered(row["canonical_hostname"]) for row in base_manifest}
    base_hashes = {encoded_digest(encode_url(row["url"], vocabulary, maximum)) for row in base_manifest}
    calibration = read_csv(args.calibration)
    blocked_urls = base_urls | {row["url"] for row in calibration}
    blocked_hosts = base_hosts | {row["canonical_hostname"] for row in calibration}
    blocked_domains = base_domains | {row["registered_domain"] for row in calibration}
    blocked_hashes = base_hashes | {encoded_digest(encode_url(row["url"], vocabulary, maximum)) for row in calibration}
    seen_urls: set[str] = set(); seen_domains: set[str] = set(); seen_hashes: set[bytes] = set()

    additions: list[dict[str, object]] = []
    rejection: Counter = Counter()

    def accept(raw: str, label: int, group: int, source: str, source_id: str, timestamp: str) -> bool:
        try:
            url, host, _ = canonicalize_url(raw)
        except ValueError:
            rejection["invalid"] += 1; return False
        domain = registered(host); digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if (url in blocked_urls or host in blocked_hosts or domain in blocked_domains or digest in blocked_hashes):
            rejection["base_or_calibration_identity"] += 1; return False
        if url in seen_urls or domain in seen_domains or digest in seen_hashes:
            rejection["recovery_duplicate_domain"] += 1; return False
        seen_urls.add(url); seen_domains.add(domain); seen_hashes.add(digest)
        additions.append({"url": url, "label": label, "canonical_hostname": host,
                          "source_group": "recovery_legitimate" if label == 0 else "recovery_phishtank",
                          "group": group, "source_id": source_id, "source_timestamp": timestamp,
                          "raw_url": raw, "registered_domain": domain, "source": source})
        return True

    benign_rows = read_csv(args.benign)
    benign_rows.sort(key=lambda row: stable(args.seed, row["url"]))
    for row in benign_rows:
        accept(row["raw_url"], 0, 1, row["source"], row["source_id"] if "source_id" in row else row.get("tranco_rank", ""), row.get("crawl_timestamp", ""))
    legitimate_count = len(additions)

    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        phish = list(csv.DictReader(handle))
    phish = [row for row in phish if row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes"
             and row.get("verification_time", "") < args.phishing_cutoff]
    phish.sort(key=lambda row: stable(args.seed + 1, row.get("url", "")))
    before = len(additions)
    for row in phish:
        if len(additions) - before >= args.phishing_rows:
            break
        accept(row.get("url", ""), 1, 3, "PhishTank recovery development", row.get("phish_id", ""), row.get("verification_time", ""))
    phishing_count = len(additions) - before
    if legitimate_count < 250 or phishing_count < args.phishing_rows:
        raise RuntimeError(f"Insufficient recovery rows: legitimate={legitimate_count}, phishing={phishing_count}")

    output.mkdir(parents=True)
    for name in ("validation_tokens.npy", "validation_labels.npy", "validation_psl_lexical.npy", "validation_manifest.csv",
                 "test_tokens.npy", "test_labels.npy", "test_psl_lexical.npy", "test_manifest.csv",
                 "domain_validation_tokens.npy", "domain_validation_labels.npy", "domain_validation_psl_lexical.npy",
                 "domain_validation_groups.npy", "domain_validation_manifest.csv", "vocabulary.json",
                 "psl_lexical_metadata.json", "psl_lexical_scaler.npz"):
        link(args.base / name, output / name)
    base_tokens = np.load(args.base / "train_tokens.npy", mmap_mode="r")
    base_labels_array = np.load(args.base / "train_labels.npy", mmap_mode="r")
    base_groups = np.load(args.base / "train_groups.npy", mmap_mode="r")
    base_features = np.load(args.base / "train_psl_lexical.npy", mmap_mode="r")
    total = len(base_labels_array) + len(additions)
    tokens = np.lib.format.open_memmap(output / "train_tokens.npy", mode="w+", dtype=np.uint16, shape=(total, maximum))
    labels = np.lib.format.open_memmap(output / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,))
    groups = np.lib.format.open_memmap(output / "train_groups.npy", mode="w+", dtype=np.uint8, shape=(total,))
    features = np.lib.format.open_memmap(output / "train_psl_lexical.npy", mode="w+", dtype=np.float32, shape=(total, base_features.shape[1]))
    tokens[:len(base_tokens)] = base_tokens; labels[:len(base_labels_array)] = base_labels_array
    groups[:len(base_groups)] = base_groups; features[:len(base_features)] = base_features
    scaler = np.load(args.base / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32); scale = np.where(np.asarray(scaler["scale"]) == 0, 1, scaler["scale"])
    for offset, row in enumerate(additions, start=len(base_labels_array)):
        tokens[offset] = encode_url(str(row["url"]), vocabulary, maximum); labels[offset] = int(row["label"])
        groups[offset] = int(row["group"]); features[offset] = (psl_aware_lexical_features(str(row["url"])) - center) / scale
    tokens.flush(); labels.flush(); groups.flush(); features.flush()
    link(output / "train_psl_lexical.npy", output / "train_features.npy")
    for split in ("validation", "test"):
        link(output / f"{split}_psl_lexical.npy", output / f"{split}_features.npy")
    fields = ["row_index", "url", "label", "canonical_hostname", "source_group", "source_id", "source_timestamp", "raw_url"]
    with (output / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in base_manifest: writer.writerow({key: row.get(key, "") for key in fields})
        for index, row in enumerate(additions, start=len(base_manifest)):
            writer.writerow({"row_index": index, **{key: row.get(key, "") for key in fields if key != "row_index"}})
    synthetic_rows = int((base_groups == 5).sum())
    report = {
        "project_version": 3, "condition": "deployment_recovery_hyb_gan10_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": "frozen winner missed the calibration recall gate: 78.8% observed versus 80% required at FPR <= 0.25%",
        "base_condition": "fair_hybrid_gan_v1/gan_10pct", "base_training_rows": len(base_labels_array),
        "recovery_legitimate_rows": legitimate_count, "recovery_phishing_rows": phishing_count,
        "real_training_rows": int(total - synthetic_rows), "synthetic_training_rows": synthetic_rows,
        "training_rows": total, "generated_fraction": 0.10,
        "stored_generated_fraction": synthetic_rows / total,
        "sampling_note": "stored synthetic fraction is reported exactly; weighted sampling preserves the registered 10% synthetic exposure",
        "calibration_rows_used_for_training": 0, "calibration_identity_excluded": True,
        "group_mapping": {"recovery_legitimate": 1, "recovery_phishtank": 3},
        "recovery_legitimate_source_counts": dict(Counter(row["source"] for row in additions if int(row["label"]) == 0)),
        "rejections": dict(rejection),
        "source_hashes": {"benign": sha256_file(args.benign), "phishtank": sha256_file(args.phishtank), "calibration": sha256_file(args.calibration)},
        "files": {name: sha256_file(output / name) for name in ("train_tokens.npy", "train_labels.npy", "train_groups.npy", "train_psl_lexical.npy", "train_manifest.csv")},
    }
    write_json(report_path, report); print(report)


if __name__ == "__main__":
    main()
