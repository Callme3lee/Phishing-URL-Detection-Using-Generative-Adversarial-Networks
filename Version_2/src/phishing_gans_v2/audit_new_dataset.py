from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import encode_url, encoded_digest, lexical_features, normalize_url


LABELS = {"legitimate": 0, "phishing": 1}
FEATURE_NAMES = (
    "url_length", "hostname_length", "path_length", "query_length",
    "digit_count", "digit_ratio", "letter_count", "special_count",
    "dot_count_host", "estimated_subdomains", "query_delimiters", "https",
    "ip_hostname", "character_entropy", "suspicious_token_count",
    "double_slash_count", "double_dot_count", "unique_character_count",
    "dot_count", "hyphen_count", "underscore_count", "slash_count",
    "question_count", "colon_count", "equals_count", "at_count",
    "percent_count", "ampersand_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only diversity and leakage audit for URLs_New.csv")
    parser.add_argument("--dataset", type=Path, default=Path("../URLs_New.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/new_dataset_audit"))
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    args = parse_args(); project = load_json(Path("configs/project.json"))
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(project["data"]["maximum_encoded_length"])
    raw_seen: dict[str, int] = {}; normalized_seen: dict[str, int] = {}
    encoded_seen: dict[bytes, int] = {}; host_labels: dict[str, set[int]] = defaultdict(set)
    host_counts = {0: Counter(), 1: Counter()}; label_counts = Counter(); invalid = Counter()
    raw_duplicates = normalized_duplicates = encoded_duplicates = 0
    raw_conflicts = normalized_conflicts = encoded_conflicts = 0
    feature_rows = {0: [], 1: []}; normalized_records = []
    label_runs = []; last_label = None; run_start = 1

    with args.dataset.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["url", "label"]:
            raise RuntimeError(f"Expected url,label schema; received {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=1):
            raw = row["url"].strip(); label_name = row["label"].strip().lower()
            if label_name not in LABELS:
                invalid[f"unknown_label:{label_name}"] += 1; continue
            label = LABELS[label_name]; label_counts[label] += 1
            if last_label is None:
                last_label = label; run_start = row_number
            elif label != last_label:
                label_runs.append({"label": last_label, "start": run_start, "stop": row_number - 1})
                last_label = label; run_start = row_number
            previous = raw_seen.get(raw)
            if previous is not None:
                raw_duplicates += 1; raw_conflicts += int(previous != label)
            else:
                raw_seen[raw] = label
            try:
                normalized, hostname = normalize_url(raw, project["data"]["maximum_raw_url_length"])
            except ValueError as error:
                invalid[str(error)] += 1; continue
            previous = normalized_seen.get(normalized)
            if previous is not None:
                normalized_duplicates += 1; normalized_conflicts += int(previous != label)
            else:
                normalized_seen[normalized] = label
            tokens = encode_url(normalized, vocabulary, maximum)
            digest = encoded_digest(tokens)
            previous = encoded_seen.get(digest)
            if previous is not None:
                encoded_duplicates += 1; encoded_conflicts += int(previous != label)
            else:
                encoded_seen[digest] = label
            host_labels[hostname].add(label); host_counts[label][hostname] += 1
            feature_rows[label].append(lexical_features(normalized))
            scheme_free = normalized.split("://", 1)[1] if "://" in normalized else normalized
            normalized_records.append((normalized, scheme_free, hostname, digest, label))
    if last_label is not None:
        label_runs.append({"label": last_label, "start": run_start, "stop": sum(label_counts.values())})

    profiles = {}
    indices = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    for label in (0, 1):
        values = np.asarray(feature_rows[label], dtype=np.float32)
        count = len(values); hosts = host_counts[label]
        top = hosts.most_common(20)
        profiles[str(label)] = {
            "label": "legitimate" if label == 0 else "phishing",
            "rows": count,
            "unique_hostnames": len(hosts),
            "rows_per_unique_hostname": safe_ratio(count, len(hosts)),
            "top_1_hostname_share": safe_ratio(top[0][1], count) if top else 0.0,
            "top_10_hostname_share": safe_ratio(sum(value for _, value in top[:10]), count),
            "top_hostnames": [{"hostname": host, "rows": value} for host, value in top],
            "root_or_empty_path_rate": float((values[:, indices["path_length"]] <= 1).mean()),
            "query_present_rate": float((values[:, indices["query_length"]] > 0).mean()),
            "https_rate": float((values[:, indices["https"]] > 0).mean()),
            "scheme_marker_rate": float((values[:, indices["double_slash_count"]] > 0).mean()),
            "longer_than_127_rate": float((values[:, indices["url_length"]] > 127).mean()),
            "ip_hostname_rate": float((values[:, indices["ip_hostname"]] > 0).mean()),
            "suspicious_token_present_rate": float((values[:, indices["suspicious_token_count"]] > 0).mean()),
            "feature_means": {name: float(values[:, index].mean()) for name, index in indices.items()},
            "url_length_quantiles": {
                str(point): float(value) for point, value in zip(
                    (0, .25, .5, .75, .9, .95, .99, 1),
                    np.quantile(values[:, indices["url_length"]], (0, .25, .5, .75, .9, .95, .99, 1)),
                )
            },
        }

    # Use a temporary table in the existing SQLite database: no persistent
    # Version 2 data is changed.
    database = sqlite3.connect("file:data/processed/master.sqlite?mode=ro", uri=True)
    database.execute("CREATE TEMP TABLE candidate(normalized_url TEXT, scheme_free_url TEXT, hostname TEXT, encoded_hash BLOB, label INTEGER)")
    database.executemany("INSERT INTO candidate VALUES(?,?,?,?,?)", normalized_records)
    database.execute("CREATE INDEX temp.candidate_url ON candidate(normalized_url)")
    database.execute("CREATE INDEX temp.candidate_scheme_free ON candidate(scheme_free_url)")
    database.execute("CREATE INDEX temp.candidate_host ON candidate(hostname)")
    database.execute("CREATE INDEX temp.candidate_hash ON candidate(encoded_hash)")
    overlap = {
        "exact_rows": database.execute("SELECT COUNT(*) FROM candidate c JOIN urls u ON u.normalized_url=c.normalized_url").fetchone()[0],
        "exact_unique_urls": database.execute("SELECT COUNT(DISTINCT c.normalized_url) FROM candidate c JOIN urls u ON u.normalized_url=c.normalized_url").fetchone()[0],
        "exact_label_conflicts": database.execute("SELECT COUNT(*) FROM candidate c JOIN urls u ON u.normalized_url=c.normalized_url WHERE c.label!=u.label").fetchone()[0],
        "scheme_insensitive_exact_rows": database.execute("SELECT COUNT(*) FROM candidate c JOIN urls u ON u.normalized_url=c.scheme_free_url").fetchone()[0],
        "scheme_insensitive_unique_urls": database.execute("SELECT COUNT(DISTINCT c.scheme_free_url) FROM candidate c JOIN urls u ON u.normalized_url=c.scheme_free_url").fetchone()[0],
        "scheme_insensitive_label_conflicts": database.execute("SELECT COUNT(*) FROM candidate c JOIN urls u ON u.normalized_url=c.scheme_free_url WHERE c.label!=u.label").fetchone()[0],
        "hostname_rows": database.execute("SELECT COUNT(*) FROM candidate c WHERE EXISTS(SELECT 1 FROM urls u WHERE u.hostname=c.hostname)").fetchone()[0],
        "unique_hostnames": database.execute("SELECT COUNT(DISTINCT c.hostname) FROM candidate c WHERE EXISTS(SELECT 1 FROM urls u WHERE u.hostname=c.hostname)").fetchone()[0],
        "encoded_rows": database.execute("SELECT COUNT(*) FROM candidate c JOIN encoded_rows e ON e.encoded_hash=c.encoded_hash").fetchone()[0],
        "encoded_label_conflicts": database.execute("SELECT COUNT(*) FROM candidate c JOIN encoded_rows e ON e.encoded_hash=c.encoded_hash WHERE c.label!=e.label").fetchone()[0],
    }
    database.close()

    valid_rows = len(normalized_records)
    total_rows = sum(label_counts.values())
    legitimate_https = int(sum(row[indices["https"]] > 0 for row in feature_rows[0]))
    phishing_https = int(sum(row[indices["https"]] > 0 for row in feature_rows[1]))
    scheme_tn = legitimate_https
    scheme_fp = len(feature_rows[0]) - legitimate_https
    scheme_fn = phishing_https
    scheme_tp = len(feature_rows[1]) - phishing_https
    scheme_only = {
        "rule": "predict legitimate for HTTPS and phishing otherwise",
        "accuracy": safe_ratio(scheme_tn + scheme_tp, valid_rows),
        "precision": safe_ratio(scheme_tp, scheme_tp + scheme_fp),
        "recall": safe_ratio(scheme_tp, scheme_tp + scheme_fn),
        "false_positive_rate": safe_ratio(scheme_fp, scheme_fp + scheme_tn),
        "confusion_matrix": {"tn": scheme_tn, "fp": scheme_fp, "fn": scheme_fn, "tp": scheme_tp},
    }

    payload = {
        "dataset": str(args.dataset), "sha256": sha256_file(args.dataset),
        "schema": ["url", "label"], "rows": total_rows, "valid_rows": valid_rows,
        "invalid_rows": total_rows - valid_rows, "invalid_reasons": dict(invalid),
        "label_counts": {"legitimate": label_counts[0], "phishing": label_counts[1]},
        "label_fractions": {"legitimate": safe_ratio(label_counts[0], total_rows), "phishing": safe_ratio(label_counts[1], total_rows)},
        "label_runs": label_runs,
        "raw_unique_urls": len(raw_seen), "raw_duplicate_rows": raw_duplicates, "raw_label_conflicts": raw_conflicts,
        "normalized_unique_urls": len(normalized_seen), "normalized_duplicate_rows": normalized_duplicates,
        "normalized_label_conflicts": normalized_conflicts,
        "encoded_unique_rows": len(encoded_seen), "encoded_duplicate_rows": encoded_duplicates,
        "encoded_label_conflicts": encoded_conflicts,
        "unique_hostnames": len(host_labels),
        "hostnames_present_in_both_labels": sum(len(labels) > 1 for labels in host_labels.values()),
        "rows_on_cross_label_hostnames": sum(sum(host_counts[label][host] for label in (0, 1)) for host, labels in host_labels.items() if len(labels) > 1),
        "profiles": profiles, "https_scheme_only_classifier": scheme_only,
        "overlap_with_version_2_master": overlap,
        "metadata_limitations": [
            "No source column: independent-source diversity cannot be verified.",
            "No collection date: temporal diversity and chronological splitting cannot be verified.",
            "Rows are sorted in two contiguous label blocks and must be shuffled/group-split before training.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "audit.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
