from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_json, resolve_from_config, sha256_file, write_json
from .url_data import SPECIAL_TOKENS, canonicalize_url, encode_url, encoded_digest, split_for_hostname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable, leakage-safe Version 3 dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--commit-every", type=int, default=25000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=TRUNCATE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript("""
        CREATE TABLE urls (
            normalized_url TEXT PRIMARY KEY,
            label INTEGER NOT NULL,
            hostname TEXT NOT NULL,
            split TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            raw_url TEXT NOT NULL,
            raw_scheme TEXT NOT NULL
        );
        CREATE TABLE url_conflicts (normalized_url TEXT PRIMARY KEY);
        CREATE TABLE quarantine (
            source_row INTEGER,
            raw_url TEXT,
            raw_label TEXT,
            normalized_url TEXT,
            reason TEXT NOT NULL
        );
        CREATE TABLE encoded_rows (
            encoded_hash BLOB PRIMARY KEY,
            url_rowid INTEGER NOT NULL UNIQUE,
            label INTEGER NOT NULL,
            split TEXT NOT NULL
        );
        CREATE TABLE encoded_conflicts (encoded_hash BLOB PRIMARY KEY, reason TEXT NOT NULL);
    """)
    return connection


def open_v2_reference(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def quarantine(
    connection: sqlite3.Connection, source_row: int | None, raw_url: str,
    raw_label: str, normalized_url: str | None, reason: str,
) -> None:
    connection.execute(
        "INSERT INTO quarantine VALUES (?, ?, ?, ?, ?)",
        (source_row, raw_url, raw_label, normalized_url, reason),
    )


def ingest(config: dict, raw_path: Path, connection: sqlite3.Connection, v2: sqlite3.Connection | None,
           commit_every: int, seed: int) -> dict:
    counts = Counter()
    mapping = {str(key): int(value) for key, value in config["label_mapping"].items()}
    fractions = config["split_fractions"]
    started = time.time()
    with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {config["url_column"], config["label_column"]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required)}")
        connection.execute("BEGIN")
        for source_row, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            raw_url = str(row[config["url_column"]])
            raw_label = str(row[config["label_column"]]).strip()
            if raw_label not in mapping:
                counts["unknown_label"] += 1
                quarantine(connection, source_row, raw_url, raw_label, None, "unknown_label")
                continue
            label = mapping[raw_label]
            try:
                normalized, hostname, raw_scheme = canonicalize_url(raw_url, config["maximum_raw_url_length"])
            except ValueError as error:
                reason = f"invalid::{error}"
                counts[reason] += 1
                quarantine(connection, source_row, raw_url, raw_label, None, reason)
                continue
            counts[f"raw_scheme::{raw_label}::{raw_scheme or 'absent'}"] += 1
            if raw_scheme not in set(config["allowed_schemes"]):
                counts[f"unsupported_scheme::{raw_scheme or 'absent'}"] += 1
                quarantine(connection, source_row, raw_url, raw_label, normalized, "unsupported_scheme")
                continue

            if connection.execute(
                "SELECT 1 FROM url_conflicts WHERE normalized_url=?", (normalized,)
            ).fetchone():
                counts["after_internal_label_conflict"] += 1
                quarantine(connection, source_row, raw_url, raw_label, normalized, "internal_label_conflict")
                continue

            if v2 is not None:
                previous = v2.execute(
                    "SELECT label FROM urls WHERE normalized_url=?", (normalized,)
                ).fetchone()
                if previous is not None:
                    counts["version_2_exact_overlap"] += 1
                    if int(previous[0]) != label:
                        counts["version_2_label_conflict_quarantined"] += 1
                        quarantine(connection, source_row, raw_url, raw_label, normalized, "version_2_label_conflict")
                        continue

            split = split_for_hostname(hostname, seed, fractions["train"], fractions["validation"])
            cursor = connection.execute(
                "INSERT OR IGNORE INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
                (normalized, label, hostname, split, source_row, raw_url, raw_scheme),
            )
            if cursor.rowcount == 1:
                counts["inserted_unique"] += 1
            else:
                existing = connection.execute(
                    "SELECT label, source_row, raw_url FROM urls WHERE normalized_url=?", (normalized,)
                ).fetchone()
                if existing is not None and int(existing[0]) == label:
                    counts["duplicate_same_label"] += 1
                    quarantine(connection, source_row, raw_url, raw_label, normalized, "duplicate_same_label")
                else:
                    counts["duplicate_conflicting_label"] += 1
                    if existing is not None:
                        quarantine(connection, int(existing[1]), str(existing[2]), str(existing[0]), normalized, "internal_label_conflict")
                    quarantine(connection, source_row, raw_url, raw_label, normalized, "internal_label_conflict")
                    connection.execute("DELETE FROM urls WHERE normalized_url=?", (normalized,))
                    connection.execute("INSERT OR IGNORE INTO url_conflicts VALUES (?)", (normalized,))
            if counts["input_rows"] % commit_every == 0:
                connection.commit()
                connection.execute("BEGIN")
                print(json.dumps({"stage": "ingest", "rows": counts["input_rows"], "seconds": round(time.time()-started, 1)}), flush=True)
        connection.commit()
    counts["retained_after_canonical_deduplication"] = connection.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
    counts["quarantined_records"] = connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    return dict(counts)


def build_vocabulary(config: dict, connection: sqlite3.Connection) -> dict[str, int]:
    counts = Counter()
    for (url,) in connection.execute("SELECT normalized_url FROM urls WHERE split='train'"):
        counts.update(url)
    maximum = int(config["maximum_vocabulary_size"])
    characters = [character for character, _ in counts.most_common(maximum - len(SPECIAL_TOKENS))]
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocabulary.update({character: index + len(SPECIAL_TOKENS) for index, character in enumerate(characters)})
    return vocabulary


def audit_encoded(config: dict, vocabulary: dict[str, int], connection: sqlite3.Connection,
                  commit_every: int) -> dict:
    maximum = int(config["maximum_encoded_length"])
    counts = Counter()
    started = time.time()
    connection.execute("BEGIN")
    for rowid, url, label, split, source_row, raw_url in connection.execute(
        "SELECT rowid, normalized_url, label, split, source_row, raw_url FROM urls ORDER BY rowid"
    ):
        counts["processed"] += 1
        digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if connection.execute("SELECT 1 FROM encoded_conflicts WHERE encoded_hash=?", (digest,)).fetchone():
            counts["after_encoded_conflict"] += 1
            quarantine(connection, source_row, raw_url, str(label), url, "after_encoded_conflict")
            continue
        cursor = connection.execute(
            "INSERT OR IGNORE INTO encoded_rows VALUES (?, ?, ?, ?)", (digest, rowid, label, split)
        )
        if cursor.rowcount == 0:
            existing = connection.execute(
                "SELECT e.label, e.split, u.source_row, u.raw_url, u.normalized_url "
                "FROM encoded_rows e JOIN urls u ON u.rowid=e.url_rowid WHERE e.encoded_hash=?",
                (digest,),
            ).fetchone()
            reason = (
                "encoded_label_conflict" if int(existing[0]) != int(label)
                else "encoded_cross_split_collision" if existing[1] != split
                else "encoded_within_split_duplicate"
            )
            counts[reason] += 1
            quarantine(connection, source_row, raw_url, str(label), url, reason)
            if reason != "encoded_within_split_duplicate":
                quarantine(connection, int(existing[2]), str(existing[3]), str(existing[0]), str(existing[4]), reason)
                connection.execute("DELETE FROM encoded_rows WHERE encoded_hash=?", (digest,))
                connection.execute("INSERT OR REPLACE INTO encoded_conflicts VALUES (?, ?)", (digest, reason))
        if counts["processed"] % commit_every == 0:
            connection.commit()
            connection.execute("BEGIN")
            print(json.dumps({"stage": "encoded_audit", "rows": counts["processed"], "seconds": round(time.time()-started, 1)}), flush=True)
    connection.commit()
    counts["retained_encoded_unique"] = connection.execute("SELECT COUNT(*) FROM encoded_rows").fetchone()[0]
    counts["encoded_conflict_groups"] = connection.execute("SELECT COUNT(*) FROM encoded_conflicts").fetchone()[0]
    return dict(counts)


def export(config: dict, vocabulary: dict[str, int], connection: sqlite3.Connection, output: Path) -> dict:
    maximum = int(config["maximum_encoded_length"])
    summary = {}
    query = (
        "SELECT u.normalized_url,u.label,u.hostname,u.source_row FROM encoded_rows e "
        "JOIN urls u ON u.rowid=e.url_rowid WHERE e.split=? ORDER BY u.rowid"
    )
    for split in ("train", "validation", "test"):
        total = connection.execute("SELECT COUNT(*) FROM encoded_rows WHERE split=?", (split,)).fetchone()[0]
        labels = np.lib.format.open_memmap(output / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(total,))
        tokens = np.lib.format.open_memmap(output / f"{split}_tokens.npy", mode="w+", dtype=np.uint16, shape=(total, maximum))
        malicious_total = connection.execute(
            "SELECT COUNT(*) FROM encoded_rows WHERE split=? AND label=1", (split,)
        ).fetchone()[0]
        malicious = np.lib.format.open_memmap(output / f"{split}_malicious_indices.npy", mode="w+", dtype=np.int64, shape=(malicious_total,))
        manifest = output / f"{split}_manifest.csv"
        label_counts = Counter()
        malicious_position = 0
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "url", "label", "canonical_hostname", "source_row"])
            for index, (url, label, hostname, source_row) in enumerate(connection.execute(query, (split,))):
                tokens[index] = encode_url(url, vocabulary, maximum)
                labels[index] = label
                label_counts[int(label)] += 1
                if int(label) == 1:
                    malicious[malicious_position] = index
                    malicious_position += 1
                writer.writerow([index, url, label, hostname, source_row])
        tokens.flush(); labels.flush(); malicious.flush()
        summary[split] = {
            "total": total, "label_0": label_counts[0], "label_1": label_counts[1],
            "unique_hostnames": connection.execute(
                "SELECT COUNT(DISTINCT u.hostname) FROM encoded_rows e JOIN urls u ON u.rowid=e.url_rowid WHERE e.split=?", (split,)
            ).fetchone()[0],
            "manifest_sha256": sha256_file(manifest),
        }
    quarantine_path = output / "quarantine.csv"
    with quarantine_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_row", "raw_url", "raw_label", "normalized_url", "reason"])
        writer.writerows(connection.execute("SELECT source_row,raw_url,raw_label,normalized_url,reason FROM quarantine ORDER BY rowid"))
    return {"splits": summary, "quarantine_sha256": sha256_file(quarantine_path)}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project = load_json(config_path)
    config = project["data"]
    raw_path = resolve_from_config(config_path, config["master_csv"])
    observed_hash = sha256_file(raw_path)
    if observed_hash != config["master_sha256"]:
        raise RuntimeError(f"Master CSV hash mismatch: {observed_hash}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    database_path = output / "master.sqlite"
    if database_path.exists():
        if not args.force:
            raise FileExistsError(f"Prepared data already exist at {database_path}; use verify_data or explicitly pass --force")
        database_path.unlink()
    connection = create_database(database_path)
    v2_path = resolve_from_config(config_path, config["version_2_conflict_database"])
    v2 = open_v2_reference(v2_path)
    started = time.time()
    ingestion = ingest(config, raw_path, connection, v2, args.commit_every, int(project["seed"]))
    if v2 is not None:
        v2.close()
    vocabulary = build_vocabulary(config, connection)
    write_json(output / "vocabulary.json", vocabulary)
    encoded = audit_encoded(config, vocabulary, connection, args.commit_every)
    exported = export(config, vocabulary, connection, output)
    report = {
        "project_version": 3,
        "master_csv": str(raw_path),
        "master_sha256": observed_hash,
        "configuration": config,
        "ingestion": ingestion,
        "encoded_audit": encoded,
        "vocabulary_size": len(vocabulary),
        "splits": exported["splits"],
        "quarantine_sha256": exported["quarantine_sha256"],
        "scheme_removed_from_all_model_inputs": True,
        "scheme_feature_removed": True,
        "version_2_rows_imported": 0,
        "version_2_used_only_for_conflict_quarantine": v2_path.exists(),
        "gan_training_contract": "only phishing indices from the Version 3 train split",
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "preparation_report.json", report)
    connection.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
