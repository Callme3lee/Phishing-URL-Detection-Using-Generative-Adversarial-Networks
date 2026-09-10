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
from .url_data import SPECIAL_TOKENS, encode_url, encoded_digest, normalize_url, split_for_hostname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the immutable Version 2 master dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--commit-every", type=int, default=25000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing prepared dataset. Without this flag, prepared data are immutable.",
    )
    return parser.parse_args()


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=TRUNCATE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS urls (
            normalized_url TEXT PRIMARY KEY,
            label INTEGER NOT NULL,
            hostname TEXT NOT NULL,
            split TEXT NOT NULL,
            source_row INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS url_conflicts (
            normalized_url TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS encoded_rows (
            encoded_hash BLOB PRIMARY KEY,
            url_rowid INTEGER NOT NULL UNIQUE,
            label INTEGER NOT NULL,
            split TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS encoded_conflicts (
            encoded_hash BLOB PRIMARY KEY,
            reason TEXT NOT NULL
        );
    """)
    return connection


def ingest(
    config: dict,
    raw_path: Path,
    connection: sqlite3.Connection,
    commit_every: int,
    split_seed: int,
) -> dict:
    mapping = {str(key): int(value) for key, value in config["label_mapping"].items()}
    fractions = config["split_fractions"]
    counts = Counter()
    started = time.time()
    with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {config["url_column"], config["label_column"]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required)}")
        connection.execute("BEGIN")
        for source_row, row in enumerate(reader, start=2):
            counts["input_rows"] += 1
            raw_label = str(row[config["label_column"]]).strip()
            if raw_label not in mapping:
                counts["unknown_label"] += 1
                continue
            try:
                normalized, hostname = normalize_url(
                    row[config["url_column"]], config["maximum_raw_url_length"]
                )
            except ValueError as error:
                counts[f"rejected::{error}"] += 1
                continue
            if connection.execute(
                "SELECT 1 FROM url_conflicts WHERE normalized_url=?", (normalized,)
            ).fetchone():
                counts["rows_after_known_conflict"] += 1
                continue
            label = mapping[raw_label]
            split = split_for_hostname(
                hostname, split_seed, fractions["train"], fractions["validation"]
            )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO urls VALUES (?, ?, ?, ?, ?)",
                (normalized, label, hostname, split, source_row),
            )
            if cursor.rowcount == 1:
                counts["inserted_unique"] += 1
            else:
                existing = connection.execute(
                    "SELECT label FROM urls WHERE normalized_url=?", (normalized,)
                ).fetchone()
                if existing and int(existing[0]) == label:
                    counts["duplicate_same_label"] += 1
                else:
                    connection.execute("DELETE FROM urls WHERE normalized_url=?", (normalized,))
                    connection.execute(
                        "INSERT OR IGNORE INTO url_conflicts VALUES (?)", (normalized,)
                    )
                    counts["duplicate_conflicting_label"] += 1
            if counts["input_rows"] % commit_every == 0:
                connection.commit(); connection.execute("BEGIN")
                elapsed = time.time() - started
                print(json.dumps({"stage":"ingest","rows":counts["input_rows"],"seconds":round(elapsed,1)}), flush=True)
        connection.commit()
    counts["retained_after_exact_deduplication"] = connection.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
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


def audit_encoded(config: dict, vocabulary: dict[str, int], connection: sqlite3.Connection, commit_every: int) -> dict:
    maximum = int(config["maximum_encoded_length"])
    counts = Counter()
    started = time.time()
    connection.execute("DELETE FROM encoded_rows")
    connection.execute("DELETE FROM encoded_conflicts")
    connection.commit(); connection.execute("BEGIN")
    query = "SELECT rowid, normalized_url, label, split FROM urls ORDER BY rowid"
    for rowid, url, label, split in connection.execute(query):
        counts["processed"] += 1
        digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if connection.execute(
            "SELECT 1 FROM encoded_conflicts WHERE encoded_hash=?", (digest,)
        ).fetchone():
            counts["after_conflict"] += 1
            continue
        cursor = connection.execute(
            "INSERT OR IGNORE INTO encoded_rows VALUES (?, ?, ?, ?)",
            (digest, rowid, label, split),
        )
        if cursor.rowcount == 0:
            existing = connection.execute(
                "SELECT label, split FROM encoded_rows WHERE encoded_hash=?", (digest,)
            ).fetchone()
            reason = (
                "label_conflict" if int(existing[0]) != int(label)
                else "cross_split_collision" if existing[1] != split
                else "within_split_duplicate"
            )
            counts[reason] += 1
            if reason in {"label_conflict", "cross_split_collision"}:
                connection.execute("DELETE FROM encoded_rows WHERE encoded_hash=?", (digest,))
                connection.execute(
                    "INSERT OR REPLACE INTO encoded_conflicts VALUES (?, ?)", (digest, reason)
                )
        if counts["processed"] % commit_every == 0:
            connection.commit(); connection.execute("BEGIN")
            print(json.dumps({"stage":"encoded_audit","rows":counts["processed"],"seconds":round(time.time()-started,1)}), flush=True)
    connection.commit()
    counts["retained_encoded_unique"] = connection.execute("SELECT COUNT(*) FROM encoded_rows").fetchone()[0]
    counts["encoded_conflict_groups"] = connection.execute("SELECT COUNT(*) FROM encoded_conflicts").fetchone()[0]
    return dict(counts)


def export(config: dict, vocabulary: dict[str, int], connection: sqlite3.Connection, output: Path) -> dict:
    maximum = int(config["maximum_encoded_length"])
    summary = {}
    query_base = """
        SELECT u.normalized_url, u.label, u.hostname, u.source_row
        FROM encoded_rows e JOIN urls u ON u.rowid=e.url_rowid
        WHERE e.split=? ORDER BY u.rowid
    """
    for split in ("train", "validation", "test"):
        total = connection.execute(
            "SELECT COUNT(*) FROM encoded_rows WHERE split=?", (split,)
        ).fetchone()[0]
        labels = np.lib.format.open_memmap(
            output / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
        )
        tokens = np.lib.format.open_memmap(
            output / f"{split}_tokens.npy", mode="w+", dtype=np.uint16, shape=(total, maximum)
        )
        malicious_total = connection.execute(
            "SELECT COUNT(*) FROM encoded_rows WHERE split=? AND label=1", (split,)
        ).fetchone()[0]
        malicious_indices = np.lib.format.open_memmap(
            output / f"{split}_malicious_indices.npy", mode="w+", dtype=np.int64,
            shape=(malicious_total,),
        )
        manifest_path = output / f"{split}_manifest.csv"
        label_counts = Counter(); malicious_position = 0
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "url", "label", "canonical_hostname", "source_row"])
            for index, (url, label, hostname, source_row) in enumerate(
                connection.execute(query_base, (split,))
            ):
                tokens[index] = encode_url(url, vocabulary, maximum)
                labels[index] = label
                label_counts[int(label)] += 1
                if int(label) == 1:
                    malicious_indices[malicious_position] = index
                    malicious_position += 1
                writer.writerow([index, url, label, hostname, source_row])
                if index and index % 250000 == 0:
                    print(json.dumps({"stage":"export","split":split,"rows":index}), flush=True)
        tokens.flush(); labels.flush(); malicious_indices.flush()
        summary[split] = {
            "total": total,
            "label_0": label_counts[0],
            "label_1": label_counts[1],
            "malicious_index_count": malicious_total,
            "manifest_sha256": sha256_file(manifest_path),
        }
    return summary


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
            raise FileExistsError(
                f"Prepared dataset already exists at {database_path}. "
                "Run verify_data instead, or pass --force to rebuild it deliberately."
            )
        database_path.unlink()
    connection = database(database_path)
    started = time.time()
    ingestion = ingest(config, raw_path, connection, args.commit_every, int(project["seed"]))
    vocabulary = build_vocabulary(config, connection)
    write_json(output / "vocabulary.json", vocabulary)
    encoded = audit_encoded(config, vocabulary, connection, args.commit_every)
    splits = export(config, vocabulary, connection, output)
    report = {
        "project_version": 2,
        "master_csv": str(raw_path),
        "master_sha256": observed_hash,
        "split_seed": int(project["seed"]),
        "configuration": config,
        "ingestion": ingestion,
        "encoded_audit": encoded,
        "vocabulary_size": len(vocabulary),
        "splits": splits,
        "gan_training_contract": "train_malicious_indices only",
        "external_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "preparation_report.json", report)
    connection.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
