from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .common import load_json, sha256_file, write_json


STRUCTURES = ("root", "shallow", "deep", "query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a URLs_New-only structure-balanced training condition.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--base-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-features", type=Path, default=Path("data/features"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/experiments/urls_new_structure_balanced_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legitimate-per-structure", type=int, default=29882)
    return parser.parse_args()


def structure(url: str) -> str:
    parsed = urlsplit("http://" + url.lstrip("/"))
    if parsed.query:
        return "query"
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if not segments:
        return "root"
    if len(segments) == 1:
        return "shallow"
    return "deep"


def stable_key(seed: int, url: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{url}".encode("utf-8"), digest_size=16).digest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    processed = args.output_root / "processed"
    features = args.output_root / "features"
    report_path = args.output_root / "experiment_report.json"
    if report_path.exists():
        raise FileExistsError(f"Structure-balanced experiment is already frozen at {args.output_root}")
    project = load_json(args.config)
    base_manifest = load_csv(args.base_data / "train_manifest.csv")
    base_labels = np.load(args.base_data / "train_labels.npy", mmap_mode="r")
    if len(base_manifest) != len(base_labels):
        raise RuntimeError("Base training manifest and labels differ in length")

    legitimate: dict[str, list[tuple[bytes, int]]] = {name: [] for name in STRUCTURES}
    phishing_indices: list[int] = []
    base_structure_counts = Counter()
    for index, row in enumerate(base_manifest):
        category = structure(row["url"])
        label = int(row["label"])
        base_structure_counts[(label, category)] += 1
        if label == 1:
            phishing_indices.append(index)
        else:
            legitimate[category].append((stable_key(args.seed, row["url"]), index))
    for category in STRUCTURES:
        if len(legitimate[category]) < args.legitimate_per_structure:
            raise RuntimeError(
                f"Only {len(legitimate[category])} legitimate {category} rows; "
                f"requested {args.legitimate_per_structure}"
            )
    selected_legitimate = []
    selected_structure_counts = Counter()
    for category in STRUCTURES:
        chosen = sorted(legitimate[category])[:args.legitimate_per_structure]
        selected_legitimate.extend(index for _, index in chosen)
        selected_structure_counts[category] = len(chosen)
    selected_indices = sorted(selected_legitimate + phishing_indices)
    selected_set = set(selected_indices)
    if len(selected_set) != len(selected_indices):
        raise RuntimeError("Selection unexpectedly contains duplicate indices")

    processed.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    base_tokens = np.load(args.base_data / "train_tokens.npy", mmap_mode="r")
    base_features = np.load(args.base_features / "train_features.npy", mmap_mode="r")
    train_tokens = np.lib.format.open_memmap(
        processed / "train_tokens.npy", mode="w+", dtype=base_tokens.dtype,
        shape=(len(selected_indices), base_tokens.shape[1]),
    )
    train_labels = np.lib.format.open_memmap(
        processed / "train_labels.npy", mode="w+", dtype=base_labels.dtype,
        shape=(len(selected_indices),),
    )
    train_features = np.lib.format.open_memmap(
        features / "train_features.npy", mode="w+", dtype=base_features.dtype,
        shape=(len(selected_indices), base_features.shape[1]),
    )
    chunk_size = 25000
    index_array = np.asarray(selected_indices, dtype=np.int64)
    for start in range(0, len(index_array), chunk_size):
        stop = min(len(index_array), start + chunk_size)
        source = index_array[start:stop]
        train_tokens[start:stop] = base_tokens[source]
        train_labels[start:stop] = base_labels[source]
        train_features[start:stop] = base_features[source]
    train_tokens.flush()
    train_labels.flush()
    train_features.flush()

    manifest_rows = []
    for new_index, old_index in enumerate(selected_indices):
        row = dict(base_manifest[old_index])
        row["row_index"] = new_index
        manifest_rows.append(row)
    write_csv(processed / "train_manifest.csv", manifest_rows, list(base_manifest[0].keys()))
    for split in ("validation", "test"):
        for suffix in ("tokens.npy", "labels.npy", "manifest.csv"):
            shutil.copy2(args.base_data / f"{split}_{suffix}", processed / f"{split}_{suffix}")
        shutil.copy2(args.base_features / f"{split}_features.npy", features / f"{split}_features.npy")
    shutil.copy2(args.base_data / "vocabulary.json", processed / "vocabulary.json")
    shutil.copy2(args.base_features / "metadata.json", features / "metadata.json")

    split_report = {}
    for split in ("train", "validation", "test"):
        labels = np.load(processed / f"{split}_labels.npy", mmap_mode="r")
        split_report[split] = {
            "total": len(labels), "label_0": int((labels == 0).sum()), "label_1": int((labels == 1).sum()),
            "manifest_sha256": sha256_file(processed / f"{split}_manifest.csv"),
            "tokens_sha256": sha256_file(processed / f"{split}_tokens.npy"),
            "labels_sha256": sha256_file(processed / f"{split}_labels.npy"),
            "features_sha256": sha256_file(features / f"{split}_features.npy"),
        }
    base_report = load_json(args.base_data / "preparation_report.json")
    write_json(processed / "preparation_report.json", {
        **base_report,
        "purpose": "URLs_New-only legitimate-structure-balanced training experiment",
        "splits": split_report,
    })
    report = {
        "project_version": 3,
        "condition": "urls_new_structure_balanced_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_contract": "Every row comes from the original frozen URLs_New.csv-derived Version 3 split; outside rows added: 0",
        "source_master_sha256": project["data"]["master_sha256"],
        "selection_seed": args.seed,
        "selection_method": "lowest BLAKE2b-128(seed:url) values within each legitimate structure; all phishing rows retained",
        "base_train_rows": len(base_labels),
        "final_train_rows": len(selected_indices),
        "rows_removed": len(base_labels) - len(selected_indices),
        "final_label_counts": {"legitimate": len(selected_legitimate), "phishing": len(phishing_indices)},
        "base_structure_counts": {
            label: {category: base_structure_counts[(numeric, category)] for category in STRUCTURES}
            for numeric, label in ((0, "legitimate"), (1, "phishing"))
        },
        "selected_legitimate_structure_counts": dict(selected_structure_counts),
        "contracts": {
            "outside_training_rows": 0,
            "duplicate_or_repeated_rows": 0,
            "all_original_phishing_training_rows_retained": len(phishing_indices) == int((base_labels == 1).sum()),
            "original_validation_unchanged": split_report["validation"]["tokens_sha256"] == sha256_file(args.base_data / "validation_tokens.npy"),
            "original_test_unchanged": split_report["test"]["tokens_sha256"] == sha256_file(args.base_data / "test_tokens.npy"),
            "hyperparameters_unchanged": True,
            "threshold_policy_unchanged": project["threshold_policy"],
        },
        "splits": split_report,
        "limitations": [
            "This changes both legitimate structural proportions and the class ratio by deterministic downsampling.",
            "It cannot add domain/source patterns absent from URLs_New.csv.",
            "External sets remain proxy-labelled and balanced rather than deployment-prevalence samples.",
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
