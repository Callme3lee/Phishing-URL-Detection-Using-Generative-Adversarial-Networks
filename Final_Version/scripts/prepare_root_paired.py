#!/usr/bin/env python3
"""Build the isolated, label-symmetric root representation training set."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
V3_SOURCE = REPOSITORY / "Version_3" / "src"
sys.path.insert(0, str(V3_SOURCE))

from phishing_gans_v3.common import load_json, sha256_file, write_json  # noqa: E402
from phishing_gans_v3.url_data import encode_url, psl_aware_lexical_features  # noqa: E402


SOURCE = REPOSITORY / "Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/processed"
SOURCE_REPORT = REPOSITORY / "Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/experiment_report.json"
OUTPUT_ROOT = REPOSITORY / "Final_Version/data/root_paired_hyb_ms_gan10_v1"
OUTPUT = OUTPUT_ROOT / "gan_10pct/processed"


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def root_counterpart(url: str) -> str | None:
    """Return the other harmless root representation, or None for non-roots."""
    value = str(url).strip()
    try:
        parsed = urlsplit("http://" + value.lstrip("/"))
    except ValueError:
        return None
    if not parsed.hostname or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        return None
    other_path = "/" if parsed.path == "" else ""
    rebuilt = urlunsplit(("http", parsed.netloc, other_path, "", ""))
    return rebuilt[len("http://") :]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Frozen output already exists: {OUTPUT_ROOT}")

    source_report = load_json(SOURCE_REPORT)
    manifest = read_manifest(SOURCE / "train_manifest.csv")
    labels = np.load(SOURCE / "train_labels.npy", mmap_mode="r")
    groups = np.load(SOURCE / "train_groups.npy", mmap_mode="r")
    tokens = np.load(SOURCE / "train_tokens.npy", mmap_mode="r")
    lexical = np.load(SOURCE / "train_psl_lexical.npy", mmap_mode="r")
    if not (len(manifest) == len(labels) == len(groups) == len(tokens) == len(lexical)):
        raise RuntimeError("Source training arrays and manifest are not aligned")

    vocabulary = load_json(SOURCE / "vocabulary.json")
    scaler = np.load(SOURCE / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.where(np.asarray(scaler["scale"], dtype=np.float32) == 0, 1.0,
                     np.asarray(scaler["scale"], dtype=np.float32))

    paired: list[tuple[int, str]] = []
    base_root_counts: Counter[tuple[int, int, str]] = Counter()
    for index, row in enumerate(manifest):
        counterpart = root_counterpart(row["url"])
        if counterpart is None:
            continue
        representation = "slash" if row["url"].endswith("/") else "empty"
        base_root_counts[(int(labels[index]), int(groups[index]), representation)] += 1
        paired.append((index, counterpart))

    OUTPUT.mkdir(parents=True)
    total = len(manifest) + len(paired)
    out_tokens = np.lib.format.open_memmap(OUTPUT / "train_tokens.npy", mode="w+", dtype=np.uint16,
                                           shape=(total, tokens.shape[1]))
    out_labels = np.lib.format.open_memmap(OUTPUT / "train_labels.npy", mode="w+", dtype=np.uint8,
                                           shape=(total,))
    out_groups = np.lib.format.open_memmap(OUTPUT / "train_groups.npy", mode="w+", dtype=np.uint8,
                                           shape=(total,))
    out_lexical = np.lib.format.open_memmap(OUTPUT / "train_psl_lexical.npy", mode="w+", dtype=np.float32,
                                            shape=(total, lexical.shape[1]))
    out_tokens[: len(tokens)] = tokens
    out_labels[: len(labels)] = labels
    out_groups[: len(groups)] = groups
    out_lexical[: len(lexical)] = lexical

    augmented_manifest = list(manifest)
    added_counts: Counter[tuple[int, int, str]] = Counter()
    for offset, (source_index, counterpart) in enumerate(paired, start=len(manifest)):
        label = int(labels[source_index])
        group = int(groups[source_index])
        out_tokens[offset] = encode_url(counterpart, vocabulary, tokens.shape[1])
        out_labels[offset] = label
        out_groups[offset] = group
        out_lexical[offset] = (psl_aware_lexical_features(counterpart) - center) / scale
        source_row = manifest[source_index]
        representation = "slash" if counterpart.endswith("/") else "empty"
        added_counts[(label, group, representation)] += 1
        augmented_manifest.append({
            **source_row,
            "row_index": str(offset),
            "url": counterpart,
            "raw_url": counterpart,
            "source_id": f"{source_row.get('source_id', source_index)}:root-pair:{representation}",
        })

    out_tokens.flush(); out_labels.flush(); out_groups.flush(); out_lexical.flush()
    hardlink_or_copy(OUTPUT / "train_psl_lexical.npy", OUTPUT / "train_features.npy")

    fields = list(manifest[0].keys())
    with (OUTPUT / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(augmented_manifest)

    untouched = [
        "validation_tokens.npy", "validation_labels.npy", "validation_psl_lexical.npy",
        "validation_manifest.csv", "test_tokens.npy", "test_labels.npy", "test_psl_lexical.npy",
        "test_manifest.csv", "domain_validation_tokens.npy", "domain_validation_labels.npy",
        "domain_validation_groups.npy", "domain_validation_psl_lexical.npy",
        "domain_validation_manifest.csv", "vocabulary.json", "psl_lexical_metadata.json",
        "psl_lexical_scaler.npz",
    ]
    for name in untouched:
        hardlink_or_copy(SOURCE / name, OUTPUT / name)
    hardlink_or_copy(OUTPUT / "validation_psl_lexical.npy", OUTPUT / "validation_features.npy")
    hardlink_or_copy(OUTPUT / "test_psl_lexical.npy", OUTPUT / "test_features.npy")

    final_root_counts: Counter[tuple[int, int, str]] = Counter()
    for row, label, group in zip(augmented_manifest, out_labels, out_groups):
        if root_counterpart(row["url"]) is not None:
            final_root_counts[(int(label), int(group), "slash" if row["url"].endswith("/") else "empty")] += 1
    for label, group in {(key[0], key[1]) for key in final_root_counts}:
        if final_root_counts[(label, group, "empty")] != final_root_counts[(label, group, "slash")]:
            raise RuntimeError(f"Unbalanced paired roots for label={label}, group={group}")

    synthetic = int(np.sum(np.asarray(out_groups) == 5))
    report = {
        "project_version": "Final Version",
        "experiment": "root_paired_hyb_ms_gan10_v1",
        "condition": "gan_10pct",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": "Version_3/recent_update_gan10_v1/gan_10pct",
        "source_report_sha256": sha256_file(SOURCE_REPORT),
        "source_training_rows": len(manifest),
        "source_real_training_rows": int(source_report["real_training_rows"]),
        "source_synthetic_training_rows": int(source_report["synthetic_training_rows"]),
        "root_rows_paired": len(paired),
        "added_equivalent_rows": len(paired),
        "training_rows": total,
        "real_training_rows": total - synthetic,
        "synthetic_training_rows": synthetic,
        "generated_fraction": 0.10,
        "stored_generated_fraction": synthetic / total,
        "sampling_note": "weighted sampler still targets exactly 10% synthetic exposure",
        "treatment": "each root-only source row gains the opposite empty-path/root-slash representation with the same label and source group",
        "non_root_rows_changed": 0,
        "validation_test_changed": False,
        "base_root_counts": {f"label_{l}_group_{g}_{r}": n for (l, g, r), n in sorted(base_root_counts.items())},
        "added_root_counts": {f"label_{l}_group_{g}_{r}": n for (l, g, r), n in sorted(added_counts.items())},
        "final_root_counts": {f"label_{l}_group_{g}_{r}": n for (l, g, r), n in sorted(final_root_counts.items())},
        "train_group_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(out_groups, return_counts=True))},
        "source_files": {name: sha256_file(SOURCE / name) for name in (
            "train_tokens.npy", "train_labels.npy", "train_groups.npy", "train_psl_lexical.npy", "train_manifest.csv"
        )},
        "files": {name: sha256_file(OUTPUT / name) for name in (
            "train_tokens.npy", "train_labels.npy", "train_groups.npy", "train_psl_lexical.npy", "train_manifest.csv",
            "validation_tokens.npy", "validation_labels.npy", "test_tokens.npy", "test_labels.npy"
        )},
    }
    write_json(OUTPUT_ROOT / "gan_10pct/experiment_report.json", report)
    write_json(OUTPUT_ROOT / "index.json", {
        "project_version": "Final Version",
        "experiment": report["experiment"],
        "condition": report,
        "comparison_control": "unchanged Version 3 browser-extension HYB-MS +10% checkpoints",
        "controlled_difference": "root-only paired representation augmentation during training",
    })
    print(json.dumps({
        "source_rows": len(manifest), "root_rows_paired": len(paired), "final_rows": total,
        "real_rows": total - synthetic, "synthetic_rows": synthetic,
        "stored_synthetic_fraction": synthetic / total,
    }, indent=2))


if __name__ == "__main__":
    main()
