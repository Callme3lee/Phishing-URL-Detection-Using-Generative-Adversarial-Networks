from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, encode_url, psl_aware_lexical_features


CONDITIONS = ("real_only", "gan_10pct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the fair recent-update real-only versus +10% GAN experiment.")
    parser.add_argument("--update", type=Path, default=Path("data/updates/recent_observed_update_2026-08-01_to_2026-08-15_v1/updated_samples.csv"))
    parser.add_argument("--real-base", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--gan-base", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1/gan_10pct/processed"))
    parser.add_argument("--generated", type=Path, default=Path("data/augmentations/ms_hardgan_v2/generated_10pct.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/recent_update_gan10_v1"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def registered(host: str) -> str:
    value = PSL_EXTRACTOR(host.lower())
    return ".".join(part for part in (value.domain, value.suffix) if part) or host.lower()


def link(source: Path, target: Path) -> None:
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Recent-update experiment already frozen at {args.output_root}")
    update_rows = read_csv(args.update); update_domains = {row["registered_domain"] for row in update_rows}
    vocabulary = load_json(args.real_base / "vocabulary.json")
    scaler = np.load(args.real_base / "psl_lexical_scaler.npz")
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.where(np.asarray(scaler["scale"], dtype=np.float32) == 0, 1.0,
                     np.asarray(scaler["scale"], dtype=np.float32))
    update_tokens = np.stack([encode_url(row["url"], vocabulary, 128) for row in update_rows]).astype(np.uint16)
    update_features = np.stack([(psl_aware_lexical_features(row["url"]) - center) / scale for row in update_rows]).astype(np.float32)
    update_labels = np.asarray([int(row["label"]) for row in update_rows], dtype=np.uint8)
    update_groups = np.where(update_labels == 0, 1, 3).astype(np.uint8)
    base_manifest = read_csv(args.real_base / "train_manifest.csv")
    generated_rows = read_csv(args.generated)
    reports = []

    for condition in CONDITIONS:
        source = args.real_base if condition == "real_only" else args.gan_base
        processed = args.output_root / condition / "processed"; processed.mkdir(parents=True)
        base_tokens = np.load(source / "train_tokens.npy", mmap_mode="r")
        base_labels = np.load(source / "train_labels.npy", mmap_mode="r")
        base_groups = np.load(source / "train_groups.npy", mmap_mode="r")
        base_features = np.load(source / "train_psl_lexical.npy", mmap_mode="r")
        total = len(base_labels) + len(update_rows)
        tokens = np.lib.format.open_memmap(processed / "train_tokens.npy", mode="w+", dtype=np.uint16, shape=(total, 128))
        labels = np.lib.format.open_memmap(processed / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,))
        groups = np.lib.format.open_memmap(processed / "train_groups.npy", mode="w+", dtype=np.uint8, shape=(total,))
        features = np.lib.format.open_memmap(processed / "train_psl_lexical.npy", mode="w+", dtype=np.float32, shape=(total, base_features.shape[1]))
        tokens[:len(base_tokens)] = base_tokens; tokens[len(base_tokens):] = update_tokens
        labels[:len(base_labels)] = base_labels; labels[len(base_labels):] = update_labels
        groups[:len(base_groups)] = base_groups; groups[len(base_groups):] = update_groups
        features[:len(base_features)] = base_features; features[len(base_features):] = update_features
        tokens.flush(); labels.flush(); groups.flush(); features.flush()
        link(processed / "train_psl_lexical.npy", processed / "train_features.npy")

        manifest = list(base_manifest)
        if condition == "gan_10pct":
            for index, row in enumerate(generated_rows, start=len(manifest)):
                manifest.append({"row_index": index, "url": row["url"], "label": 1,
                                 "canonical_hostname": row["hostname"], "source_group": "synthetic_phishing",
                                 "source_id": row["raw_sha256"], "source_timestamp": "", "raw_url": row["url"]})
        for index, row in enumerate(update_rows, start=len(manifest)):
            manifest.append({"row_index": index, "url": row["url"], "label": row["label"],
                             "canonical_hostname": row["canonical_hostname"],
                             "source_group": "recent_update_legitimate" if row["label"] == "0" else "recent_update_phishing",
                             "source_id": row["source_id"], "source_timestamp": row["source_record_time"],
                             "raw_url": row["raw_url"]})
        fields = ["row_index", "url", "label", "canonical_hostname", "source_group", "source_id", "source_timestamp", "raw_url"]
        write_csv(processed / "train_manifest.csv", manifest, fields)

        removed = {}
        for split in ("validation", "test"):
            split_manifest = read_csv(args.real_base / f"{split}_manifest.csv")
            keep = np.asarray([registered(row["canonical_hostname"]) not in update_domains for row in split_manifest])
            kept_rows = [{**row, "row_index": index} for index, row in enumerate(row for row, selected in zip(split_manifest, keep) if selected)]
            removed[split] = len(split_manifest) - len(kept_rows)
            for suffix in ("tokens", "labels", "psl_lexical"):
                values = np.load(args.real_base / f"{split}_{suffix}.npy", mmap_mode="r")
                np.save(processed / f"{split}_{suffix}.npy", np.asarray(values[keep]))
            write_csv(processed / f"{split}_manifest.csv", kept_rows, list(split_manifest[0].keys()))
            link(processed / f"{split}_psl_lexical.npy", processed / f"{split}_features.npy")
        for name in ("domain_validation_tokens.npy", "domain_validation_labels.npy", "domain_validation_groups.npy",
                     "domain_validation_psl_lexical.npy", "domain_validation_manifest.csv", "vocabulary.json",
                     "psl_lexical_metadata.json", "psl_lexical_scaler.npz"):
            link(args.real_base / name, processed / name)
        synthetic = int((groups == 5).sum()); real = total - synthetic
        report = {
            "project_version": 3, "condition": condition, "experiment": "recent_update_gan10_v1",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_real_training_rows": len(base_manifest), "recent_update_rows": len(update_rows),
            "recent_update_counts": {"legitimate": int((update_labels == 0).sum()), "phishing": int((update_labels == 1).sum())},
            "real_training_rows": real, "synthetic_training_rows": synthetic, "training_rows": total,
            "generated_fraction": 0.0 if condition == "real_only" else 0.10,
            "stored_generated_fraction": synthetic / total,
            "generated_exposure_note": "weighted sampler targets 10% synthetic exposure" if synthetic else "no synthetic rows",
            "same_real_rows_across_conditions": True, "same_update_rows_across_conditions": True,
            "update_sha256": sha256_file(args.update), "update_rows_used": len(update_rows),
            "protected_confirmation_rows_used": 0,
            "validation_test_overlap_rows_removed": removed,
            "domain_validation_overlap_with_update": 0,
            "evaluation_scope": "filtered original internal validation/test only; all earlier outside/calibration cohorts containing update identities are retired",
            "train_group_counts": dict(Counter(int(value) for value in groups)),
            "files": {name: sha256_file(processed / name) for name in
                      ("train_tokens.npy", "train_labels.npy", "train_groups.npy", "train_psl_lexical.npy",
                       "train_manifest.csv", "validation_tokens.npy", "validation_labels.npy", "test_tokens.npy", "test_labels.npy")},
        }
        write_json(args.output_root / condition / "experiment_report.json", report); reports.append(report)
    write_json(args.output_root / "index.json", {
        "experiment": "recent_update_gan10_v1", "conditions": reports,
        "comparison_contract": "same architecture, seeds, original real rows, recent update, preprocessing, weighted sampling family, validation/test filtering, and metrics; only GAN synthetic rows differ",
    })
    print({"experiment": "recent_update_gan10_v1", "conditions": [(r["condition"], r["training_rows"]) for r in reports]})


if __name__ == "__main__":
    main()
