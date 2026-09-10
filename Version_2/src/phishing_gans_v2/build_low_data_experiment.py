from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a hostname-grouped low-data classifier experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/train_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/low_data/real_10_percent"))
    parser.add_argument("--real-fraction", type=float, default=.10)
    parser.add_argument("--generated-fractions", default=".01,.02,.05,.10,.15")
    parser.add_argument("--generator", choices=("seqgan", "gumbel_lsgan"), default="seqgan")
    parser.add_argument("--synthetic-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def selected_hostname(hostname: str, seed: int, fraction: float) -> bool:
    digest = hashlib.blake2b(f"low-data-v2:{seed}:{hostname}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    if not 0 < args.real_fraction < 1: raise ValueError("--real-fraction must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = []; counts = Counter(); selected_hosts = set(); all_hosts = set()
    subset_manifest = args.output_dir / "train_subset_manifest.csv"
    with args.manifest.open("r", encoding="utf-8", newline="") as source, subset_manifest.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source); writer = csv.DictWriter(target, fieldnames=reader.fieldnames); writer.writeheader()
        for row in reader:
            hostname = row["canonical_hostname"]; all_hosts.add(hostname); counts["source_rows"] += 1
            counts[f"source_label_{row['label']}"] += 1
            if not selected_hostname(hostname, args.seed, args.real_fraction): continue
            index = int(row["row_index"]); indices.append(index); selected_hosts.add(hostname); writer.writerow(row)
            counts["selected_rows"] += 1; counts[f"selected_label_{row['label']}"] += 1
    indices_array = np.asarray(indices, dtype=np.int64); indices_path = args.output_dir / "train_indices.npy"
    np.save(indices_path, indices_array)
    labels = np.load("data/processed/train_labels.npy", mmap_mode="r")
    if not np.array_equal(np.bincount(np.asarray(labels[indices_array]), minlength=2), [counts["selected_label_0"], counts["selected_label_1"]]):
        raise RuntimeError("Subset manifest labels do not match frozen training labels")
    fractions = sorted({float(value) for value in args.generated_fractions.split(",") if value.strip()})
    generator_root = Path("data/augmented") / args.generator
    generator_index = load_json(generator_root / "index.json")
    pool_tokens = generator_index["conditions"]["generated_10_percent"]["generated_tokens"]
    pool_features = generator_index["conditions"]["generated_10_percent"]["generated_features"]
    conditions = {}
    for fraction in fractions:
        generated = round(len(indices_array) * fraction / (1 - fraction)); name = f"generated_{round(100*fraction):02d}_percent"
        payload = {
            "project_version": 2, "experiment": "real_10_percent", "condition": name,
            "generator": args.generator, "real_indices": str(indices_path), "real_train_rows": len(indices_array),
            "generated_rows": generated, "final_training_rows": len(indices_array) + generated,
            "generated_fraction": generated / (len(indices_array) + generated),
            "generated_tokens": pool_tokens, "generated_features": pool_features,
            "synthetic_weight": args.synthetic_weight, "class_balance_source": "frozen real subset only",
            "master_sha256": project["data"]["master_sha256"],
        }
        path = args.output_dir / f"{args.generator}_{name}.json"; write_json(path, payload); conditions[name] = {**payload, "manifest": str(path)}
    report = {
        "project_version": 2, "experiment": "real_10_percent", "selection_unit": "canonical_hostname",
        "selection_method": "seeded BLAKE2b hostname inclusion", "seed": args.seed,
        "requested_real_fraction": args.real_fraction, "observed_real_fraction": len(indices_array) / counts["source_rows"],
        "source_rows": counts["source_rows"], "selected_rows": len(indices_array),
        "source_label_0": counts["source_label_0"], "source_label_1": counts["source_label_1"],
        "selected_label_0": counts["selected_label_0"], "selected_label_1": counts["selected_label_1"],
        "source_hostnames": len(all_hosts), "selected_hostnames": len(selected_hosts),
        "indices": str(indices_path), "indices_sha256": sha256_file(indices_path),
        "subset_manifest": str(subset_manifest), "subset_manifest_sha256": sha256_file(subset_manifest),
        "synthetic_weight": args.synthetic_weight, "conditions": conditions,
        "validation_and_test": "unchanged full frozen Version 2 splits",
    }
    write_json(args.output_dir / "experiment.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
