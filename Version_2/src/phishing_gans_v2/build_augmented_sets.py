from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json
from .url_data import LEXICAL_FEATURE_NAMES, encode_url, lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested, immutable augmentation conditions.")
    parser.add_argument("--model", choices=("gumbel_lsgan", "seqgan", "relgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--generated-root", type=Path, default=Path("artifacts/generated"))
    parser.add_argument("--output-root", type=Path, default=Path("data/augmented"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    report = load_json(args.data_dir / "preparation_report.json")
    vocabulary = load_json(args.data_dir / "vocabulary.json")
    accepted_path = args.generated_root / args.model / "accepted.csv"
    generator_metrics = load_json(args.generated_root / args.model / "metrics.json")
    if generator_metrics["master_sha256"] != project["data"]["master_sha256"]:
        raise RuntimeError("Generated corpus belongs to a different master dataset")
    real_count = int(report["splits"]["train"]["total"])
    fractions = [float(project["augmentation"]["primary_generated_fraction"])]
    fractions.extend(float(value) for value in project["augmentation"]["exploratory_generated_fractions"])
    fractions = sorted(set(fractions))
    required = {fraction: round(real_count * fraction / (1.0 - fraction)) for fraction in fractions}
    maximum = max(required.values())
    available = int(generator_metrics["accepted_count"])
    if available < maximum:
        raise RuntimeError(f"Need {maximum:,} accepted unique URLs for 40% but only have {available:,}")

    selected: list[tuple[str, str]] = []
    with accepted_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            selected.append((row["url"], row["hostname"]))
            if len(selected) == maximum:
                break
    rng = np.random.default_rng(int(project["seed"]))
    order = rng.permutation(maximum); selected = [selected[int(index)] for index in order]
    output = args.output_root / args.model; output.mkdir(parents=True, exist_ok=True)
    pool_path = output / "generated_pool_tokens.npy"
    tokens = np.lib.format.open_memmap(
        pool_path, mode="w+", dtype=np.uint16,
        shape=(maximum, int(project["data"]["maximum_encoded_length"])),
    )
    feature_path = output / "generated_pool_features.npy"
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32,
        shape=(maximum, len(LEXICAL_FEATURE_NAMES)),
    )
    pool_manifest = output / "generated_pool_manifest.csv"
    with pool_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("pool_index", "url", "label", "generator", "hostname"))
        for index, (url, hostname) in enumerate(selected):
            tokens[index] = encode_url(url, vocabulary, tokens.shape[1])
            features[index] = lexical_features(url)
            writer.writerow((index, url, 1, args.model, hostname))
            if index and index % 250000 == 0:
                print(json.dumps({"model": args.model, "encoded_generated": index}), flush=True)
    tokens.flush(); features.flush()
    conditions = {}
    for fraction, generated_count in required.items():
        name = f"generated_{round(100 * fraction):02d}_percent"
        condition = {
            "project_version": 2, "condition": name, "generator": args.model,
            "real_train_rows": real_count, "generated_rows": generated_count,
            "final_training_rows": real_count + generated_count,
            "generated_fraction": generated_count / (real_count + generated_count),
            "generated_pool_slice": [0, generated_count],
            "real_tokens": str(args.data_dir / "train_tokens.npy"),
            "real_labels": str(args.data_dir / "train_labels.npy"),
            "generated_tokens": str(pool_path),
            "generated_features": str(feature_path),
            "generated_label": 1,
            "master_sha256": project["data"]["master_sha256"],
            "accepted_csv_sha256": generator_metrics["accepted_csv_sha256"],
        }
        write_json(output / f"{name}.json", condition); conditions[name] = condition
    index = {
        "project_version": 2, "generator": args.model, "conditions": conditions,
        "nested_design": True,
        "fraction_definition": project["augmentation"]["fraction_definition"],
        "pool_tokens_sha256": sha256_file(pool_path),
        "pool_features_sha256": sha256_file(feature_path),
        "pool_manifest_sha256": sha256_file(pool_manifest),
    }
    write_json(output / "index.json", index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
