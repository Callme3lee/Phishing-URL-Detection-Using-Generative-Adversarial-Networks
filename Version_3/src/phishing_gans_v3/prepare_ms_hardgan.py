from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from .common import load_json, sha256_file, write_json


EXPECTED_SOURCES = ("urls_new_phishing", "phishtank_phishing", "openphish_phishing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the multi-source phishing corpus for MS-HardGAN.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/ms_hardgan_v1"))
    parser.add_argument("--patterns", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"Frozen MS-HardGAN data already exists: {args.output_dir}")
    labels = np.load(args.input_dir / "train_labels.npy", mmap_mode="r")
    tokens = np.load(args.input_dir / "train_tokens.npy", mmap_mode="r")
    lexical = np.load(args.input_dir / "train_psl_lexical_raw.npy", mmap_mode="r")
    selected = np.flatnonzero(labels == 1)
    rows: list[dict[str, str]] = []
    with (args.input_dir / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if labels[index] == 1:
                if row["source_group"] not in EXPECTED_SOURCES:
                    raise RuntimeError(f"Unexpected phishing source: {row['source_group']}")
                rows.append(row)
    if len(rows) != len(selected):
        raise RuntimeError("Manifest and label arrays disagree")

    features = np.asarray(lexical[selected], dtype=np.float32)
    scaler = RobustScaler().fit(features)
    clusterer = MiniBatchKMeans(
        n_clusters=args.patterns, random_state=int(project["seed"]),
        batch_size=2048, n_init=20,
    ).fit(scaler.transform(features))
    patterns = clusterer.labels_.astype(np.uint8)
    source_mapping = {name: index for index, name in enumerate(EXPECTED_SOURCES)}
    sources = np.asarray([source_mapping[row["source_group"]] for row in rows], dtype=np.uint8)

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "train_tokens.npy", np.asarray(tokens[selected], dtype=np.uint16))
    np.save(args.output_dir / "train_patterns.npy", patterns)
    np.save(args.output_dir / "train_sources.npy", sources)
    (args.output_dir / "train_urls.txt").write_text(
        "\n".join(row["url"] for row in rows) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "train_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) + ["ms_hardgan_pattern", "ms_hardgan_source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, pattern, source in zip(rows, patterns, sources):
            writer.writerow({**row, "ms_hardgan_pattern": int(pattern), "ms_hardgan_source": int(source)})
    joblib.dump({"scaler": scaler, "clusterer": clusterer}, args.output_dir / "pattern_model.joblib", compress=3)
    vocabulary_target = args.output_dir / "vocabulary.json"
    vocabulary_target.write_text((args.input_dir / "vocabulary.json").read_text(encoding="utf-8"), encoding="utf-8")

    files = ("train_tokens.npy", "train_patterns.npy", "train_sources.npy", "train_urls.txt",
             "train_manifest.csv", "pattern_model.joblib", "vocabulary.json")
    report = {
        "condition": "ms_hardgan_v1", "project_version": 3,
        "purpose": "defensive generation of realistic, diverse, classifier-challenging phishing-like URL strings",
        "input_condition": "hybrid_cnn_coral_v1 exact stored multi-source real training rows",
        "training_rows": len(rows), "phishing_only": True,
        "source_mapping": source_mapping,
        "source_counts": dict(Counter(row["source_group"] for row in rows)),
        "pattern_count": args.patterns,
        "pattern_counts": {str(key): int(value) for key, value in enumerate(np.bincount(patterns, minlength=args.patterns))},
        "pattern_method": "RobustScaler plus MiniBatchKMeans over 35 scheme-free PSL-aware lexical features",
        "leakage_contract": "no validation, test, calibration, external, or generated rows",
        "master_sha256": project["data"]["master_sha256"],
        "files": {name: sha256_file(args.output_dir / name) for name in files},
    }
    write_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
