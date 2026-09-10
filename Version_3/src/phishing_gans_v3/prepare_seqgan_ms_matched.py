from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from .common import load_json, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SeqGAN with the exact phishing-token corpus used by MS-HardGAN-C."
    )
    parser.add_argument("--project", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--ms-data", type=Path, default=Path("data/experiments/ms_hardgan_v1"))
    parser.add_argument("--original-data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-data", type=Path, default=Path("data/experiments/seqgan_ms_matched_v1")
    )
    parser.add_argument(
        "--output-config", type=Path, default=Path("configs/seqgan_ms_matched_v1.json")
    )
    return parser.parse_args()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if args.output_data.exists() or args.output_config.exists():
        raise FileExistsError("Matched SeqGAN data/config already frozen")
    report = load_json(args.ms_data / "preparation_report.json")
    rows = int(report["training_rows"])
    tokens = np.load(args.ms_data / "train_tokens.npy", mmap_mode="r")
    if len(tokens) != rows or rows != 84280:
        raise RuntimeError(f"Expected the frozen 84,280-row MS phishing corpus; found {len(tokens)}")

    args.output_data.mkdir(parents=True)
    for name in ("train_tokens.npy", "train_manifest.csv", "vocabulary.json"):
        link(args.ms_data / name, args.output_data / name)
    link(args.original_data / "master.sqlite", args.output_data / "master.sqlite")
    np.save(args.output_data / "train_malicious_indices.npy", np.arange(rows, dtype=np.int64))
    np.save(args.output_data / "train_labels.npy", np.ones(rows, dtype=np.uint8))

    project = load_json(args.project)
    project["generators"]["shared"]["training_rows"] = rows
    project["matched_generator_experiment"] = {
        "name": "seqgan_ms_matched_v1",
        "purpose": "architecture comparison with MS-HardGAN-C using identical phishing training rows",
        "source_corpus": str(args.ms_data),
        "source_train_tokens_sha256": sha256_file(args.ms_data / "train_tokens.npy"),
        "training_rows": rows,
        "other_seqgan_hyperparameters_unchanged": True,
    }
    write_json(args.output_config, project)

    files = (
        "train_tokens.npy", "train_malicious_indices.npy", "train_labels.npy",
        "train_manifest.csv", "vocabulary.json", "master.sqlite",
    )
    output_report = {
        "project_version": 3,
        "experiment": "seqgan_ms_matched_v1",
        "training_rows": rows,
        "phishing_only": True,
        "exact_ms_hardgan_training_tokens": (
            sha256_file(args.output_data / "train_tokens.npy")
            == report["files"]["train_tokens.npy"]
        ),
        "ms_hardgan_preparation_report": str(args.ms_data / "preparation_report.json"),
        "ms_hardgan_preparation_report_sha256": sha256_file(args.ms_data / "preparation_report.json"),
        "source_counts": report["source_counts"],
        "leakage_contract": report["leakage_contract"],
        "config": str(args.output_config),
        "config_sha256": sha256_file(args.output_config),
        "files": {name: sha256_file(args.output_data / name) for name in files},
    }
    if not output_report["exact_ms_hardgan_training_tokens"]:
        raise RuntimeError("Token identity check failed")
    write_json(args.output_data / "preparation_report.json", output_report)
    print(output_report)


if __name__ == "__main__":
    main()
