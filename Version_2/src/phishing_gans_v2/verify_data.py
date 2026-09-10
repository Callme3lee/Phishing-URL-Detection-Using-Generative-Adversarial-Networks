from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_json, resolve_from_config, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Version 2 prepared data.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project = load_json(config_path)
    raw = resolve_from_config(config_path, project["data"]["master_csv"])
    report = load_json(args.data_dir / "preparation_report.json")
    checks: dict[str, object] = {}
    checks["master_hash_matches_config"] = sha256_file(raw) == project["data"]["master_sha256"]
    checks["report_hash_matches_config"] = report["master_sha256"] == project["data"]["master_sha256"]
    checks["external_data_used_is_false"] = report["external_data_used"] is False
    checks["encoded_conflicts_removed"] = report["encoded_audit"]["retained_encoded_unique"] == sum(
        split["total"] for split in report["splits"].values()
    )
    for split in ("train", "validation", "test"):
        tokens = np.load(args.data_dir / f"{split}_tokens.npy", mmap_mode="r")
        labels = np.load(args.data_dir / f"{split}_labels.npy", mmap_mode="r")
        indices = np.load(args.data_dir / f"{split}_malicious_indices.npy", mmap_mode="r")
        expected = report["splits"][split]
        checks[f"{split}_shape"] = list(tokens.shape) == [expected["total"], project["data"]["maximum_encoded_length"]]
        checks[f"{split}_labels_shape"] = labels.shape == (expected["total"],)
        checks[f"{split}_label_counts"] = (
            int((labels == 0).sum()) == expected["label_0"]
            and int((labels == 1).sum()) == expected["label_1"]
        )
        checks[f"{split}_gan_indices_are_malicious"] = bool(np.all(labels[indices] == 1))
        checks[f"{split}_manifest_hash"] = sha256_file(args.data_dir / f"{split}_manifest.csv") == expected["manifest_sha256"]
    passed = all(value is True for value in checks.values())
    payload = {"passed": passed, "checks": checks}
    write_json(args.data_dir / "verification_report.json", payload)
    if not passed:
        raise RuntimeError(f"Data verification failed: {payload}")
    print(payload)


if __name__ == "__main__":
    main()
