#!/usr/bin/env python3
"""Contract checks for the isolated Final Version experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "Final_Version/data/root_paired_hyb_ms_gan10_v1/gan_10pct"
V3 = ROOT / "Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/processed"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    report = json.loads((FINAL / "experiment_report.json").read_text())
    processed = FINAL / "processed"
    assert report["source_training_rows"] == 407_594
    assert report["source_real_training_rows"] == 367_184
    assert report["source_synthetic_training_rows"] == 40_410
    assert report["root_rows_paired"] == 55_262
    assert report["non_root_rows_changed"] == 0
    assert report["validation_test_changed"] is False

    for key, value in report["final_root_counts"].items():
        if key.endswith("_empty"):
            assert value == report["final_root_counts"][key.removesuffix("empty") + "slash"], key

    factors = np.load(processed / "train_sample_factors.npy")
    groups = np.load(processed / "train_groups.npy")
    assert len(factors) == len(groups) == report["training_rows"]
    assert float(factors.sum()) == 407_594.0
    expected = {0: 276_400, 1: 4_758, 2: 82_219, 3: 3_746, 4: 61, 5: 40_410}
    assert {group: float(factors[groups == group].sum()) for group in expected} == expected

    unchanged = [
        "validation_tokens.npy", "validation_labels.npy", "validation_psl_lexical.npy",
        "validation_manifest.csv", "test_tokens.npy", "test_labels.npy", "test_psl_lexical.npy",
        "test_manifest.csv", "domain_validation_tokens.npy", "domain_validation_labels.npy",
        "domain_validation_psl_lexical.npy", "domain_validation_manifest.csv", "vocabulary.json",
        "psl_lexical_scaler.npz",
    ]
    for name in unchanged:
        assert digest(V3 / name) == digest(processed / name), name

    with (processed / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == report["training_rows"]
    print("PASS: Final Version data, pairing, sampling mass, and evaluation isolation contracts hold")


if __name__ == "__main__":
    main()
