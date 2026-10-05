#!/usr/bin/env python3
"""Assign half mass to each root pair so duplication does not reweight identities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "Final_Version/data/root_paired_hyb_ms_gan10_v1/gan_10pct"
PROCESSED = ROOT / "processed"


def is_root(url: str) -> bool:
    try:
        parsed = urlsplit("http://" + str(url).lstrip("/"))
    except ValueError:
        return False
    return bool(parsed.hostname) and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment


def main() -> None:
    path = PROCESSED / "train_sample_factors.npy"
    if path.exists():
        raise FileExistsError(path)
    report_path = ROOT / "experiment_report.json"
    report = json.loads(report_path.read_text())
    source_rows = int(report["source_training_rows"])
    with (PROCESSED / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    groups = np.load(PROCESSED / "train_groups.npy", mmap_mode="r")
    factors = np.ones(len(manifest), dtype=np.float64)
    for index, row in enumerate(manifest):
        if is_root(row["url"]):
            factors[index] = 0.5
    if int(np.sum(factors == 0.5)) != 2 * int(report["root_rows_paired"]):
        raise RuntimeError("Root-pair factor count does not match preparation report")
    np.save(path, factors)
    effective = {str(int(group)): float(factors[groups == group].sum()) for group in np.unique(groups)}
    report["pair_preserving_sampling"] = {
        "factor_file": "processed/train_sample_factors.npy",
        "non_root_factor": 1.0,
        "each_root_representation_factor": 0.5,
        "effective_training_mass": float(factors.sum()),
        "effective_group_mass": effective,
        "reason": "both root representations remain present while each original root identity retains total sampling mass 1.0",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["pair_preserving_sampling"], indent=2))


if __name__ == "__main__":
    main()
