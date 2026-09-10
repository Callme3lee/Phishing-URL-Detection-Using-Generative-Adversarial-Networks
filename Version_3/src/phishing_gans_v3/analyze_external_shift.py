from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import write_json
from .url_data import LEXICAL_FEATURE_NAMES, lexical_features


FEATURES = ("url_length", "hostname_length", "path_length", "dot_count_host", "estimated_subdomains", "slash_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Version 3 internal and external URL structure.")
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-11/external_urls.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/external/2026-08-11_topdomains_phishtank/structural_shift.json"))
    return parser.parse_args()


def means(matrix: np.ndarray) -> dict[str, float]:
    return {name: float(matrix[:, LEXICAL_FEATURE_NAMES.index(name)].mean()) for name in FEATURES}


def main() -> None:
    args = parse_args()
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    external = {
        "legitimate": means(np.stack([lexical_features(row["url"]) for row in rows if row["label"] == "0"])),
        "phishing": means(np.stack([lexical_features(row["url"]) for row in rows if row["label"] == "1"])),
    }
    internal = {}
    for split in ("train", "test"):
        matrix = np.load(args.feature_dir / f"{split}_features.npy", mmap_mode="r")
        labels = np.load(args.data_dir / f"{split}_labels.npy", mmap_mode="r")
        internal[split] = {
            "legitimate": means(np.asarray(matrix[labels == 0])),
            "phishing": means(np.asarray(matrix[labels == 1])),
        }
    payload = {
        "project_version": 3, "feature_means": {"internal": internal, "external": external},
        "observations": [
            "External legitimate examples are root proxies and are much shorter than internal legitimate URLs.",
            "The internal hostname-dot relationship reverses externally: internal legitimate URLs have more dots than phishing, while current external phishing URLs have more dots than legitimate roots.",
            "This shift can explain high external false-positive rates and traditional-model ROC-AUC below 0.5.",
        ],
    }
    write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
