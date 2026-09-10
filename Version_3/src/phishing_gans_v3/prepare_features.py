from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import load_json, write_json
from .url_data import LEXICAL_FEATURE_NAMES, lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scheme-independent Version 3 lexical features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/features"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_json(args.data_dir / "preparation_report.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        total = report["splits"][split]["total"]
        matrix = np.lib.format.open_memmap(
            args.output_dir / f"{split}_features.npy", mode="w+", dtype=np.float32,
            shape=(total, len(LEXICAL_FEATURE_NAMES)),
        )
        with (args.data_dir / f"{split}_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                matrix[index] = lexical_features(row["url"])
                if index and index % 100000 == 0:
                    print(json.dumps({"split": split, "rows": index}), flush=True)
        matrix.flush()
    write_json(args.output_dir / "metadata.json", {
        "project_version": 3,
        "feature_names": LEXICAL_FEATURE_NAMES,
        "feature_count": len(LEXICAL_FEATURE_NAMES),
        "scheme_feature_present": False,
        "single_master_dataset": True,
    })


if __name__ == "__main__":
    main()
