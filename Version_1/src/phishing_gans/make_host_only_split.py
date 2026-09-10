from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import encode_url, load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a host-only sensitivity split from a frozen URL evaluation set."
    )
    parser.add_argument("--frozen-csv", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.frozen_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = load_json(args.training_data_dir / "metadata.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(
        args.output_dir / "test_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), metadata["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        args.output_dir / "test_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    for index, row in enumerate(rows):
        tokens[index] = encode_url(
            row["hostname"], metadata["vocab"], metadata["max_length"]
        )
        labels[index] = int(row["label"])
    tokens.flush()
    labels.flush()
    report = {
        "role": "diagnostic sensitivity split only; not the primary browser-style evaluation",
        "transformation": "both phishing and benign inputs reduced to their frozen canonical hostname",
        "examples": len(rows),
        "source_frozen_csv": str(args.frozen_csv),
        "warning": "This split removes path/query phishing signals and does not represent complete browser URLs.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
