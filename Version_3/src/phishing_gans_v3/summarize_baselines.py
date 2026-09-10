from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Version 3 baseline seeds.")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/baseline_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"project_version": 3, "models": {}}
    for model_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        runs = [load_json(path) for path in sorted(model_dir.glob("seed_*/metrics.json"))]
        if not runs:
            continue
        payload["models"][model_dir.name] = {
            "seeds": [run["seed"] for run in runs],
            "real_training_rows": runs[0]["real_training_rows"],
            "metrics": {
                metric: {
                    "mean": float(np.mean([run[metric] for run in runs])),
                    "standard_deviation": float(np.std([run[metric] for run in runs], ddof=1)) if len(runs) > 1 else 0.0,
                }
                for metric in METRICS
            },
        }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
