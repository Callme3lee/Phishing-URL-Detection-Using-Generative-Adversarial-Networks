from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated classifier experiments with 95% confidence intervals."
    )
    parser.add_argument(
        "--condition",
        action="append",
        nargs="+",
        metavar=("NAME", "METRICS_JSON"),
        required=True,
        help="Condition name followed by two or more classifier metrics.json files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def t_critical_95(degrees_of_freedom: int) -> float:
    # Exact common small-sample values; normal approximation after 30 samples.
    values = {
        1: 12.7062047364,
        2: 4.3026527299,
        3: 3.1824463053,
        4: 2.7764451052,
        5: 2.5705818356,
        6: 2.4469118511,
        7: 2.3646242510,
        8: 2.3060041350,
        9: 2.2621571628,
        10: 2.2281388520,
        15: 2.1314495456,
        20: 2.0859634473,
        25: 2.0595385528,
        30: 2.0422724563,
    }
    if degrees_of_freedom in values:
        return values[degrees_of_freedom]
    lower = max(key for key in values if key < degrees_of_freedom)
    upper_candidates = [key for key in values if key > degrees_of_freedom]
    if not upper_candidates:
        return 1.9599639845
    upper = min(upper_candidates)
    fraction = (degrees_of_freedom - lower) / (upper - lower)
    return values[lower] + fraction * (values[upper] - values[lower])


def load_test_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("test", payload.get("metrics"))
    if result is None:
        raise ValueError(f"No 'test' or 'metrics' object in {path}")
    return {metric: float(result[metric]) for metric in METRICS}


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("Each condition needs at least two runs.")
    runs = [load_test_metrics(path) for path in paths]
    critical = t_critical_95(len(runs) - 1)
    summary: dict[str, dict[str, float | list[float]]] = {}
    for metric in METRICS:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        standard_deviation = float(values.std(ddof=1))
        half_width = critical * standard_deviation / math.sqrt(len(values))
        mean = float(values.mean())
        summary[metric] = {
            "values": values.tolist(),
            "mean": mean,
            "sample_standard_deviation": standard_deviation,
            "ci95": [mean - half_width, mean + half_width],
            "ci95_half_width": half_width,
        }
    return {
        "n": len(runs),
        "files": [str(path) for path in paths],
        "metrics": summary,
    }


def main() -> None:
    args = parse_args()
    conditions: dict[str, object] = {}
    for condition in args.condition:
        name, *path_strings = condition
        if name in conditions:
            raise ValueError(f"Duplicate condition name: {name}")
        conditions[name] = summarize([Path(path) for path in path_strings])

    payload = {
        "confidence_interval": "two-sided 95% Student's t interval",
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("condition\tF1 mean+-SD\tROC-AUC mean+-SD\tFPR mean+-SD")
    for name, condition in conditions.items():
        metric_summary = condition["metrics"]
        cells = []
        for metric in ("f1", "roc_auc", "false_positive_rate"):
            result = metric_summary[metric]
            cells.append(
                f"{100 * result['mean']:.4f}+-{100 * result['sample_standard_deviation']:.4f}%"
            )
        print(f"{name}\t" + "\t".join(cells))


if __name__ == "__main__":
    main()
