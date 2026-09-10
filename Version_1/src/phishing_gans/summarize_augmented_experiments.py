from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("cnn", "lstm", "rf", "svm")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the controlled augmented-data classifier matrix."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/augmented_experiments"),
    )
    return parser.parse_args()


def parse_condition(name: str) -> tuple[str, int]:
    if name == "real_only_reference_69300":
        return "real_only", 0
    for gan in ("combined_gans", "wgan_gp", "lsgan"):
        prefix = f"{gan}_"
        if name.startswith(prefix) and name.endswith("_percent"):
            percentage = int(name[len(prefix) : -len("_percent")])
            return gan, percentage
    raise ValueError(f"Unrecognized experiment condition: {name}")


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload["test"]
    return {metric: float(test[metric]) for metric in METRICS}


def main() -> None:
    args = parse_args()
    results_root = args.experiment_root / "results"
    rows = []
    for condition_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        gan, percentage = parse_condition(condition_dir.name)
        for model in MODELS:
            path = condition_dir / model / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "condition": condition_dir.name,
                    "gan": gan,
                    "generated_percentage": percentage,
                    "model": model,
                    **load_metrics(path),
                }
            )

    references = {
        row["model"]: row for row in rows if row["generated_percentage"] == 0
    }
    for row in rows:
        reference = references[row["model"]]
        for metric in METRICS:
            row[f"{metric}_delta"] = row[metric] - reference[metric]

    summary = {
        "protocol": {
            "real_training_rows": 69_300,
            "fixed_original_validation_rows": 254_549,
            "fixed_original_test_rows": 252_430,
            "generated_percentages": [20, 30, 40],
            "metrics": list(METRICS),
            "positive_class": "malicious",
        },
        "rows": rows,
        "best_f1_by_model": {},
    }
    for model in MODELS:
        candidates = [row for row in rows if row["model"] == model]
        best = max(candidates, key=lambda row: row["f1"])
        summary["best_f1_by_model"][model] = {
            key: best[key]
            for key in ("condition", "gan", "generated_percentage", *METRICS, "f1_delta")
        }

    json_path = args.experiment_root / "summary.json"
    csv_path = args.experiment_root / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("BEST F1 BY MODEL")
    print("Model\tCondition\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC\tF1 delta")
    for model in MODELS:
        best = summary["best_f1_by_model"][model]
        print(
            f"{model}\t{best['condition']}\t"
            + "\t".join(f"{100 * best[metric]:.4f}%" for metric in METRICS)
            + f"\t{100 * best['f1_delta']:+.4f} pp"
        )
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
