from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .summarize_augmented_experiments import METRICS, MODELS, parse_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize GAN-unseen external evaluation for augmented classifiers."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/augmented_experiments"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.experiment_root / "results"
    rows = []
    for condition_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        gan, percentage = parse_condition(condition_dir.name)
        for model in MODELS:
            model_dir = condition_dir / model
            for path in sorted(model_dir.glob("external_heldout_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "source": payload["source"],
                        "condition": condition_dir.name,
                        "gan": gan,
                        "generated_percentage": percentage,
                        "model": model,
                        **{metric: float(payload["metrics"][metric]) for metric in METRICS},
                    }
                )

    expected = len({row["source"] for row in rows}) * 10 * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Found {len(rows)} rows, expected {expected}.")
    references = {
        (row["source"], row["model"]): row
        for row in rows
        if row["generated_percentage"] == 0
    }
    for row in rows:
        reference = references[(row["source"], row["model"])]
        for metric in METRICS:
            row[f"{metric}_delta"] = row[metric] - reference[metric]

    sources = sorted({row["source"] for row in rows})
    best_by_source_model = {}
    for source in sources:
        best_by_source_model[source] = {}
        for model in MODELS:
            candidates = [
                row for row in rows if row["source"] == source and row["model"] == model
            ]
            best_f1 = max(candidates, key=lambda row: row["f1"])
            best_roc = max(candidates, key=lambda row: row["roc_auc"])
            best_by_source_model[source][model] = {
                "best_f1": {
                    key: best_f1[key]
                    for key in ("condition", *METRICS, "f1_delta", "roc_auc_delta")
                },
                "best_roc_auc": {
                    key: best_roc[key]
                    for key in ("condition", *METRICS, "f1_delta", "roc_auc_delta")
                },
            }

    macro_rows = []
    for condition in sorted({row["condition"] for row in rows}):
        for model in MODELS:
            selected = [
                row for row in rows if row["condition"] == condition and row["model"] == model
            ]
            if len(selected) != len(sources):
                raise RuntimeError(f"Missing a source for {condition}/{model}")
            gan, percentage = parse_condition(condition)
            macro_rows.append(
                {
                    "condition": condition,
                    "gan": gan,
                    "generated_percentage": percentage,
                    "model": model,
                    **{
                        metric: sum(row[metric] for row in selected) / len(selected)
                        for metric in METRICS
                    },
                }
            )
    macro_references = {
        row["model"]: row for row in macro_rows if row["generated_percentage"] == 0
    }
    for row in macro_rows:
        reference = macro_references[row["model"]]
        for metric in METRICS:
            row[f"{metric}_delta"] = row[metric] - reference[metric]

    best_macro_by_model = {}
    for model in MODELS:
        candidates = [row for row in macro_rows if row["model"] == model]
        best_joint = max(candidates, key=lambda row: (row["roc_auc"], row["f1"]))
        best_macro_by_model[model] = best_joint

    summary = {
        "protocol": {
            "sources": sources,
            "thresholds": "frozen from original validation set",
            "GAN_unseen": True,
            "source_macro_average": "unweighted mean across the two held-out sources",
        },
        "rows": rows,
        "best_by_source_model": best_by_source_model,
        "source_macro_rows": macro_rows,
        "best_macro_roc_by_model": best_macro_by_model,
    }
    json_path = args.experiment_root / "external_heldout_summary.json"
    csv_path = args.experiment_root / "external_heldout_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for source in sources:
        print(f"SOURCE: {source}")
        print("Model\tBest F1 condition\tF1\tROC-AUC\tBest ROC condition\tROC-AUC\tF1")
        for model in MODELS:
            result = best_by_source_model[source][model]
            f1 = result["best_f1"]
            roc = result["best_roc_auc"]
            print(
                f"{model}\t{f1['condition']}\t{100*f1['f1']:.2f}%\t{100*f1['roc_auc']:.2f}%\t"
                f"{roc['condition']}\t{100*roc['roc_auc']:.2f}%\t{100*roc['f1']:.2f}%"
            )
    print("SOURCE-MACRO BEST ROC-AUC CONDITION")
    for model in MODELS:
        row = best_macro_by_model[model]
        print(
            f"{model}\t{row['condition']}\tF1={100*row['f1']:.2f}%\t"
            f"ROC-AUC={100*row['roc_auc']:.2f}%"
        )
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
