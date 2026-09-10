from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import save_json
from .summarize_robust_augmentation import (
    CONDITIONS,
    METRICS,
    MODELS,
    result_root,
    summary,
)


SOURCES = ("new_data_urls", "url_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize three-seed external results for the robust experiment."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/robust_augmentation"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Summarize matching-source validation-calibrated thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = {
        model: {
            condition: {source: {} for source in SOURCES} for condition in CONDITIONS
        }
        for model in MODELS
    }
    for seed in args.seeds:
        root = result_root(args.experiment_root, seed)
        for model in MODELS:
            for condition in CONDITIONS:
                for source in SOURCES:
                    filename = (
                        f"calibrated_external_{source}.json"
                        if args.calibrated
                        else f"external_heldout_{source}.json"
                    )
                    path = root / condition / model / filename
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    values[model][condition][source][seed] = payload["metrics"]

    aggregate = {}
    rows = []
    for model in MODELS:
        aggregate[model] = {}
        for condition in CONDITIONS:
            source_summaries = {}
            macro_summaries = {}
            for source in SOURCES:
                source_summaries[source] = {
                    metric: summary(
                        [values[model][condition][source][seed][metric] for seed in args.seeds]
                    )
                    for metric in METRICS
                }
            for metric in METRICS:
                macro_by_seed = [
                    float(
                        np.mean(
                            [
                                values[model][condition][source][seed][metric]
                                for source in SOURCES
                            ]
                        )
                    )
                    for seed in args.seeds
                ]
                macro_summaries[metric] = summary(macro_by_seed)
            aggregate[model][condition] = {
                "sources": source_summaries,
                "source_macro": macro_summaries,
            }
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    **{
                        f"{source}_{metric}_mean": source_summaries[source][metric]["mean"]
                        for source in SOURCES
                        for metric in METRICS
                    },
                    **{
                        f"macro_{metric}_{field}": macro_summaries[metric][field]
                        for metric in METRICS
                        for field in ("mean", "std", "ci95_low", "ci95_high")
                    },
                }
            )

    best = {}
    for model in MODELS:
        best_f1 = max(
            CONDITIONS,
            key=lambda condition: aggregate[model][condition]["source_macro"]["f1"]["mean"],
        )
        best_roc = max(
            CONDITIONS,
            key=lambda condition: aggregate[model][condition]["source_macro"]["roc_auc"]["mean"],
        )
        best[model] = {
            "f1_condition": best_f1,
            "f1_mean": aggregate[model][best_f1]["source_macro"]["f1"]["mean"],
            "roc_auc_condition": best_roc,
            "roc_auc_mean": aggregate[model][best_roc]["source_macro"]["roc_auc"]["mean"],
        }
    prefix = "calibrated_external" if args.calibrated else "external"
    save_json(
        args.experiment_root / f"{prefix}_three_seed_summary.json",
        {
            "experiment": "source-balanced held-out external evaluation",
            "sources": list(SOURCES),
            "seeds": args.seeds,
            "metrics": aggregate,
            "best_source_macro": best,
        },
    )
    with (args.experiment_root / f"{prefix}_three_seed_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    label_map = {
        "real_only": "Real",
        "lsgan_quality_10": "LSGAN 10%",
        "lsgan_quality_20": "LSGAN 20%",
        "combined_random_30": "Random mix 30%",
        "combined_quality_30": "Quality mix 30%",
        "lsgan_quality_30": "LSGAN 30%",
    }
    labels = [label_map[condition] for condition in CONDITIONS]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(CONDITIONS))
    for axis, model in zip(axes.flat, MODELS):
        f1 = [
            100 * aggregate[model][condition]["source_macro"]["f1"]["mean"]
            for condition in CONDITIONS
        ]
        roc = [
            100 * aggregate[model][condition]["source_macro"]["roc_auc"]["mean"]
            for condition in CONDITIONS
        ]
        axis.plot(x, f1, marker="o", linewidth=2, label="Macro F1")
        axis.plot(x, roc, marker="s", linewidth=2, label="Macro ROC-AUC")
        axis.set_title(model.upper())
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("Score (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Held-out source performance after source-balanced training (3-seed mean)")
    figure.tight_layout()
    figure.savefig(args.experiment_root / f"{prefix}_three_seed.png", dpi=200, bbox_inches="tight")
    figure.savefig(args.experiment_root / f"{prefix}_three_seed.pdf", bbox_inches="tight")
    plt.close(figure)

    for model in MODELS:
        item = best[model]
        print(
            f"{model.upper()}: macro-F1 {item['f1_condition']}={100 * item['f1_mean']:.2f}% | "
            f"macro-ROC {item['roc_auc_condition']}={100 * item['roc_auc_mean']:.2f}%"
        )


if __name__ == "__main__":
    main()
