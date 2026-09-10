from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import save_json


MODELS = ("cnn", "lstm", "rf", "svm")
CONDITIONS = (
    "real_only",
    "lsgan_quality_10",
    "lsgan_quality_20",
    "combined_random_30",
    "combined_quality_30",
    "lsgan_quality_30",
)
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
T_95_DF_2 = 4.30265272975


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the three-seed source-balanced augmentation experiment."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/robust_augmentation"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def result_root(root: Path, seed: int) -> Path:
    return root / ("results" if seed == 42 else f"results_seed{seed}")


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    margin = T_95_DF_2 * std / np.sqrt(len(array)) if len(array) == 3 else None
    return {
        "values": [float(value) for value in array],
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
    }


def main() -> None:
    args = parse_args()
    raw: dict[str, dict[str, dict[int, dict]]] = {
        model: {condition: {} for condition in CONDITIONS} for model in MODELS
    }
    for seed in args.seeds:
        root = result_root(args.experiment_root, seed)
        for condition in CONDITIONS:
            for model in MODELS:
                path = root / condition / model / "metrics.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw[model][condition][seed] = payload["test"]

    aggregate = {}
    rows = []
    for model in MODELS:
        aggregate[model] = {}
        real = raw[model]["real_only"]
        for condition in CONDITIONS:
            metrics_summary = {}
            for metric in METRICS:
                values = [raw[model][condition][seed][metric] for seed in args.seeds]
                metric_summary = summary(values)
                changes = [
                    raw[model][condition][seed][metric] - real[seed][metric]
                    for seed in args.seeds
                ]
                metric_summary["paired_change_vs_real"] = summary(changes)
                metrics_summary[metric] = metric_summary
            aggregate[model][condition] = metrics_summary
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    **{
                        f"{metric}_{field}": metrics_summary[metric][field]
                        for metric in METRICS
                        for field in ("mean", "std", "ci95_low", "ci95_high")
                    },
                    **{
                        f"{metric}_paired_change_mean": metrics_summary[metric][
                            "paired_change_vs_real"
                        ]["mean"]
                        for metric in METRICS
                    },
                }
            )

    best = {}
    for model in MODELS:
        condition = max(CONDITIONS, key=lambda item: aggregate[model][item]["f1"]["mean"])
        best[model] = {
            "condition": condition,
            "f1_mean": aggregate[model][condition]["f1"]["mean"],
            "f1_change_vs_real": aggregate[model][condition]["f1"][
                "paired_change_vs_real"
            ],
            "roc_auc_mean": aggregate[model][condition]["roc_auc"]["mean"],
        }

    output = {
        "experiment": "source-balanced quality-filtered GAN augmentation",
        "seeds": args.seeds,
        "metrics": aggregate,
        "best_mean_f1": best,
    }
    save_json(args.experiment_root / "three_seed_summary.json", output)
    with (args.experiment_root / "three_seed_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "real_only": "Real only",
        "lsgan_quality_10": "LSGAN quality 10%",
        "lsgan_quality_20": "LSGAN quality 20%",
        "combined_random_30": "Combined random 30%",
        "combined_quality_30": "Combined quality 30%",
        "lsgan_quality_30": "LSGAN quality 30%",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    for axis, model in zip(axes.flat, MODELS):
        means = [100 * aggregate[model][condition]["f1"]["mean"] for condition in CONDITIONS]
        stds = [100 * aggregate[model][condition]["f1"]["std"] for condition in CONDITIONS]
        positions = np.arange(len(CONDITIONS))
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
        axis.bar(positions, means, yerr=stds, capsize=4, color=colors)
        axis.set_title(model.upper())
        axis.set_ylabel("F1 (%)")
        axis.set_xticks(positions, [labels[value] for value in CONDITIONS], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        lower = min(means) - max(max(stds), 0.2) - 0.2
        upper = max(means) + max(max(stds), 0.2) + 0.2
        axis.set_ylim(lower, upper)
    figure.suptitle("Source-balanced GAN augmentation: mean F1 across seeds 42, 43, and 44")
    figure.tight_layout()
    figure.savefig(args.experiment_root / "three_seed_f1.png", dpi=200, bbox_inches="tight")
    figure.savefig(args.experiment_root / "three_seed_f1.pdf", bbox_inches="tight")
    plt.close(figure)

    for model in MODELS:
        item = best[model]
        change = 100 * item["f1_change_vs_real"]["mean"]
        print(
            f"{model.upper()}: {item['condition']} F1={100 * item['f1_mean']:.4f}% "
            f"change={change:+.4f} pp ROC-AUC={100 * item['roc_auc_mean']:.4f}%"
        )


if __name__ == "__main__":
    main()
