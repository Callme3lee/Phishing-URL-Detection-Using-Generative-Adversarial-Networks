from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
CONDITIONS = {"real_only": "Modern real-only", "lsgan_10": "Modern + 10% LSGAN"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")
T_CRITICAL_DF2 = 4.3026527297


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half_width = T_CRITICAL_DF2 * std / np.sqrt(len(array))
    return {
        "mean": mean,
        "std_across_seeds": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modern 10% LSGAN augmentation.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Modern Temporally Held-Out URL Evaluation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for seed in SEEDS:
            training_path = args.results_dir / f"seed{seed}" / condition / "metrics.json"
            checkpoint_path = args.results_dir / condition / f"seed{seed}.json"
            if training_path.exists():
                raw[condition][seed] = json.loads(
                    training_path.read_text(encoding="utf-8")
                )["test"]
            elif checkpoint_path.exists():
                raw[condition][seed] = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )["metrics"]
            else:
                raise FileNotFoundError(
                    f"Found neither {training_path} nor {checkpoint_path}"
                )
    report = {
        "experiment": "modern real-only versus exact-10%-quality-LSGAN classifier adaptation",
        "seeds": list(SEEDS),
        "conditions": {},
        "paired_augmented_minus_real": {},
        "threshold_policy": "maximum validation F1 subject to FPR <= 1%; no test threshold adjustment",
    }
    for condition in CONDITIONS:
        report["conditions"][condition] = {
            metric: stats([raw[condition][seed][metric] for seed in SEEDS])
            for metric in METRICS
        }
        report["conditions"][condition]["per_seed"] = {
            str(seed): raw[condition][seed] for seed in SEEDS
        }
    for metric in METRICS:
        report["paired_augmented_minus_real"][metric] = stats(
            [raw["lsgan_10"][seed][metric] - raw["real_only"][seed][metric] for seed in SEEDS]
        )

    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "mean", "std", "ci95_low", "ci95_high"])
        for condition, label in CONDITIONS.items():
            for metric in METRICS:
                item = report["conditions"][condition][metric]
                writer.writerow([label, metric, item["mean"], item["std_across_seeds"], item["ci95_low"], item["ci95_high"]])

    plot_metrics = METRICS[:5]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.2), sharey=True)
    colors = ("#235789", "#2A9D8F")
    for axis, metric in zip(axes, plot_metrics):
        means = [report["conditions"][condition][metric]["mean"] * 100 for condition in CONDITIONS]
        stds = [report["conditions"][condition][metric]["std_across_seeds"] * 100 for condition in CONDITIONS]
        bars = axis.bar(range(2), means, yerr=stds, capsize=5, color=colors, width=0.68)
        axis.set_title(metric.replace("_", " ").upper(), weight="bold", fontsize=11)
        axis.set_xticks(range(2), ["Real-only", "+ 10% modern LSGAN"], rotation=18, ha="right")
        axis.set_ylim(0, 100)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle(args.title, fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.015,
        "Mean of seeds 42/43/44 • error bars = seed SD • frozen threshold selected at validation FPR ≤ 1%",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(args.output_dir / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
