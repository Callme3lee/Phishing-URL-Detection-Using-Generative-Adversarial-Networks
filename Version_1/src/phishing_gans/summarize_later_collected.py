from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
GROUPS = {
    "real_only": "Real-only CNN",
    "aug20_then_real": "20% GAN pretrain + real fine-tune",
}
SEEDS = (42, 43, 44)
T_CRITICAL_DF2 = 4.3026527297


def load_results(root: Path) -> dict[str, dict[int, dict]]:
    results: dict[str, dict[int, dict]] = {}
    for group in GROUPS:
        results[group] = {}
        for seed in SEEDS:
            path = root / group / f"seed{seed}.json"
            results[group][seed] = json.loads(path.read_text(encoding="utf-8"))
    return results


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = T_CRITICAL_DF2 * std / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "std_across_seeds": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize(results: dict[str, dict[int, dict]]) -> dict:
    report: dict = {"seeds": list(SEEDS), "groups": {}, "paired_aug_minus_real": {}}
    for group in GROUPS:
        report["groups"][group] = {
            metric: stats(
                [results[group][seed]["metrics"][metric] for seed in SEEDS]
            )
            for metric in METRICS
        }
        report["groups"][group]["per_seed"] = {
            str(seed): results[group][seed]["metrics"] for seed in SEEDS
        }
    for metric in METRICS:
        deltas = [
            results["aug20_then_real"][seed]["metrics"][metric]
            - results["real_only"][seed]["metrics"][metric]
            for seed in SEEDS
        ]
        report["paired_aug_minus_real"][metric] = {**stats(deltas), "values": deltas}
    return report


def save_csv(report: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", "mean", "std_across_seeds", "ci95_low", "ci95_high"])
        for group, label in GROUPS.items():
            for metric in METRICS:
                item = report["groups"][group][metric]
                writer.writerow(
                    [label, metric, item["mean"], item["std_across_seeds"], item["ci95_low"], item["ci95_high"]]
                )


def save_plot(report: dict, path: Path, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5.2), sharey=True)
    colors = ("#235789", "#E07A5F")
    for axis, metric in zip(axes, METRICS):
        means = [report["groups"][group][metric]["mean"] * 100 for group in GROUPS]
        stds = [report["groups"][group][metric]["std_across_seeds"] * 100 for group in GROUPS]
        bars = axis.bar(range(2), means, yerr=stds, capsize=5, color=colors, width=0.68)
        axis.set_title(metric.replace("_", " ").upper(), fontsize=11, weight="bold")
        axis.set_xticks(range(2), ["Real-only", "GAN + fine-tune"], rotation=18, ha="right")
        axis.set_ylim(0, 100)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.015,
        "Frozen thresholds • 255 unseen-host phishing + 255 unseen-host benign URLs • bars = mean of seeds 42/43/44; error bars = seed SD",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize later-collected checkpoint evaluations.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(load_results(args.results_dir))
    report["interpretation"] = (
        "External frozen evaluation only. Confidence intervals describe variation across three training seeds, "
        "not uncertainty from resampling the 510 evaluation examples."
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_csv(report, args.output_dir / "summary.csv")
    save_plot(report, args.output_dir / "comparison.png", args.title)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
