from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
CONDITIONS = ("real_only", "seqgan_01_percent", "seqgan_02_percent", "seqgan_05_percent", "seqgan_10_percent", "seqgan_15_percent")
MODELS = ("cnn", "lstm")
PROMOTED = {"cnn": "seqgan_02_percent", "lstm": "seqgan_10_percent"}
CONFIRMATION_SEEDS = (42, 43, 44)
T_95_DF_2 = 4.302652729911275


def _paired_summary(real: list[float], augmented: list[float]) -> dict[str, float | list[float]]:
    changes = np.asarray(augmented, dtype=float) - np.asarray(real, dtype=float)
    mean_change = float(changes.mean())
    margin = float(T_95_DF_2 * changes.std(ddof=1) / np.sqrt(changes.size))
    return {
        "real_only_mean": float(np.mean(real)),
        "augmented_mean": float(np.mean(augmented)),
        "paired_changes": changes.tolist(),
        "mean_paired_change": mean_change,
        "ci95_low": mean_change - margin,
        "ci95_high": mean_change + margin,
    }


def main() -> None:
    root = Path("artifacts/low_data/real_10_percent"); output = root / "results"; output.mkdir(parents=True, exist_ok=True)
    rows = []; payload = {"project_version": 2, "experiment": "real_10_percent", "screening_seed": 42, "models": {}}
    for model in MODELS:
        baseline = load_json(root / "real_only" / model / "seed_42" / "metrics.json")
        payload["models"][model] = {}
        for condition in CONDITIONS:
            metrics = load_json(root / condition / model / "seed_42" / "metrics.json")
            result = {metric: {"score": metrics[metric], "delta_vs_real_only": metrics[metric] - baseline[metric]} for metric in METRICS}
            payload["models"][model][condition] = result
            rows.extend({"model": model, "condition": condition, "metric": metric, **values} for metric, values in result.items())
    payload["promoted_confirmations"] = {}
    confirmation_rows = []
    for model, condition in PROMOTED.items():
        payload["promoted_confirmations"][model] = {
            "condition": condition,
            "seeds": list(CONFIRMATION_SEEDS),
            "metrics": {},
        }
        for metric in METRICS:
            real = [load_json(root / "real_only" / model / f"seed_{seed}" / "metrics.json")[metric]
                    for seed in CONFIRMATION_SEEDS]
            augmented = [load_json(root / condition / model / f"seed_{seed}" / "metrics.json")[metric]
                         for seed in CONFIRMATION_SEEDS]
            result = _paired_summary(real, augmented)
            payload["promoted_confirmations"][model]["metrics"][metric] = result
            for seed, real_score, augmented_score, change in zip(
                CONFIRMATION_SEEDS, real, augmented, result["paired_changes"]
            ):
                confirmation_rows.append({
                    "model": model,
                    "condition": condition,
                    "metric": metric,
                    "seed": seed,
                    "real_only": real_score,
                    "augmented": augmented_score,
                    "paired_change": change,
                    "mean_paired_change": result["mean_paired_change"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                })
    write_json(output / "summary.json", payload)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "condition", "metric", "score", "delta_vs_real_only")); writer.writeheader(); writer.writerows(rows)
    with (output / "confirmation.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("model", "condition", "metric", "seed", "real_only", "augmented", "paired_change",
                  "mean_paired_change", "ci95_low", "ci95_high")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(confirmation_rows)
    labels = ("Real only", "SeqGAN 1%", "SeqGAN 2%", "SeqGAN 5%", "SeqGAN 10%", "SeqGAN 15%")
    colors = ("#4a5568", "#1a365d", "#2b6cb0", "#3182ce", "#63b3ed", "#90cdf4")
    for model in MODELS:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.5))
        for axis, metric in zip(axes, METRICS):
            values = [100 * payload["models"][model][condition][metric]["score"] for condition in CONDITIONS]
            axis.bar(np.arange(len(CONDITIONS)), values, color=colors); axis.set_xticks(np.arange(len(CONDITIONS)), labels, rotation=28, ha="right")
            axis.set_title(metric.replace("_", " ").upper()); axis.set_ylabel("Score (%)"); axis.grid(axis="y", alpha=.25)
            low = max(0, min(values) - max(.3, (max(values)-min(values))*1.5)); high = min(100, max(values) + max(.3, (max(values)-min(values))*1.5)); axis.set_ylim(low, high)
        fig.suptitle(f"10%-real low-data screen — {model.upper()} (seed 42, synthetic weight 0.5)", fontweight="bold")
        fig.tight_layout(); fig.savefig(output / f"{model}_metric_comparison.png", dpi=220, bbox_inches="tight")
        fig.savefig(output / f"{model}_metric_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    matrix = np.asarray([[100 * payload["models"][model][condition][metric]["delta_vs_real_only"]
                          for condition in CONDITIONS[1:] for metric in METRICS] for model in MODELS])
    maximum = max(.01, float(np.abs(matrix).max())); fig, axis = plt.subplots(figsize=(14, 3.7))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-maximum, vmax=maximum, aspect="auto")
    column_labels = [f"{condition.split('_')[1]}% {metric.replace('_',' ')}" for condition in CONDITIONS[1:] for metric in METRICS]
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=45, ha="right"); axis.set_yticks((0,1), ("CNN","LSTM"))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): axis.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=7)
    axis.set_title("Augmented minus matched real-only (percentage points)"); fig.colorbar(image, ax=axis, label="Change (percentage points)")
    fig.tight_layout(); fig.savefig(output / "delta_heatmap.png", dpi=220, bbox_inches="tight"); fig.savefig(output / "delta_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    # Confirmation figure: a confidence interval crossing zero means the observed
    # direction is not stable across these three paired random seeds.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    metric_labels = [metric.replace("_", " ").upper() for metric in METRICS]
    x = np.arange(len(METRICS))
    for axis, model in zip(axes, MODELS):
        summaries = payload["promoted_confirmations"][model]["metrics"]
        means = np.asarray([100 * summaries[metric]["mean_paired_change"] for metric in METRICS])
        lows = np.asarray([100 * summaries[metric]["ci95_low"] for metric in METRICS])
        highs = np.asarray([100 * summaries[metric]["ci95_high"] for metric in METRICS])
        error = np.vstack((means - lows, highs - means))
        axis.axhline(0, color="#17202a", linewidth=1.2)
        axis.errorbar(x, means, yerr=error, fmt="o", markersize=7, capsize=5,
                      color="#2166ac", ecolor="#4d4d4d", linewidth=1.6)
        for metric_index, metric in enumerate(METRICS):
            paired = 100 * np.asarray(summaries[metric]["paired_changes"])
            axis.scatter(np.full(paired.size, metric_index), paired, marker="x", color="#b2182b", s=38, zorder=3)
        percent = PROMOTED[model].split("_")[1]
        axis.set_title(f"{model.upper()}: SeqGAN {int(percent)}% vs matched real-only", fontweight="bold")
        axis.set_ylabel("Augmented - real only\n(percentage points)")
        axis.grid(axis="y", alpha=.22)
    axes[-1].set_xticks(x, metric_labels)
    fig.suptitle("Low-data confirmation across paired seeds 42, 43, and 44\n"
                 "Blue: mean with 95% t interval; red ×: individual paired seed", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "paired_confirmation_ci.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "paired_confirmation_ci.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
