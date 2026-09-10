from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "random_forest", "linear_svm")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def main() -> None:
    baseline = Path("artifacts/baselines")
    augmented = Path("artifacts/augmentation/relgan/generated_10_percent")
    output = Path("artifacts/results/augmentation/relgan_screen"); output.mkdir(parents=True, exist_ok=True)
    rows = []; model_payload = {}
    for model in MODELS:
        real = load_json(baseline / model / "seed_42" / "metrics.json")
        relgan = load_json(augmented / model / "seed_42" / "metrics.json")
        metrics = {}
        for metric in METRICS:
            values = {"real_only": real[metric], "relgan_10_percent": relgan[metric],
                      "delta": relgan[metric] - real[metric]}
            metrics[metric] = values; rows.append({"model": model, "metric": metric, **values})
        model_payload[model] = {"seed": 42, "metrics": metrics,
                                "improved_all_metrics": all(metrics[m]["delta"] > 0 for m in METRICS)}
    payload = {"project_version": 2, "generator": "relgan", "condition": "generated_10_percent",
               "screening_seed": 42, "promotion_rule": "strictly improve all five requested metrics",
               "promoted_to_confirmation": False, "models": model_payload}
    write_json(output / "summary.json", payload)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)

    x = np.arange(len(MODELS)); width = .36
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    labels = ("CNN", "LSTM", "Random Forest", "Linear SVM")
    for axis, metric in zip(axes, METRICS):
        real_values = [100 * model_payload[m]["metrics"][metric]["real_only"] for m in MODELS]
        aug_values = [100 * model_payload[m]["metrics"][metric]["relgan_10_percent"] for m in MODELS]
        axis.bar(x - width/2, real_values, width, color="#4a5568", label="Real only")
        axis.bar(x + width/2, aug_values, width, color="#2f855a", label="RelGAN 10%")
        axis.set_xticks(x, labels, rotation=20, ha="right"); axis.set_title(metric.replace("_", " ").upper())
        axis.set_ylabel("Score (%)"); axis.grid(axis="y", alpha=.25)
        low = min(real_values + aug_values); high = max(real_values + aug_values)
        axis.set_ylim(max(0, low - max(.3, high-low)), min(100, high + max(.3, high-low)))
    axes[-1].axis("off"); handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(.92, .09), frameon=False)
    fig.suptitle("RelGAN 10% augmentation screen — seed 42", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, .02, 1, .95)); fig.savefig(output / "all_metrics_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "all_metrics_comparison.pdf", bbox_inches="tight"); plt.close(fig)

    matrix = np.asarray([[100 * model_payload[model]["metrics"][metric]["delta"]
                          for metric in METRICS] for model in MODELS])
    limit = max(.01, float(np.abs(matrix).max())); fig, axis = plt.subplots(figsize=(9.5, 4.5))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(METRICS)), [m.replace("_", " ").upper() for m in METRICS])
    axis.set_yticks(np.arange(len(MODELS)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row,column]:+.3f}", ha="center", va="center", fontsize=9)
    axis.set_title("RelGAN 10% minus matched real-only (percentage points)", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Change (percentage points)"); fig.tight_layout()
    fig.savefig(output / "delta_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "delta_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
