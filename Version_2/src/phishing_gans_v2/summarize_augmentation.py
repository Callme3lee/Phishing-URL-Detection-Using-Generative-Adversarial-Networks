from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def condition_dose(condition: str) -> int:
    match = re.search(r"(\d+)", condition)
    if not match:
        raise ValueError(f"Cannot extract augmentation dose from {condition!r}")
    return int(match.group(1))


def interval(values: np.ndarray) -> tuple[float, float] | None:
    mean = float(values.mean())
    if len(values) < 2:
        return None
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired real-only versus augmented analysis.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--augmentation-dir", type=Path, default=Path("artifacts/augmentation"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/augmentation"))
    args = parser.parse_args()
    baselines = {}
    for path in args.baseline_dir.glob("*/seed_*/metrics.json"):
        row = load_json(path); baselines[(row["model"], int(row["seed"]))] = row
    grouped = defaultdict(list)
    for path in args.augmentation_dir.glob("*/*/*/seed_*/metrics.json"):
        row = load_json(path); grouped[(row["model"], row["generator"], row["condition"])].append(row)
    if not grouped:
        raise RuntimeError("No completed augmented metrics found")
    output_rows = []; summary = {"project_version": 2, "paired_comparisons": {}}
    for key, runs in sorted(grouped.items()):
        model, generator, condition = key
        paired = [(run, baselines[(model, int(run["seed"]))]) for run in runs if (model, int(run["seed"])) in baselines]
        if not paired:
            continue
        comparison = {"model": model, "generator": generator, "condition": condition, "seeds": [int(a["seed"]) for a, _ in paired], "metrics": {}}
        for metric in METRICS:
            augmented = np.asarray([a[metric] for a, _ in paired]); real = np.asarray([b[metric] for _, b in paired]); changes = augmented - real
            confidence_interval = interval(changes)
            comparison["metrics"][metric] = {
                "real_mean": float(real.mean()), "augmented_mean": float(augmented.mean()),
                "mean_paired_change": float(changes.mean()),
                "paired_95_ci": list(confidence_interval) if confidence_interval else None,
                "paired_95_ci_status": "available" if confidence_interval else "requires at least two paired seeds",
                "all_seeds_improved": bool(np.all(changes > 0)), "changes": changes.tolist(),
            }
            output_rows.append({"model": model, "generator": generator, "condition": condition, "metric": metric, **comparison["metrics"][metric]})
        summary["paired_comparisons"]["|".join(key)] = comparison
    args.output_dir.mkdir(parents=True, exist_ok=True); write_json(args.output_dir / "summary.json", summary)
    fields = ("model", "generator", "condition", "metric", "real_mean", "augmented_mean", "mean_paired_change", "paired_95_ci", "paired_95_ci_status", "all_seeds_improved", "changes")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in output_rows:
            writer.writerow({**row, "paired_95_ci": json.dumps(row["paired_95_ci"]), "changes": json.dumps(row["changes"])})
    for metric in METRICS:
        rows = [row for row in output_rows if row["metric"] == metric]
        labels = [f'{row["model"]}\n{row["generator"]}\n{row["condition"]}' for row in rows]
        x = np.arange(len(rows)); width = .36
        fig, axis = plt.subplots(figsize=(max(8, 1.25 * len(rows)), 5.2))
        axis.bar(x - width / 2, [100*r["real_mean"] for r in rows], width, label="Real only", color="#4a5568")
        axis.bar(x + width / 2, [100*r["augmented_mean"] for r in rows], width, label="Augmented", color="#2b6cb0")
        axis.set_xticks(x, labels, rotation=20, ha="right"); axis.set_ylabel("Score (%)")
        axis.set_title(f'Version 2 paired comparison — {metric.replace("_", " ").upper()}')
        axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
        fig.savefig(args.output_dir / f"comparison_{metric}.png", dpi=200, bbox_inches="tight")
        fig.savefig(args.output_dir / f"comparison_{metric}.pdf", bbox_inches="tight"); plt.close(fig)

    # Compact, presentation-ready dose-response plots: one classifier per figure.
    dose_dir = args.output_dir / "dose_response"
    heatmap_dir = args.output_dir / "delta_heatmaps"
    dose_dir.mkdir(parents=True, exist_ok=True); heatmap_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model"] for row in output_rows})
    generators = sorted({row["generator"] for row in output_rows})
    palette = {"gumbel_lsgan": "#2b6cb0", "seqgan": "#c05621", "relgan": "#2f855a"}
    markers = {"gumbel_lsgan": "o", "seqgan": "s", "relgan": "^"}
    for model in models:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.2)); axes = axes.ravel()
        model_rows = [row for row in output_rows if row["model"] == model]
        for index, metric in enumerate(METRICS):
            axis = axes[index]
            metric_rows = [row for row in model_rows if row["metric"] == metric]
            real_values = [row["real_mean"] for row in metric_rows]
            if real_values:
                axis.axhline(100 * real_values[0], color="#2d3748", linewidth=1.8,
                             linestyle="--", label="Real only")
            for generator in generators:
                selected = sorted(
                    (row for row in metric_rows if row["generator"] == generator),
                    key=lambda row: condition_dose(row["condition"]),
                )
                if not selected:
                    continue
                axis.plot([condition_dose(row["condition"]) for row in selected],
                          [100 * row["augmented_mean"] for row in selected],
                          marker=markers.get(generator, "o"), linewidth=2,
                          color=palette.get(generator), label=generator.replace("_", " ").upper())
            axis.set_title(metric.replace("_", " ").upper()); axis.set_xlabel("Generated share (%)")
            axis.set_ylabel("Score (%)"); axis.grid(alpha=.25); axis.set_xticks((10, 20, 30, 40))
        axes[-1].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(.94, .08), frameon=False)
        fig.suptitle(f"{model.upper()} — real-only versus GAN augmentation", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, .02, 1, .95))
        fig.savefig(dose_dir / f"{model}_all_metrics.png", dpi=220, bbox_inches="tight")
        fig.savefig(dose_dir / f"{model}_all_metrics.pdf", bbox_inches="tight"); plt.close(fig)

    # A delta heatmap makes positive/negative effects immediately visible.
    condition_labels = [f"{generator.replace('_', ' ')} {dose}%" for generator in generators for dose in (10, 20, 30, 40)]
    condition_keys = [(generator, dose) for generator in generators for dose in (10, 20, 30, 40)]
    for metric in METRICS:
        matrix = np.full((len(models), len(condition_keys)), np.nan)
        for row in (item for item in output_rows if item["metric"] == metric):
            key = (row["generator"], condition_dose(row["condition"]))
            if key in condition_keys:
                matrix[models.index(row["model"]), condition_keys.index(key)] = 100 * row["mean_paired_change"]
        finite = np.abs(matrix[np.isfinite(matrix)])
        limit = max(float(finite.max()) if finite.size else .01, .01)
        fig, axis = plt.subplots(figsize=(11.5, 4.6))
        image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        for row_index in range(len(models)):
            for column_index in range(len(condition_keys)):
                value = matrix[row_index, column_index]
                axis.text(column_index, row_index, "—" if np.isnan(value) else f"{value:+.3f}",
                          ha="center", va="center", fontsize=8)
        axis.set_xticks(np.arange(len(condition_labels)), condition_labels, rotation=28, ha="right")
        axis.set_yticks(np.arange(len(models)), [model.upper() for model in models])
        axis.set_title(f"Augmented minus real-only — {metric.replace('_', ' ').upper()} (percentage points)")
        fig.colorbar(image, ax=axis, label="Paired mean change (percentage points)")
        fig.tight_layout(); fig.savefig(heatmap_dir / f"delta_{metric}.png", dpi=220, bbox_inches="tight")
        fig.savefig(heatmap_dir / f"delta_{metric}.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
