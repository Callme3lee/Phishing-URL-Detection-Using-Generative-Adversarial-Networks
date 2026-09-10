from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF", "linear_svm": "SVM"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")
CONDITIONS = ("real_only", "coverage_only", "seq_10", "seq_20", "seq_30", "seq_40",
              "coverage_seq_10", "coverage_seq_20", "coverage_seq_30", "coverage_seq_40")
LABELS = ("Real", "Coverage", "S10", "S20", "S30", "S40", "Cov+S10", "Cov+S20", "Cov+S30", "Cov+S40")


def mean_sd(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values}


def internal_runs(root: Path, condition: str) -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in (42, 43, 44):
            path = root / model / f"seed_{seed}" / "metrics.json"
            value = load_json(path)
            rows.append({"condition": condition, "model": model, "seed": seed,
                         **{metric: float(value[metric]) for metric in METRICS}})
    return rows


def external_runs(path: Path, condition: str) -> list[dict]:
    summary = load_json(path)
    return [{"condition": condition, "model": row["model"], "seed": int(row["seed"]),
             **{metric: float(row["metrics"][metric]) for metric in METRICS}}
            for row in summary["per_run"]]


def summarize(rows: list[dict]) -> dict:
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for model in MODELS:
            chosen = [row for row in rows if row["condition"] == condition and row["model"] == model]
            output[condition][model] = {metric: mean_sd([row[metric] for row in chosen]) for metric in METRICS}
    return output


def paired_deltas(rows: list[dict], baseline: str, condition: str) -> dict:
    lookup = {(row["condition"], row["model"], row["seed"]): row for row in rows}
    output = {}
    for model in MODELS:
        output[model] = {}
        for metric in METRICS:
            values = [lookup[(condition, model, seed)][metric] - lookup[(baseline, model, seed)][metric]
                      for seed in (42, 43, 44)]
            output[model][metric] = {**mean_sd(values), "same_direction": len(set(np.sign(values))) == 1}
    return output


def graph(summary: dict, title: str, output: Path) -> None:
    x = np.arange(len(CONDITIONS)); colors = plt.cm.tab10(np.linspace(0, 1, len(CONDITIONS)))
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        for offset, model in enumerate(MODELS):
            means = [summary[c][model][metric]["mean"] * 100 for c in CONDITIONS]
            axis.plot(x, means, marker="o", linewidth=1.7, label=DISPLAY[model])
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, LABELS, rotation=35, ha="right"); axis.grid(alpha=.25)
        if metric not in {"false_positive_rate", "false_negative_rate"}: axis.set_ylim(0, 100)
    axes.ravel()[-1].axis("off"); handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5); fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, .06, 1, .95)); fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main() -> None:
    output = Path("artifacts/results/gan_augmentation_v1/summary"); output.mkdir(parents=True, exist_ok=True)
    internal = internal_runs(Path("artifacts/baselines"), "real_only")
    internal += internal_runs(Path("artifacts/experiments/legitimate_coverage_v1"), "coverage_only")
    for base, prefix in (("urls_new", "seq"), ("coverage_plus", "coverage_seq")):
        for pct in (10, 20, 30, 40):
            internal += internal_runs(Path(f"artifacts/experiments/gan_augmentation_v1/seqgan/{base}/pct_{pct:02d}"), f"{prefix}_{pct}")
    evaluations = {"internal": internal}
    reference_paths = {
        "topdomains_phishtank": (
            Path("artifacts/results/external/2026-08-11_topdomains_phishtank/summary.json"),
            Path("artifacts/results/legitimate_coverage_v1/external_topdomains_phishtank/coverage/summary.json"),
        ),
        "tranco_openphish": (
            Path("artifacts/results/legitimate_coverage_v1/external_openphish_tranco/original/summary.json"),
            Path("artifacts/results/legitimate_coverage_v1/external_openphish_tranco/coverage/summary.json"),
        ),
    }
    for snapshot, (original_path, coverage_path) in reference_paths.items():
        rows = external_runs(original_path, "real_only")
        rows += external_runs(coverage_path, "coverage_only")
        for base, prefix in (("urls_new", "seq"), ("coverage_plus", "coverage_seq")):
            for pct in (10, 20, 30, 40):
                rows += external_runs(Path(f"artifacts/results/gan_augmentation_v1/seqgan/{base}/pct_{pct:02d}/{snapshot}/summary.json"), f"{prefix}_{pct}")
        evaluations[snapshot] = rows
    payload = {"project_version": 3, "metrics": list(METRICS), "conditions": list(CONDITIONS),
               "gan_audits": {name: load_json(Path(f"data/generated/{name}/audit.json")) for name in ("seqgan", "relgan")},
               "quality_audits": {name: load_json(Path(f"artifacts/results/gan_quality/{name}/quality.json")) for name in ("seqgan", "relgan")},
               "evaluations": {}, "paired_deltas": {}}
    for name, rows in evaluations.items():
        payload["evaluations"][name] = summarize(rows)
        payload["paired_deltas"][name] = {}
        for condition in CONDITIONS[2:]:
            baseline = "coverage_only" if condition.startswith("coverage_") else "real_only"
            payload["paired_deltas"][name][condition] = paired_deltas(rows, baseline, condition)
        graph(payload["evaluations"][name], f"Version 3 GAN augmentation — {name.replace('_', ' ')}", output / f"{name}_all_metrics")
    write_json(output / "summary.json", payload)
    with (output / "all_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("evaluation", "condition", "model", "metric", "mean", "sd")); writer.writeheader()
        for evaluation, conditions in payload["evaluations"].items():
            for condition, models in conditions.items():
                for model, metrics in models.items():
                    for metric, values in metrics.items(): writer.writerow({"evaluation": evaluation, "condition": condition,
                        "model": model, "metric": metric, "mean": values["mean"], "sd": values["sd"]})
    lines = ["# Version 3 Controlled GAN Augmentation Study", "",
             "All rows use the frozen URLs_New split, seeds 42/43/44, unchanged classifiers, and unchanged validation-selected thresholds. Synthetic percentages are generated/final-training fractions.", "",
             "SeqGAN passed the strict augmentation gate. RelGAN did not produce enough TLD-like, non-overlapping URLs for even 10%, so no RelGAN classifier condition was fabricated by weakening filters or recycling rows.", "",
             "Paired seed deltas are stored in `summary.json`. With only three seeds they are descriptive, not strong significance evidence.", "",
             "## Files", "", "- `all_results.csv`: tidy mean/SD results.",
             "- `*_all_metrics.png` / `.pdf`: every metric shown separately.",
             "- Generator and quality audits are embedded in `summary.json`.", "",
             "## Interpretation", "", "A higher uniqueness score alone does not prove useful augmentation. Use the unchanged real external sets to judge recall/FNR and legitimate FPR together. A later-collected untouched snapshot is still required for temporal confirmation."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "evaluations": list(evaluations)}, indent=2))


if __name__ == "__main__":
    main()
