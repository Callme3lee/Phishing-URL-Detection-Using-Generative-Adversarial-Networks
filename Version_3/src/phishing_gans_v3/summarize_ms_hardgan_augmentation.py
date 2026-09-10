from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "RF", "linear_svm": "SVM"}
CONDITIONS = ("real_only", "gan_10pct", "gan_20pct", "gan_30pct", "gan_40pct")
CONDITION_DISPLAY = {"real_only": "Real only", "gan_10pct": "+10%", "gan_20pct": "+20%", "gan_30pct": "+30%", "gan_40pct": "+40%"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")
INTERNAL_ROOTS = {
    "real_only": Path("artifacts/experiments/multisource_baselines_v1"),
    **{f"gan_{pct}pct": Path(f"artifacts/experiments/ms_hardgan_augmentation_v2/pct_{pct}") for pct in (10, 20, 30, 40)},
}
EXTERNAL_SUMMARIES = {
    "real_only": Path("artifacts/results/multisource_baselines_v1/external_large_2026-08-12/summary.json"),
    **{f"gan_{pct}pct": Path(f"artifacts/results/ms_hardgan_augmentation_v2/pct_{pct}/external_10k/summary.json") for pct in (10, 20, 30, 40)},
}
OUTPUT = Path("artifacts/results/ms_hardgan_augmentation_v2/comparison")


def stats(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)), "values": values}


def load_internal() -> list[dict]:
    rows = []
    for condition, root in INTERNAL_ROOTS.items():
        for model in MODELS:
            for seed in (42, 43, 44):
                metrics = load_json(root / model / f"seed_{seed}" / "metrics.json")
                rows.append({"condition": condition, "model": model, "seed": seed,
                             **{metric: float(metrics[metric]) for metric in METRICS}})
    return rows


def load_external() -> list[dict]:
    rows = []
    for condition, path in EXTERNAL_SUMMARIES.items():
        report = load_json(path)
        for record in report["per_run"]:
            rows.append({"condition": condition, "model": record["model"], "seed": int(record["seed"]),
                         **{metric: float(record["metrics"][metric]) for metric in METRICS}})
    return rows


def group(rows: list[dict]) -> tuple[dict, dict]:
    grouped: dict = {}
    paired: dict = {}
    for condition in CONDITIONS:
        grouped[condition] = {}
        paired[condition] = {}
        for model in MODELS:
            selected = sorted((row for row in rows if row["condition"] == condition and row["model"] == model), key=lambda row: row["seed"])
            baseline = sorted((row for row in rows if row["condition"] == "real_only" and row["model"] == model), key=lambda row: row["seed"])
            grouped[condition][model] = {metric: stats([row[metric] for row in selected]) for metric in METRICS}
            paired[condition][model] = {
                metric: stats([row[metric] - base[metric] for row, base in zip(selected, baseline)])
                for metric in METRICS
            }
    return grouped, paired


def save_metric_csv(path: Path, scope: str, grouped: dict, paired: dict) -> None:
    fields = ["scope", "condition", "model", "metric", "mean", "sd", "paired_delta_vs_real_mean", "paired_delta_vs_real_sd"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            for model in MODELS:
                for metric in METRICS:
                    writer.writerow({"scope": scope, "condition": condition, "model": model, "metric": metric,
                                     "mean": grouped[condition][model][metric]["mean"],
                                     "sd": grouped[condition][model][metric]["standard_deviation"],
                                     "paired_delta_vs_real_mean": paired[condition][model][metric]["mean"],
                                     "paired_delta_vs_real_sd": paired[condition][model][metric]["standard_deviation"]})


def plot_scope(scope: str, grouped: dict) -> None:
    colors = {"cnn": "#4C78A8", "lstm": "#F58518", "rnn": "#54A24B", "random_forest": "#E45756", "linear_svm": "#72B7B2"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    x = np.arange(len(CONDITIONS))
    for axis, metric in zip(axes.ravel(), METRICS):
        for model in MODELS:
            means = [100 * grouped[c][model][metric]["mean"] for c in CONDITIONS]
            errors = [100 * grouped[c][model][metric]["standard_deviation"] for c in CONDITIONS]
            axis.errorbar(x, means, yerr=errors, marker="o", linewidth=2, capsize=3, label=DISPLAY[model], color=colors[model])
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, [CONDITION_DISPLAY[c] for c in CONDITIONS], rotation=20)
        axis.set_ylabel("Percent")
        axis.grid(alpha=.25)
        if scope == "internal" and metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(50, 101)
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc="center", ncol=1, frameon=False, fontsize=11)
    title = "Unchanged internal real test set" if scope == "internal" else "Frozen external 10K snapshot (5K legitimate + 5K phishing)"
    fig.suptitle(f"MS-HardGAN-C augmentation — {title}\nmean ± SD across seeds 42/43/44", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUTPUT / f"{scope}_all_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{scope}_all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_challenge(challenge: dict) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(CONDITIONS))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    for model, color in zip(MODELS, colors):
        means = [100 * challenge["grouped"][condition][model]["recall"]["mean"] for condition in CONDITIONS]
        errors = [100 * challenge["grouped"][condition][model]["recall"]["standard_deviation"] for condition in CONDITIONS]
        axis.errorbar(x, means, yerr=errors, marker="o", linewidth=2.2, capsize=3, label=DISPLAY[model], color=color)
    axis.set_xticks(x, [CONDITION_DISPLAY[c] for c in CONDITIONS])
    axis.set_ylabel("Phishing recall (%)")
    axis.set_ylim(45, 101)
    axis.grid(alpha=.25)
    axis.legend(ncol=5, loc="lower center")
    axis.set_title(f"Frozen MS-HardGAN-C challenge holdout ({challenge['rows']:,} phishing URLs)\nmean ± SD across seeds; higher means better recognition of this generator", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "synthetic_challenge_recall.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "synthetic_challenge_recall.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: dict) -> str:
    return f"{value['mean']:.4f} ± {value['standard_deviation']:.4f}"


def make_report(internal: dict, external: dict, challenge: dict) -> str:
    lines = [
        "# MS-HardGAN-C Controlled Augmentation Experiment", "",
        "All conditions use the same 363,692 real MS training rows, the same frozen validation/test splits, classifiers, seeds (42/43/44), and threshold policy. Synthetic percentages are fractions of the final training set.", "",
        "## Internal real test set", "",
        "| Model | Condition | F1 | ROC-AUC | Recall | FPR | FNR |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for condition in CONDITIONS:
            values = internal[condition][model]
            lines.append(f"| {DISPLAY[model]} | {CONDITION_DISPLAY[condition]} | {fmt(values['f1'])} | {fmt(values['roc_auc'])} | {fmt(values['recall'])} | {fmt(values['false_positive_rate'])} | {fmt(values['false_negative_rate'])} |")
    lines.extend(("", "## Frozen external 10K snapshot", "", "| Model | Condition | F1 | ROC-AUC | Recall | FPR | FNR |", "|---|---|---:|---:|---:|---:|---:|"))
    for model in MODELS:
        for condition in CONDITIONS:
            values = external[condition][model]
            lines.append(f"| {DISPLAY[model]} | {CONDITION_DISPLAY[condition]} | {fmt(values['f1'])} | {fmt(values['roc_auc'])} | {fmt(values['recall'])} | {fmt(values['false_positive_rate'])} | {fmt(values['false_negative_rate'])} |")
    lines.extend(("", "## Frozen synthetic challenge", "", "| Model | Real only recall | +10% | +20% | +30% | +40% |", "|---|---:|---:|---:|---:|---:|"))
    for model in MODELS:
        vals = [fmt(challenge["grouped"][condition][model]["recall"]) for condition in CONDITIONS]
        lines.append(f"| {DISPLAY[model]} | " + " | ".join(vals) + " |")
    lines.extend((
        "", "## Interpretation", "",
        "- Augmentation improves recall on held-out URLs produced by the same GAN. This demonstrates learned generator-specific coverage, not zero-day prediction.",
        "- The external snapshot remains dominated by legitimate-class errors. High phishing recall is paired with an unacceptable FPR, so thresholded external F1/accuracy are not deployment-ready.",
        "- LSTM has the strongest external ranking among the neural models, but its frozen internal threshold is poorly transported to the shifted external distribution. ROC-AUC and FPR must therefore be interpreted together.",
        "- RF and SVM generally degrade internally as synthetic dose rises, indicating model-family interaction with the generated character distribution.",
        "- With only three seeds, mean ± SD and paired deltas are descriptive evidence; they are not strong grounds for a significance claim.",
        "", "## Conclusion", "",
        "MS-HardGAN-C augmentation succeeds at its narrow objective of increasing recognition of held-out generator samples. It does not improve overall real-world generalization on the frozen 10K snapshot and cannot yet be claimed to improve zero-day phishing detection. The next controlled step should address legitimate distribution coverage and threshold calibration using validation data only, while preserving this test set as untouched evidence.", "",
    ))
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    internal_rows = load_internal()
    external_rows = load_external()
    internal, internal_paired = group(internal_rows)
    external, external_paired = group(external_rows)
    challenge = load_json(Path("artifacts/results/ms_hardgan_augmentation_v2/synthetic_challenge.json"))
    report = {"project_version": 3, "conditions": list(CONDITIONS), "models": list(MODELS), "seeds": [42, 43, 44],
              "internal": internal, "internal_paired_delta_vs_real_only": internal_paired,
              "external_10k": external, "external_10k_paired_delta_vs_real_only": external_paired,
              "synthetic_challenge": challenge["grouped"],
              "interpretation_guardrail": "Synthetic challenge improvement is generator-specific and is not evidence of zero-day prediction."}
    write_json(OUTPUT / "summary.json", report)
    save_metric_csv(OUTPUT / "internal_metrics.csv", "internal", internal, internal_paired)
    save_metric_csv(OUTPUT / "external_10k_metrics.csv", "external_10k", external, external_paired)
    plot_scope("internal", internal)
    plot_scope("external", external)
    plot_challenge(challenge)
    (OUTPUT / "REPORT.md").write_text(make_report(internal, external, challenge), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "internal_runs": len(internal_rows), "external_runs": len(external_rows)}, indent=2))


if __name__ == "__main__":
    main()
