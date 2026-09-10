"""Generate Chapter 6 graphs directly from stored experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPORT_ROOT.parent
OUT = REPORT_ROOT / "assets" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#204a87",
    "cyan": "#4c9fba",
    "orange": "#d97706",
    "red": "#b13b3b",
    "green": "#397d54",
    "grey": "#5f6b73",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def baseline_comparison() -> None:
    model_paths = {
        "CNN": "cnn",
        "LSTM": "lstm",
        "RNN": "rnn",
        "RF": "random_forest",
        "SVM": "linear_svm",
    }
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    values = []
    for _, folder in model_paths.items():
        rows = []
        for seed in (42, 43, 44):
            path = PROJECT_ROOT / "Version_3" / "artifacts" / "baselines" / folder / f"seed_{seed}" / "metrics.json"
            rows.append(json.loads(path.read_text()))
        values.append([100 * np.mean([row[m] for row in rows]) for m in metrics])

    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["grey"]]
    for index, ((name, _), row, colour) in enumerate(zip(model_paths.items(), values, palette)):
        ax.bar(x + (index - 2) * width, row, width, label=name, color=colour)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(45, 101)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", frameon=False)
    save(fig, "v3_baseline_comparison")


def gan_quality_comparison() -> None:
    names = ["SeqGAN", "URL-GAN", "RelGAN"]
    acceptance = [69.94, 47.87, 2.57]
    coverage = [94.0, 90.0, 86.0]
    challenging = [10.92, 5.06, 0.085]
    source_auc = [73.57, 88.17, 98.88]
    x = np.arange(len(names))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].bar(x - width, acceptance, width, label="Strict acceptance", color=COLORS["blue"])
    axes[0].bar(x, coverage, width, label="Real-cluster coverage", color=COLORS["green"])
    axes[0].bar(x + width, challenging, width, label="Majority challenging", color=COLORS["orange"])
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(names, source_auc, color=[COLORS["blue"], COLORS["cyan"], COLORS["grey"]])
    axes[1].axhline(50, color=COLORS["red"], linestyle="--", linewidth=1.4, label="Indistinguishable target")
    axes[1].set_ylabel("Source-discrimination ROC-AUC (%)")
    axes[1].set_ylim(45, 102)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    save(fig, "gan_quality_comparison")


def seqgan_f1_comparison() -> None:
    ratios = ["Real-only", "10%", "20%", "30%", "40%"]
    series = {
        "CNN": [0.9736, 0.9764, 0.9758, 0.9751, 0.9724],
        "LSTM": [0.9756, 0.9722, 0.9744, 0.9750, 0.9731],
        "RNN": [0.9679, 0.9679, 0.9532, 0.9639, 0.9623],
        "RF": [0.8667, 0.8625, 0.8576, 0.8508, 0.8433],
        "SVM": [0.6835, 0.6924, 0.6935, 0.6920, 0.6887],
    }
    palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["orange"], COLORS["grey"]]
    x = np.arange(len(ratios))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for (name, values), colour in zip(series.items(), palette):
        ax.plot(x, [100 * value for value in values], marker="o", linewidth=2, label=name, color=colour)
    ax.set_ylabel("F1-score (%)")
    ax.set_xticks(x, ratios)
    ax.set_ylim(65, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, frameon=False)
    save(fig, "seqgan_internal_metrics")


def threshold_tradeoff() -> None:
    path = PROJECT_ROOT / "Version_3" / "artifacts" / "results" / "threshold_sets_v2" / "all_metrics.csv"
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["system"] == "single_stage"]
    labels = [r["name"] for r in rows]
    recall = [100 * float(r["recall"]) for r in rows]
    fpr = [100 * float(r["false_positive_rate"]) for r in rows]
    fnr = [100 * float(r["false_negative_rate"]) for r in rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(x, recall, marker="o", linewidth=2.3, label="Recall", color=COLORS["blue"])
    ax.plot(x, fnr, marker="s", linewidth=2.3, label="FNR", color=COLORS["red"])
    ax.plot(x, fpr, marker="^", linewidth=2.3, label="FPR", color=COLORS["orange"])
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    save(fig, "threshold_tradeoff")


def external_summary() -> None:
    sources = [
        (
            "30K, 5% phishing",
            PROJECT_ROOT / "Version_3/artifacts/results/final_confirmation_2026-08-22_30k_5pct/results.json",
            "json",
        ),
        (
            "PhishTank 10K, 5%",
            PROJECT_ROOT / "Version_3/artifacts/results/phishtank_only_fresh_2026-08-28_5pct/research_pair/results.json",
            "json",
        ),
        (
            "Final 6K, 5%",
            PROJECT_ROOT / "Version_3/artifacts/results/final_browser_evaluation_2026-09-01_v1/all_metrics.csv",
            "csv",
        ),
        (
            "Post-freeze 10K, 3%",
            PROJECT_ROOT / "Version_3/artifacts/results/postfreeze_10k_3pct_2026-09-01_v1/all_metrics.csv",
            "csv",
        ),
    ]
    collected: dict[str, list[dict[str, float]]] = {"real_only": [], "gan_10pct": []}
    labels = []
    for label, path, kind in sources:
        labels.append(label)
        if kind == "csv":
            with path.open(newline="") as handle:
                indexed = {r["condition"]: r for r in csv.DictReader(handle)}
            for condition in collected:
                collected[condition].append({k: float(indexed[condition][k]) for k in ("f1", "false_positive_rate", "false_negative_rate")})
        else:
            data = json.loads(path.read_text())
            for condition in collected:
                row = data["conditions"][condition].get("metrics", data["conditions"][condition])
                collected[condition].append({k: float(row[k]) for k in ("f1", "false_positive_rate", "false_negative_rate")})

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2))
    for offset, condition, name, colour in [
        (-width / 2, "real_only", "Real-only", COLORS["grey"]),
        (width / 2, "gan_10pct", "GAN +10%", COLORS["blue"]),
    ]:
        axes[0].bar(x + offset, [100 * r["f1"] for r in collected[condition]], width, label=name, color=colour)
    axes[0].set_ylabel("F1-score (%)")
    axes[0].set_ylim(70, 90)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for axis, metric, title in [
        (axes[1], "false_negative_rate", "False-negative rate (FNR)"),
        (axes[2], "false_positive_rate", "False-positive rate (FPR)"),
    ]:
        axis.plot(
            x,
            [100 * r[metric] for r in collected["real_only"]],
            marker="s",
            linestyle="--",
            linewidth=2.2,
            label="Real-only (dashed squares)",
            color=COLORS["grey"],
        )
        axis.plot(
            x,
            [100 * r[metric] for r in collected["gan_10pct"]],
            marker="o",
            linestyle="-",
            linewidth=2.2,
            label="GAN +10% (solid circles)",
            color=COLORS["blue"],
        )
        axis.set_title(title)
        axis.set_ylabel("Rate (%)")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    save(fig, "external_real_only_vs_gan10")


def external_model_comparison() -> None:
    path = PROJECT_ROOT / "Version_3/artifacts/results/all_models_equal_calibration_v1/ranked.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {
        "experiments__multisource__baselines__v1__cnn": "CNN",
        "experiments__multisource__baselines__v1__lstm": "LSTM",
        "experiments__multisource__baselines__v1__rnn": "RNN",
        "experiments__multisource__baselines__v1__random__forest": "RF",
        "experiments__multisource__baselines__v1__linear__svm": "SVM",
        "experiments__hybrid__cnn__coral__v1": "HYB-MS",
        "experiments__fair__hybrid__gan__v1__gan__10pct__hyb__ms": "HYB-MS +10%",
    }
    indexed = {r["id"]: r for r in rows}
    labels = list(wanted.values())
    selected = [indexed[key] for key in wanted]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    width = 0.25
    for offset, metric, label, colour in [
        (-width, "external_10k_f1", "F1", COLORS["blue"]),
        (0, "external_10k_roc_auc", "ROC-AUC", COLORS["green"]),
        (width, "external_10k_pr_auc", "PR-AUC", COLORS["orange"]),
    ]:
        axes[0].bar(x + offset, [100 * float(r[metric]) for r in selected], width, label=label, color=colour)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)

    for metric, label, colour, marker in [
        ("external_10k_false_positive_rate", "FPR", COLORS["orange"], "o"),
        ("external_10k_false_negative_rate", "FNR", COLORS["red"], "s"),
    ]:
        axes[1].plot(x, [100 * float(r[metric]) for r in selected], label=label, color=colour, marker=marker, linewidth=2)
    axes[1].set_ylabel("Error rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    save(fig, "external_classifier_comparison")


def final_30k_model_comparison() -> None:
    path = PROJECT_ROOT / "Version_3/artifacts/results/all_classifiers_final_30k_5pct_v2/all_metrics.csv"
    with path.open(newline="") as handle:
        indexed = {row["model"]: row for row in csv.DictReader(handle)}
    order = [
        ("cnn_ms", "CNN-MS"),
        ("lstm_ms", "LSTM-MS"),
        ("rnn_ms", "RNN-MS"),
        ("rf_ms", "RF-MS"),
        ("svm_ms", "SVM-MS"),
        ("hyb_ms_real_only", "HYB-MS"),
        ("hyb_ms_gan_10pct", "HYB-MS +10%"),
    ]
    rows = [indexed[key] for key, _ in order]
    labels = [label for _, label in order]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for offset, metric, label, colour in [
        (-width, "f1", "F1", COLORS["blue"]),
        (0, "roc_auc", "ROC-AUC", COLORS["green"]),
        (width, "pr_auc", "PR-AUC", COLORS["orange"]),
    ]:
        axes[0].bar(x + offset, [100 * float(row[metric]) for row in rows], width, label=label, color=colour)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)

    for metric, label, colour, marker in [
        ("false_positive_rate", "FPR", COLORS["orange"], "o"),
        ("false_negative_rate", "FNR", COLORS["red"], "s"),
    ]:
        axes[1].plot(x, [100 * float(row[metric]) for row in rows], label=label, color=colour, marker=marker, linewidth=2)
    axes[1].set_ylabel("Error rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    save(fig, "final_30k_all_classifiers")


def extension_shared_content_policy() -> None:
    """Plot the registered before/after browser-policy audit."""
    path = PROJECT_ROOT / "Version_3/Final Model Evaluation/Supervisor_2026-09-07/results/extension_google_shared_content_policy_v3.json"
    data = json.loads(path.read_text())
    conditions = [
        ("Blanket service trust", data["old_blanket_allow"], COLORS["grey"]),
        ("Revised shared-content policy", data["new_capped_shared_content_policy"], COLORS["blue"]),
    ]
    score_metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    error_metrics = [("false_positive_rate", "FPR"), ("false_negative_rate", "FNR")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    x = np.arange(len(score_metrics))
    width = 0.34
    for index, (name, row, colour) in enumerate(conditions):
        bars = axes[0].bar(x + (index - 0.5) * width,
                           [100 * row[key] for key, _ in score_metrics],
                           width, label=name, color=colour)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(70, 91)
    axes[0].set_xticks(x, [label for _, label in score_metrics])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(error_metrics))
    for index, (name, row, colour) in enumerate(conditions):
        bars = axes[1].bar(x + (index - 0.5) * width,
                           [100 * row[key] for key, _ in error_metrics],
                           width, label=name, color=colour)
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1].set_ylabel("Error rate (%)")
    axes[1].set_ylim(0, 25)
    axes[1].set_xticks(x, [label for _, label in error_metrics])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "extension_shared_content_policy")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    baseline_comparison()
    gan_quality_comparison()
    seqgan_f1_comparison()
    threshold_tradeoff()
    external_summary()
    external_model_comparison()
    final_30k_model_comparison()
    extension_shared_content_policy()
