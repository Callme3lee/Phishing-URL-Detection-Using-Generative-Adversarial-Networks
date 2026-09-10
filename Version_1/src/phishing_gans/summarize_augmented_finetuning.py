from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import save_json
from .summarize_robust_augmentation import METRICS, result_root, summary


DOSES = (10, 20, 30)
SOURCES = ("url_dataset", "new_data_urls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize augmented-pretraining then real-only CNN fine-tuning."
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/robust_augmentation")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["test"]


def load_external(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    args = parse_args()
    real_balanced = {}
    real_external = {source: {} for source in SOURCES}
    fine_balanced = {dose: {} for dose in DOSES}
    fine_external = {
        dose: {source: {} for source in SOURCES} for dose in DOSES
    }
    for seed in args.seeds:
        baseline_root = result_root(args.experiment_root, seed) / "real_only" / "cnn"
        real_balanced[seed] = load_metrics(baseline_root / "metrics.json")
        for source in SOURCES:
            real_external[source][seed] = load_external(
                baseline_root / f"calibrated_external_{source}.json"
            )
        for dose in DOSES:
            root = (
                args.experiment_root
                / "finetuned"
                / f"seed{seed}"
                / f"lsgan_quality_{dose}_then_real"
                / "cnn"
            )
            fine_balanced[dose][seed] = load_metrics(root / "metrics.json")
            for source in SOURCES:
                fine_external[dose][source][seed] = load_external(
                    root / f"calibrated_external_{source}.json"
                )

    real_macro_by_metric = {
        metric: {
            seed: float(
                np.mean([real_external[source][seed][metric] for source in SOURCES])
            )
            for seed in args.seeds
        }
        for metric in METRICS
    }
    aggregate = {}
    rows = []
    for dose in DOSES:
        balanced = {}
        source_metrics = {}
        source_macro = {}
        for metric in METRICS:
            values = [fine_balanced[dose][seed][metric] for seed in args.seeds]
            changes = [
                fine_balanced[dose][seed][metric] - real_balanced[seed][metric]
                for seed in args.seeds
            ]
            balanced[metric] = summary(values)
            balanced[metric]["paired_change_vs_real"] = summary(changes)
            macro_values = [
                float(
                    np.mean(
                        [fine_external[dose][source][seed][metric] for source in SOURCES]
                    )
                )
                for seed in args.seeds
            ]
            macro_changes = [
                macro_values[index] - real_macro_by_metric[metric][seed]
                for index, seed in enumerate(args.seeds)
            ]
            source_macro[metric] = summary(macro_values)
            source_macro[metric]["paired_change_vs_real"] = summary(macro_changes)
        for source in SOURCES:
            source_metrics[source] = {
                metric: summary(
                    [fine_external[dose][source][seed][metric] for seed in args.seeds]
                )
                for metric in METRICS
            }
        aggregate[str(dose)] = {
            "balanced_test": balanced,
            "sources": source_metrics,
            "source_macro": source_macro,
        }
        rows.append(
            {
                "dose_percent": dose,
                "balanced_f1_mean": balanced["f1"]["mean"],
                "balanced_f1_change_mean": balanced["f1"]["paired_change_vs_real"]["mean"],
                "balanced_roc_auc_mean": balanced["roc_auc"]["mean"],
                "macro_f1_mean": source_macro["f1"]["mean"],
                "macro_f1_change_mean": source_macro["f1"]["paired_change_vs_real"]["mean"],
                "macro_roc_auc_mean": source_macro["roc_auc"]["mean"],
                "macro_roc_auc_change_mean": source_macro["roc_auc"][
                    "paired_change_vs_real"
                ]["mean"],
            }
        )

    best_balanced = max(DOSES, key=lambda dose: aggregate[str(dose)]["balanced_test"]["f1"]["mean"])
    best_macro_f1 = max(DOSES, key=lambda dose: aggregate[str(dose)]["source_macro"]["f1"]["mean"])
    best_macro_roc = max(DOSES, key=lambda dose: aggregate[str(dose)]["source_macro"]["roc_auc"]["mean"])
    real_summary = {
        "balanced_test": {
            metric: summary([real_balanced[seed][metric] for seed in args.seeds])
            for metric in METRICS
        },
        "source_macro": {
            metric: summary([real_macro_by_metric[metric][seed] for seed in args.seeds])
            for metric in METRICS
        },
    }
    output = {
        "experiment": "quality-LSGAN pretraining followed by real-only CNN fine-tuning",
        "seeds": args.seeds,
        "real_only": real_summary,
        "doses": aggregate,
        "best": {
            "balanced_f1_dose": best_balanced,
            "source_macro_f1_dose": best_macro_f1,
            "source_macro_roc_auc_dose": best_macro_roc,
        },
    }
    output_root = args.experiment_root / "finetuned"
    save_json(output_root / "summary.json", output)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(4)
    labels = ["Real only", "10% then real", "20% then real", "30% then real"]
    balanced_f1 = [
        100 * real_summary["balanced_test"]["f1"]["mean"],
        *[100 * aggregate[str(dose)]["balanced_test"]["f1"]["mean"] for dose in DOSES],
    ]
    macro_f1 = [
        100 * real_summary["source_macro"]["f1"]["mean"],
        *[100 * aggregate[str(dose)]["source_macro"]["f1"]["mean"] for dose in DOSES],
    ]
    macro_roc = [
        100 * real_summary["source_macro"]["roc_auc"]["mean"],
        *[100 * aggregate[str(dose)]["source_macro"]["roc_auc"]["mean"] for dose in DOSES],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, balanced_f1, color=("#4c78a8", "#72b7b2", "#54a24b", "#e45756"))
    axes[0].set_title("Balanced hostname-held-out test")
    axes[0].set_ylabel("Mean F1 (%)")
    axes[1].plot(x, macro_f1, marker="o", linewidth=2, label="Source-macro F1")
    axes[1].plot(x, macro_roc, marker="s", linewidth=2, label="Source-macro ROC-AUC")
    axes[1].set_title("Two represented sources, calibrated by validation hostnames")
    axes[1].set_ylabel("Mean score (%)")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("CNN: quality-LSGAN pretraining followed by real-only fine-tuning")
    figure.tight_layout()
    figure.savefig(output_root / "comparison.png", dpi=200, bbox_inches="tight")
    figure.savefig(output_root / "comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    for row in rows:
        print(
            f"{row['dose_percent']}%: balanced F1={100 * row['balanced_f1_mean']:.4f}% "
            f"({100 * row['balanced_f1_change_mean']:+.4f} pp), "
            f"macro F1={100 * row['macro_f1_mean']:.4f}% "
            f"({100 * row['macro_f1_change_mean']:+.4f} pp), "
            f"macro ROC={100 * row['macro_roc_auc_mean']:.4f}% "
            f"({100 * row['macro_roc_auc_change_mean']:+.4f} pp)"
        )


if __name__ == "__main__":
    main()
