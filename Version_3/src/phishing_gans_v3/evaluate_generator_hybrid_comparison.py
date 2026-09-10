from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics, select_threshold
from .url_data import encode_url, psl_aware_lexical_features


CONDITIONS = {
    "real_only": ("Real-only HYB-MS", Path("artifacts/experiments/hybrid_cnn_coral_v1")),
    "seqgan_10pct": (
        "HYB-MS + 10% SeqGAN",
        Path("artifacts/experiments/generator_hybrid_comparison_v1/seqgan/gan_10pct/hyb_ms"),
    ),
    "ms_hardgan_c_10pct": (
        "HYB-MS + 10% MS-HardGAN-C",
        Path("artifacts/experiments/fair_hybrid_gan_v1/gan_10pct/hyb_ms"),
    ),
}

DATASETS = {
    "calibration": (
        Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv"),
        Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"),
    ),
    "external_10k": (
        Path("data/external/2026-08-12-large-frozen/external_urls.csv"),
        Path("data/external/2026-08-12-large-frozen/collection_report.json"),
    ),
    "external_30k": (
        Path("data/external/2026-08-13-imbalanced-30k/external_urls.csv"),
        Path("data/external/2026-08-13-imbalanced-30k/collection_report.json"),
    ),
}

METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled SeqGAN vs MS-HardGAN-C HYB-MS evaluation.")
    parser.add_argument(
        "--internal-root", type=Path,
        default=Path("data/experiments/hybrid_cnn_coral_v1/processed"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/results/generator_hybrid_comparison_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--seqgan-root", type=Path,
        default=Path("artifacts/experiments/generator_hybrid_comparison_v1/seqgan/gan_10pct/hyb_ms"),
    )
    parser.add_argument("--matched-generator-training", action="store_true")
    return parser.parse_args()


def frozen_rows(path: Path, report_path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    report = load_json(report_path)
    if sha256_file(path) != report["dataset_sha256"]:
        raise RuntimeError(f"Frozen hash mismatch: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != int(report["rows"]):
        raise RuntimeError(f"Frozen row-count mismatch: {path}")
    return rows, labels, report


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "values": [float(value) for value in values],
    }


def main() -> None:
    args = parse_args()
    device = require_cuda()
    conditions = dict(CONDITIONS)
    conditions["seqgan_10pct"] = ("HYB-MS + 10% SeqGAN", args.seqgan_root)
    datasets: dict[str, tuple[list[str], np.ndarray]] = {}
    identities: dict[str, dict] = {}
    for name, (path, report_path) in DATASETS.items():
        rows, labels, report = frozen_rows(path, report_path)
        datasets[name] = ([row["url"] for row in rows], labels)
        identities[name] = {
            "path": str(path), "sha256": report["dataset_sha256"], "rows": len(rows),
            "legitimate_rows": int((labels == 0).sum()), "phishing_rows": int((labels == 1).sum()),
        }
    with (args.internal_root / "test_manifest.csv").open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        internal_rows = list(csv.DictReader(handle))
    internal_labels = np.load(args.internal_root / "test_labels.npy")
    datasets["internal_test"] = ([row["url"] for row in internal_rows], internal_labels)
    identities["internal_test"] = {
        "path": str(args.internal_root / "test_manifest.csv"),
        "sha256": sha256_file(args.internal_root / "test_manifest.csv"),
        "rows": len(internal_rows),
        "legitimate_rows": int((internal_labels == 0).sum()),
        "phishing_rows": int((internal_labels == 1).sum()),
    }

    raw_features = {
        name: np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
        for name, (urls, _) in datasets.items()
    }
    results = {}
    saved_arrays = {}
    for key, (display, root) in conditions.items():
        seed_scores: dict[int, dict[str, np.ndarray]] = {}
        checkpoints = []
        per_seed = []
        for seed in (42, 43, 44):
            checkpoint_path = root / f"seed_{seed}" / "checkpoint.pt"
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            center = np.asarray(saved["lexical_scaler_center"], dtype=np.float32)
            scale_values = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
            scale_values = np.where(scale_values == 0, 1.0, scale_values)
            model = create_model(saved).to(device)
            model.load_state_dict(saved["state_dict"])
            seed_scores[seed] = {}
            for dataset_name, (urls, _) in datasets.items():
                tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
                lexical = ((raw_features[dataset_name] - center) / scale_values).astype(np.float32)
                seed_scores[seed][dataset_name] = score(model, tokens, lexical, args.batch_size, device)
            selected = select_threshold(
                datasets["calibration"][1], seed_scores[seed]["calibration"], maximum_fpr=0.01
            )
            threshold = float(selected["threshold"])
            per_seed.append({
                "seed": seed,
                "threshold": threshold,
                "threshold_selection": selected,
                "metrics": {
                    name: classification_metrics(labels, seed_scores[seed][name], threshold)
                    for name, (_, labels) in datasets.items() if name != "calibration"
                },
            })
            checkpoints.append({
                "seed": seed, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)
            })
            del model
            gc.collect(); torch.cuda.empty_cache()

        ensemble_scores = {
            name: np.mean(np.stack([seed_scores[seed][name] for seed in (42, 43, 44)]), axis=0)
            for name in datasets
        }
        ensemble_selection = select_threshold(
            datasets["calibration"][1], ensemble_scores["calibration"], maximum_fpr=0.01
        )
        ensemble_threshold = float(ensemble_selection["threshold"])
        ensemble_metrics = {
            name: classification_metrics(labels, ensemble_scores[name], ensemble_threshold)
            for name, (_, labels) in datasets.items() if name != "calibration"
        }
        stability = {
            dataset_name: {
                metric: summarize([row["metrics"][dataset_name][metric] for row in per_seed])
                for metric in METRICS
            }
            for dataset_name in ("internal_test", "external_10k", "external_30k")
        }
        results[key] = {
            "display_name": display,
            "checkpoint_root": str(root),
            "checkpoints": checkpoints,
            "threshold_policy": "maximum calibration F1 subject to calibration FPR <= 1%; test sets never tune thresholds",
            "ensemble_threshold": ensemble_threshold,
            "ensemble_threshold_selection": ensemble_selection,
            "ensemble_metrics": ensemble_metrics,
            "per_seed": per_seed,
            "stability_mean_sd": stability,
        }
        for name, scores in ensemble_scores.items():
            saved_arrays[f"{key}_{name}_scores"] = scores
        print(json.dumps({
            "condition": key, "threshold": ensemble_threshold,
            "internal_f1": ensemble_metrics["internal_test"]["f1"],
            "external_10k_f1": ensemble_metrics["external_10k"]["f1"],
            "external_30k_f1": ensemble_metrics["external_30k"]["f1"],
        }), flush=True)

    deltas = {}
    baseline = results["real_only"]["ensemble_metrics"]
    for condition in ("seqgan_10pct", "ms_hardgan_c_10pct"):
        deltas[condition] = {
            dataset: {
                metric: float(results[condition]["ensemble_metrics"][dataset][metric] - baseline[dataset][metric])
                for metric in METRICS
            }
            for dataset in ("internal_test", "external_10k", "external_30k")
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_version": 3,
        "experiment": "controlled generator comparison on identical HYB-MS",
        "training_contract": {
            "real_rows": 363692, "synthetic_rows": 40410, "synthetic_fraction": 0.10,
            "seeds": [42, 43, 44], "architecture": "HYB-MS", "epochs_maximum": 8,
            "early_stopping_patience": 2, "sampler": "same source-weighted sampler with 10% synthetic exposure",
            "generator_training_rows_identical": bool(args.matched_generator_training),
            "generator_training_rows": 84280 if args.matched_generator_training else None,
            "only_intended_difference": (
                "generator architecture/objective and its resulting synthetic URL pool"
                if args.matched_generator_training
                else "SeqGAN versus MS-HardGAN-C synthetic URL pool"
            ),
        },
        "datasets": identities,
        "results": results,
        "deltas_vs_real_only": deltas,
        "interpretation_guard": "This compares downstream augmentation utility; it does not claim either generator predicts zero-day attacks.",
    }
    write_json(args.output_root / "comparison.json", payload)
    np.savez_compressed(args.output_root / "ensemble_scores.npz", **saved_arrays)

    with (args.output_root / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "dataset", "threshold", *METRICS])
        for condition, result in results.items():
            for dataset, metrics in result["ensemble_metrics"].items():
                writer.writerow([condition, dataset, result["ensemble_threshold"], *[metrics[m] for m in METRICS]])

    colors = {"real_only": "#64748B", "seqgan_10pct": "#3B82F6", "ms_hardgan_c_10pct": "#E76F51"}
    labels = [conditions[key][0] for key in conditions]
    datasets_to_plot = ("internal_test", "external_10k", "external_30k")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    metric_specs = (
        ("f1", "F1"), ("recall", "Recall"), ("precision", "Precision"),
        ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("false_negative_rate", "FNR"),
    )
    x = np.arange(len(datasets_to_plot)); width = 0.24
    for axis, (metric, title) in zip(axes.flat, metric_specs):
        for index, condition in enumerate(conditions):
            values = [results[condition]["ensemble_metrics"][dataset][metric] * 100 for dataset in datasets_to_plot]
            bars = axis.bar(x + (index - 1) * width, values, width, label=labels[index], color=colors[condition])
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{value:.1f}",
                          ha="center", va="bottom", fontsize=7, rotation=90)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, ("Internal", "External 10K", "External 30K"))
        axis.set_ylabel("Percent")
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0, 107 if metric != "false_negative_rate" else None)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("HYB-MS: Real-only vs 10% SeqGAN vs 10% MS-HardGAN-C",
                 fontsize=16, fontweight="bold")
    fig.savefig(args.output_root / "hybrid_generator_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_root / "hybrid_generator_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output_root), "conditions": list(results)}, indent=2))


if __name__ == "__main__":
    main()
