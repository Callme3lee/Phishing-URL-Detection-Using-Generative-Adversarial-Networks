from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics
from .url_data import encode_url, psl_aware_lexical_features


CONDITIONS = {
    "real_only": {
        "title": "Updated HYB-MS (real only)",
        "checkpoints": Path("artifacts/experiments/recent_update_gan10_v1/real_only/hyb_ms"),
        "calibration": Path("artifacts/results/updated_models_calibration_v1/real_only/calibration_analysis.json"),
    },
    "gan_10pct": {
        "title": "Updated HYB-MS (+10% GAN)",
        "checkpoints": Path("artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms"),
        "calibration": Path("artifacts/results/updated_models_calibration_v1/gan_10pct/calibration_analysis.json"),
    },
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the final hybrid candidates on the frozen long-URL stress set.")
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-20-long-url-raw/evaluation_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-20-long-url-raw/collection_report.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/external/2026-08-20-long-url-raw"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def threshold_at(calibration: Path, maximum_fpr: float = .005) -> float:
    report = load_json(calibration)
    for row in report["tradeoffs"]:
        if abs(float(row["maximum_fpr"]) - maximum_fpr) < 1e-12:
            return float(row["selection"]["threshold"])
    raise RuntimeError(f"No {maximum_fpr:.2%} calibration-FPR threshold in {calibration}")


def subgroup_metrics(rows, labels, scores, threshold):
    predictions = scores >= threshold
    groups = {}
    definitions = {
        "legitimate_all": np.asarray([int(row["label"]) == 0 for row in rows]),
        "legitimate_80_99": np.asarray([int(row["label"]) == 0 and 80 <= int(row["url_length"]) < 100 for row in rows]),
        "legitimate_100_149": np.asarray([int(row["label"]) == 0 and 100 <= int(row["url_length"]) < 150 for row in rows]),
        "legitimate_150_199": np.asarray([int(row["label"]) == 0 and 150 <= int(row["url_length"]) < 200 for row in rows]),
        "legitimate_200_plus": np.asarray([int(row["label"]) == 0 and int(row["url_length"]) >= 200 for row in rows]),
        "legitimate_query": np.asarray([int(row["label"]) == 0 and row["structure"] == "query" for row in rows]),
        "legitimate_deep_path": np.asarray([int(row["label"]) == 0 and row["structure"] == "deep_path" for row in rows]),
        "legitimate_familiar_domain": np.asarray([int(row["label"]) == 0 and int(row["prior_registered_domain_seen"]) == 1 for row in rows]),
        "legitimate_unseen_domain": np.asarray([int(row["label"]) == 0 and int(row["prior_registered_domain_seen"]) == 0 for row in rows]),
        "phishing_all": labels == 1,
    }
    for name, mask in definitions.items():
        count = int(mask.sum())
        if not count:
            continue
        if name.startswith("legitimate"):
            groups[name] = {"rows": count, "false_positive_rate": float(predictions[mask].mean()),
                            "mean_score": float(scores[mask].mean()), "median_score": float(np.median(scores[mask]))}
        else:
            groups[name] = {"rows": count, "recall": float(predictions[mask].mean()),
                            "false_negative_rate": float(1 - predictions[mask].mean()),
                            "mean_score": float(scores[mask].mean()), "median_score": float(np.median(scores[mask]))}
    return groups


def make_graphs(output: Path, results: dict, rows, labels):
    colors = ("#3B6EA8", "#E07A5F")
    keys = list(CONDITIONS)
    titles = [CONDITIONS[key]["title"] for key in keys]
    main = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, metric in zip(axes.ravel(), main):
        values = [100 * results[key]["metrics"][metric] for key in keys]
        bars = axis.bar(titles, values, color=colors, width=.62)
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_ylim(0, 100); axis.grid(axis="y", alpha=.25)
        axis.tick_params(axis="x", rotation=9)
        axis.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    fig.suptitle("New long-URL stress test — final hybrid candidates", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(output / "main_metrics.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, title in zip(axes, ("false_positive_rate", "false_negative_rate"),
                                   ("False alarms on legitimate URLs (FPR)", "Missed phishing URLs (FNR)")):
        values = [100 * results[key]["metrics"][metric] for key in keys]
        bars = axis.bar(titles, values, color=colors, width=.58)
        axis.set_title(title, fontweight="bold"); axis.grid(axis="y", alpha=.25)
        axis.tick_params(axis="x", rotation=8); axis.bar_label(bars, fmt="%.2f%%", padding=3)
        axis.set_ylim(0, max(values) * 1.25 + .5)
    fig.suptitle("The two browser errors we want to reduce", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output / "fpr_fnr.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    groups = ("legitimate_80_99", "legitimate_100_149", "legitimate_150_199", "legitimate_200_plus")
    labels_text = ("80–99", "100–149", "150–199", "200+")
    x = np.arange(len(groups)); width = .36
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for offset, key, color in zip((-.5, .5), keys, colors):
        values = [100 * results[key]["subgroups"].get(group, {}).get("false_positive_rate", 0) for group in groups]
        bars = axis.bar(x + offset * width, values, width, label=titles[keys.index(key)], color=color)
        axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    axis.set_xticks(x, labels_text); axis.set_xlabel("Legitimate URL length (characters)")
    axis.set_ylabel("False-positive rate"); axis.set_title("Long legitimate URLs are a deliberate stress test", fontweight="bold")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(output / "legitimate_length_false_positives.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    bins = np.linspace(0, 1, 31)
    for key, color in zip(keys, colors):
        scores = np.asarray(results[key]["scores"])
        axis.hist(scores[labels == 0], bins=bins, density=True, histtype="step", linewidth=2.2,
                  color=color, label=f"{CONDITIONS[key]['title']} — legitimate")
    axis.set_yscale("log"); axis.set_xlabel("Phishing score (0 = legitimate, 1 = phishing)")
    axis.set_ylabel("Density (log scale)"); axis.set_title("Scores assigned to long legitimate URLs", fontweight="bold")
    axis.grid(alpha=.2); axis.legend(); fig.tight_layout(); fig.savefig(output / "legitimate_score_distribution.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def paired_exact(labels: np.ndarray, first_scores: np.ndarray, first_threshold: float,
                 second_scores: np.ndarray, second_threshold: float) -> dict:
    first = first_scores >= first_threshold; second = second_scores >= second_threshold
    output = {}
    for name, mask in (("all_rows", np.ones(len(labels), dtype=bool)),
                       ("phishing_rows", labels == 1), ("legitimate_rows", labels == 0)):
        first_only = int((mask & (first == labels) & (second != labels)).sum())
        second_only = int((mask & (first != labels) & (second == labels)).sum())
        discordant = first_only + second_only
        output[name] = {
            "real_only_correct_gan_wrong": first_only,
            "real_only_wrong_gan_correct": second_only,
            "discordant_rows": discordant,
            "two_sided_exact_mcnemar_p_value": float(binomtest(min(first_only, second_only), discordant, .5).pvalue) if discordant else 1.0,
        }
    return output


def main():
    args = parse_args(); collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen dataset hash mismatch")
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    raw_lexical = np.stack([psl_aware_lexical_features(row["url"]) for row in rows]).astype(np.float32)
    device = require_cuda(); args.output.mkdir(parents=True, exist_ok=True); results = {}
    prediction_columns = {"url": [row["url"] for row in rows], "label": labels.tolist()}
    for key, condition in CONDITIONS.items():
        threshold = threshold_at(condition["calibration"]); seed_scores = []; checkpoints = []
        for seed in (42, 43, 44):
            path = condition["checkpoints"] / f"seed_{seed}" / "checkpoint.pt"
            saved = torch.load(path, map_location=device, weights_only=False)
            tokens = np.stack([encode_url(row["url"], saved["vocabulary"], 128) for row in rows])
            scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32); scale = np.where(scale == 0, 1, scale)
            lexical = ((raw_lexical - np.asarray(saved["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
            model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
            seed_scores.append(score(model, tokens, lexical, args.batch_size, device))
            checkpoints.append({"seed": seed, "path": str(path), "sha256": sha256_file(path)})
            del model; gc.collect(); torch.cuda.empty_cache()
        ensemble = np.mean(np.stack(seed_scores), axis=0)
        result = {"title": condition["title"], "threshold": threshold,
                  "threshold_policy": "frozen three-seed ensemble threshold chosen earlier at <=0.5% calibration FPR",
                  "checkpoints": checkpoints, "metrics": classification_metrics(labels, ensemble, threshold),
                  "subgroups": subgroup_metrics(rows, labels, ensemble, threshold), "scores": ensemble.tolist()}
        results[key] = result; prediction_columns[f"{key}_score"] = ensemble.tolist()
        prediction_columns[f"{key}_prediction"] = (ensemble >= threshold).astype(np.uint8).tolist()
        print(json.dumps({"condition": key, **result["metrics"]}), flush=True)
    make_graphs(args.output, results, rows, labels)
    public_results = {key: {name: value for name, value in result.items() if name != "scores"} for key, result in results.items()}
    comparison = {metric: float(public_results["gan_10pct"]["metrics"][metric] - public_results["real_only"]["metrics"][metric]) for metric in METRICS}
    paired = paired_exact(labels, np.asarray(results["real_only"]["scores"]), results["real_only"]["threshold"],
                          np.asarray(results["gan_10pct"]["scores"]), results["gan_10pct"]["threshold"])
    report = {"project_version": 3, "experiment": "new raw long-legitimate-URL stress test", "dataset": str(args.dataset),
              "dataset_sha256": collection["dataset_sha256"], "selection_locked_before_inference": True,
              "thresholds_not_tuned_on_test": True, "device": str(device), "collection": collection,
              "conditions": public_results, "gan_10pct_minus_real_only": comparison, "paired_exact_comparison": paired,
              "interpretation_limit": "This intentionally difficult long-URL set is a stress test, not an estimate of normal browser traffic."}
    write_json(args.output / "results.json", report)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(prediction_columns); writer.writerows(zip(*prediction_columns.values()))
    print(json.dumps({"output": str(args.output), "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
