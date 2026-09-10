from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_final_prospective_pair import extended_metrics, score_condition
from .metrics import classification_metrics
from .url_data import psl_aware_lexical_features


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled evaluation of the frozen final model pair.")
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/final_confirmation_protocol_2026-08-22/manifest.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cohort-kind", choices=("primary", "long_url_stress"), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    return parser.parse_args()


def exact_paired(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    first_correct = first == labels
    second_correct = second == labels
    output = {}
    for name, mask in (("all_rows", np.ones(len(labels), dtype=bool)),
                       ("legitimate_rows", labels == 0), ("phishing_rows", labels == 1)):
        first_only = int(np.sum(mask & first_correct & ~second_correct))
        second_only = int(np.sum(mask & ~first_correct & second_correct))
        discordant = first_only + second_only
        output[name] = {
            "real_only_correct_gan_wrong": first_only,
            "real_only_wrong_gan_correct": second_only,
            "discordant_rows": discordant,
            "two_sided_exact_mcnemar_p_value": (
                float(binomtest(min(first_only, second_only), discordant, .5).pvalue)
                if discordant else 1.0
            ),
        }
    return output


def metric_vector(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    values = classification_metrics(labels, scores, threshold)
    return np.asarray([float(values[key]) for key in METRICS], dtype=np.float64)


def paired_bootstrap(labels: np.ndarray, real_scores: np.ndarray, real_threshold: float,
                     gan_scores: np.ndarray, gan_threshold: float, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    legitimate = np.flatnonzero(labels == 0)
    phishing = np.flatnonzero(labels == 1)
    differences = np.empty((replicates, len(METRICS)), dtype=np.float64)
    for index in range(replicates):
        sample = np.concatenate((rng.choice(legitimate, len(legitimate), replace=True),
                                 rng.choice(phishing, len(phishing), replace=True)))
        sampled_labels = labels[sample]
        differences[index] = (
            metric_vector(sampled_labels, gan_scores[sample], gan_threshold)
            - metric_vector(sampled_labels, real_scores[sample], real_threshold)
        )
    output = {}
    lower, upper = .025, .975
    for index, metric in enumerate(METRICS):
        values = differences[:, index]
        output[metric] = {
            "mean_delta": float(values.mean()),
            "confidence_interval_95": [float(np.quantile(values, lower)), float(np.quantile(values, upper))],
            "probability_delta_above_zero": float(np.mean(values > 0)),
            "probability_delta_below_zero": float(np.mean(values < 0)),
        }
    return {"replicates": replicates, "seed": seed, "stratified_by_label": True,
            "deltas_are_gan_10pct_minus_real_only": True, "metrics": output}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_pdf(path: Path, payload: dict) -> None:
    real = payload["conditions"]["real_only"]["metrics"]
    gan = payload["conditions"]["gan_10pct"]["metrics"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, payload["evaluation_name"], fontsize=24, weight="bold", color="#17324D")
        dataset = payload["dataset"]
        fig.text(.055, .855, f"{dataset['rows']:,} frozen URLs • {pct(dataset['phishing_prevalence'])} phishing • unchanged models and thresholds", fontsize=12.5, color="#3B6EA8")
        columns = ["Condition", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
        rows = []
        for key, label in (("real_only", "HYB-MS real-only"), ("gan_10pct", "HYB-MS +10% GAN")):
            metrics = payload["conditions"][key]["metrics"]
            rows.append([label] + [pct(metrics[name]) for name in METRICS])
        axis = fig.add_axes([.035, .49, .93, .24]); axis.axis("off")
        table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                           colWidths=[.20, .10, .105, .09, .09, .10, .10, .085, .085])
        table.auto_set_font_size(False); table.set_fontsize(8.7); table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0: cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
            elif row == 2: cell.set_facecolor("#E7F3F1")
        fig.text(.055, .35, payload["verdict"]["heading"], fontsize=18, weight="bold",
                 color="#2A9D8F" if payload["verdict"]["passed"] else "#D9534F")
        fig.text(.055, .285, payload["verdict"]["explanation"], fontsize=11.5, color="#526678", wrap=True)
        paired = payload["paired_exact_comparison"]["all_rows"]
        fig.text(.055, .19, f"Paired exact McNemar p = {paired['two_sided_exact_mcnemar_p_value']:.4g}; GAN uniquely corrected {paired['real_only_wrong_gan_correct']} rows and uniquely worsened {paired['real_only_correct_gan_wrong']} rows.", fontsize=11, color="#17324D")
        fig.text(.055, .10, "No model, checkpoint, calibration value, threshold, or URL was changed after inference began.", fontsize=11, weight="bold", color="#17324D")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27), facecolor="white")
        for axis, metric in zip(axes.ravel(), METRICS):
            bars = axis.bar(["Real", "+10% GAN"], [100 * real[metric], 100 * gan[metric]], color=["#3B6EA8", "#E67E22"])
            axis.set_title(metric.replace("_", " ").upper(), fontsize=10, weight="bold")
            axis.grid(axis="y", alpha=.22); axis.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=2)
        fig.suptitle("Frozen-pair metric comparison", fontsize=18, weight="bold", color="#17324D")
        fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Result already exists at {args.output}; refusing to rerun")
    protocol = load_json(args.protocol)
    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen dataset hash mismatch")
    locked = collection.get("selection_locked_before_inference",
                            collection.get("selection_locked_before_model_inference", False))
    if not locked:
        raise RuntimeError("Dataset selection was not locked before inference")
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    conditions = {}; arrays = {}
    for key in ("real_only", "gan_10pct"):
        condition = protocol["conditions"][key]
        scores, verified = score_condition(condition, urls, raw_psl, args.batch_size, device)
        threshold = float(condition["threshold"])
        conditions[key] = {"display_name": condition["display_name"], "threshold": threshold,
                           "checkpoints": verified, "metrics": extended_metrics(labels, scores, threshold)}
        arrays[f"{key}_scores"] = scores
        arrays[f"{key}_predictions"] = (scores >= threshold).astype(np.uint8)
    real = conditions["real_only"]["metrics"]; gan = conditions["gan_10pct"]["metrics"]
    deltas = {metric: float(gan[metric] - real[metric]) for metric in METRICS}
    if args.cohort_kind == "primary":
        passed = gan["recall"] > real["recall"] and gan["f1"] > real["f1"] and gan["roc_auc"] > real["roc_auc"] and gan["false_positive_rate"] <= real["false_positive_rate"]
        explanation = "Recall, F1 and ROC-AUC improved without a higher FPR." if passed else "The predeclared primary augmentation rule was not fully satisfied."
        heading = "PRIMARY AUGMENTATION RULE PASSED" if passed else "PRIMARY AUGMENTATION RULE NOT MET"
    else:
        passed = gan["false_positive_rate"] <= real["false_positive_rate"]
        explanation = "GAN augmentation did not increase false positives on long legitimate URLs." if passed else "GAN augmentation increased false positives on long legitimate URLs."
        heading = "LONG-URL NON-REGRESSION PASSED" if passed else "LONG-URL NON-REGRESSION FAILED"
    paired = exact_paired(labels, arrays["real_only_predictions"], arrays["gan_10pct_predictions"])
    bootstrap = paired_bootstrap(labels, arrays["real_only_scores"], conditions["real_only"]["threshold"],
                                 arrays["gan_10pct_scores"], conditions["gan_10pct"]["threshold"],
                                 args.bootstrap_replicates, args.bootstrap_seed)
    practical = {
        key: conditions[key]["metrics"]["false_positive_rate"] <= .02 and conditions[key]["metrics"]["recall"] >= .80
        for key in conditions
    }
    payload = {
        "project_version": 3, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": args.name, "cohort_kind": args.cohort_kind,
        "protocol": str(args.protocol), "protocol_sha256": sha256_file(args.protocol),
        "dataset": {"path": str(args.dataset), "sha256": collection["dataset_sha256"], "rows": len(rows),
                    "legitimate_rows": int(np.sum(labels == 0)), "phishing_rows": int(np.sum(labels == 1)),
                    "phishing_prevalence": float(labels.mean())},
        "conditions": conditions, "deltas_gan_10pct_minus_real_only": deltas,
        "paired_exact_comparison": paired, "paired_stratified_bootstrap": bootstrap,
        "practical_gate_passed": practical, "verdict": {"passed": passed, "heading": heading, "explanation": explanation},
        "selection_locked_before_inference": True, "thresholds_unchanged": True,
        "limitations": collection.get("limitations", []), "device": str(device),
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "frozen_predictions.npz", labels=labels, **arrays)
    payload["predictions_sha256"] = sha256_file(args.output / "frozen_predictions.npz")
    write_json(args.output / "results.json", payload)
    make_pdf(args.output / "comparison.pdf", payload)
    print(json.dumps({"verdict": payload["verdict"], "conditions": {key: value["metrics"] for key, value in conditions.items()}, "paired": paired, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
