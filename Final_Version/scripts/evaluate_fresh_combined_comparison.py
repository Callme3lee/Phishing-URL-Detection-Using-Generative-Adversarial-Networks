#!/usr/bin/env python3
"""Fair frozen comparison on the raw-preserved newly supplied URL dataset."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_final_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Version_3/src"))
sys.path.insert(0, str(REPOSITORY / "Final_Version/scripts"))

from evaluate_root_paired_candidate import TorchEnsemble  # noqa: E402
from phishing_gans_v3.analyze_threshold_sets import recall_threshold  # noqa: E402
from phishing_gans_v3.common import sha256_file, write_json  # noqa: E402
from phishing_gans_v3.metrics import classification_metrics  # noqa: E402


DATASET = REPOSITORY / "phishing_test/normalized_combined_test_v2_raw_preserved/combined_test_urls.csv"
DATASET_MANIFEST = REPOSITORY / "phishing_test/normalized_combined_test_v2_raw_preserved/manifest.json"
CALIBRATION = REPOSITORY / "Version_3/data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv"
V3_CHECKPOINTS = REPOSITORY / "Version_3/artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms"
FINAL_CHECKPOINTS = REPOSITORY / "Final_Version/artifacts/root_paired_mass_preserved_v1/gan_10pct/hyb_ms"
OUTPUT = REPOSITORY / "Final_Version/artifacts/results/fresh_combined_v2_comparison_v1"


def specificity(metrics: dict) -> float:
    matrix = metrics["confusion_matrix"]
    return matrix["tn"] / max(1, matrix["tn"] + matrix["fp"])


def complete_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = classification_metrics(labels, scores, threshold)
    result["specificity"] = specificity(result)
    result["threshold"] = threshold
    return result


def mcnemar(labels: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    correct_a = a == labels; correct_b = b == labels
    a_only = int(np.sum(correct_a & ~correct_b)); b_only = int(np.sum(~correct_a & correct_b))
    total = a_only + b_only
    statistic = ((abs(a_only - b_only) - 1) ** 2 / total) if total else 0.0
    # Chi-square(1) survival function without scipy: erfc(sqrt(x/2)).
    import math
    p_value = math.erfc(math.sqrt(statistic / 2.0)) if total else 1.0
    return {"v3_only_correct": a_only, "final_only_correct": b_only,
            "discordant_pairs": total, "continuity_corrected_chi_square": statistic,
            "approximate_p_value": p_value}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = json.loads(DATASET_MANIFEST.read_text())
    with DATASET.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(manifest["rows"]):
        raise RuntimeError("Dataset row count differs from its manifest")
    raw_urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    categories = np.asarray([row["category"] for row in rows])
    roots = np.asarray([row["root_representation"] for row in rows])
    device = torch.device("cuda")
    v3_engine = TorchEnsemble(V3_CHECKPOINTS, device)
    final_engine = TorchEnsemble(FINAL_CHECKPOINTS, device)

    with CALIBRATION.open("r", encoding="utf-8", newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
    calibration_urls = [row["url"] for row in calibration_rows]
    calibration_labels = np.asarray([int(row["label"]) for row in calibration_rows], dtype=np.uint8)
    v3_calibration_scores = v3_engine.score_urls(calibration_urls)
    final_calibration_scores = final_engine.score_urls(calibration_urls)
    v3_point = recall_threshold(calibration_labels, v3_calibration_scores, .01)
    final_point = recall_threshold(calibration_labels, final_calibration_scores, .01)

    v3_scores = v3_engine.score_urls(raw_urls)
    final_scores = final_engine.score_urls(raw_urls)
    v3_metrics = complete_metrics(labels, v3_scores, float(v3_point["threshold"]))
    final_metrics = complete_metrics(labels, final_scores, float(final_point["threshold"]))
    v3_predictions = v3_scores >= float(v3_point["threshold"])
    final_predictions = final_scores >= float(final_point["threshold"])

    slices: list[dict] = []
    for slice_type, values in (("category", categories), ("root_representation", roots)):
        for name in sorted(set(values.tolist())):
            mask = values == name
            if not np.any(mask):
                continue
            for model, scores, threshold in (
                ("v3", v3_scores, float(v3_point["threshold"])),
                ("final", final_scores, float(final_point["threshold"])),
            ):
                if len(set(labels[mask].tolist())) == 2:
                    metrics = complete_metrics(labels[mask], scores[mask], threshold)
                else:
                    predictions = scores[mask] >= threshold
                    y = labels[mask]
                    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
                    metrics = {"accuracy": float(np.mean(predictions == y)),
                               "precision": float(tp / max(1, tp + fp)), "recall": float(tp / max(1, tp + fn)),
                               "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
                               "false_positive_rate": float(fp / max(1, fp + tn)),
                               "false_negative_rate": float(fn / max(1, fn + tp)),
                               "specificity": float(tn / max(1, tn + fp)),
                               "roc_auc": None, "pr_auc": None,
                               "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
                slices.append({"slice_type": slice_type, "slice": name, "model": model,
                               "rows": int(np.sum(mask)), "legitimate": int(np.sum(labels[mask] == 0)),
                               "phishing": int(np.sum(labels[mask] == 1)), **metrics})

    payload = {
        "project_version": "Final Version",
        "evaluation": "fresh raw-preserved comparison",
        "dataset": {"path": str(DATASET.relative_to(REPOSITORY)), "sha256": sha256_file(DATASET),
                    "rows": len(rows), "legitimate": int(np.sum(labels == 0)),
                    "phishing": int(np.sum(labels == 1)),
                    "phishing_prevalence": float(np.mean(labels)),
                    "terminal_slashes_added_or_removed": 0,
                    "exact_model_identity_training_overlap": 0,
                    "training_domain_overlap_rows": manifest["training_domain_overlap_rows"]},
        "threshold_selection": {"cohort": str(CALIBRATION.relative_to(REPOSITORY)),
                                "rule": "maximize recall subject to calibration FPR <= 1%; no test-label tuning",
                                "v3": v3_point, "final": final_point},
        "v3": v3_metrics,
        "final": final_metrics,
        "deltas_final_minus_v3": {key: float(final_metrics[key] - v3_metrics[key]) for key in
                                  ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                   "false_positive_rate", "false_negative_rate", "specificity")},
        "paired_prediction_comparison": mcnemar(labels, v3_predictions, final_predictions),
        "prediction_disagreements": int(np.sum(v3_predictions != final_predictions)),
        "slice_metrics": slices,
        "safety": {"raw_urls_changed": False, "slashes_added_or_removed": False,
                   "test_urls_opened": False, "network_requests_to_test_urls": False,
                   "retraining": False, "test_threshold_tuning": False},
        "checkpoints": {
            "v3": {str(seed): sha256_file(V3_CHECKPOINTS / f"seed_{seed}/checkpoint.pt") for seed in (42, 43, 44)},
            "final": {str(seed): sha256_file(FINAL_CHECKPOINTS / f"seed_{seed}/checkpoint.pt") for seed in (42, 43, 44)},
        },
    }
    np.savez_compressed(OUTPUT / "predictions.npz", labels=labels, v3_scores=v3_scores,
                        final_scores=final_scores, v3_predictions=v3_predictions,
                        final_predictions=final_predictions)
    with (OUTPUT / "slice_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(slices[0].keys()); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(slices)

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                   "false_positive_rate", "false_negative_rate"]
    display = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
    x = np.arange(len(metric_keys)); width = .36
    fig, axis = plt.subplots(figsize=(15, 6.5))
    axis.bar(x-width/2, [100*v3_metrics[key] for key in metric_keys], width,
             label="Version 3", color="#4C78A8")
    axis.bar(x+width/2, [100*final_metrics[key] for key in metric_keys], width,
             label="Final root-paired", color="#1C9A8B")
    axis.set_xticks(x, display); axis.set_ylabel("Percent")
    axis.set_title("Fresh raw-preserved URL comparison", weight="bold")
    axis.legend(); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT / "model_comparison.png", dpi=200); plt.close(fig)
    payload["artifacts"] = {"predictions_sha256": sha256_file(OUTPUT / "predictions.npz"),
                            "slice_metrics_sha256": sha256_file(OUTPUT / "slice_metrics.csv"),
                            "graph_sha256": sha256_file(OUTPUT / "model_comparison.png")}
    write_json(OUTPUT / "results.json", payload)
    print(json.dumps({"dataset": payload["dataset"], "threshold_selection": payload["threshold_selection"],
                      "v3": payload["v3"], "final": payload["final"],
                      "deltas_final_minus_v3": payload["deltas_final_minus_v3"],
                      "paired_prediction_comparison": payload["paired_prediction_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
