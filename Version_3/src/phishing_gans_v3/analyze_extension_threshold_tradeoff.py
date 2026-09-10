from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics


CALIBRATION_DATASET = Path("data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv")
CALIBRATION_REPORT = Path("data/calibration/updated_models_2026-08-15_v1/collection_report.json")
CALIBRATION_PREDICTIONS = Path(
    "browser_extension_google_workspace_v0_5_7/tests/browser_onnx_calibration_predictions.npz"
)
CURRENT_POLICY = Path("browser_extension_google_workspace_v0_5_7/models/policy.json")
POSTFREEZE_RESULTS = Path("artifacts/results/postfreeze_10k_3pct_2026-09-01_v1")
V057_RESULTS = Path("artifacts/results/browser_extension_v0_5_7_postfreeze_10k_3pct_2026-09-07_v1")
OUTPUT = Path("artifacts/results/extension_threshold_tradeoff_v0_5_7_2026-09-07_v1")
CAPS = (0.005, 0.01, 0.02, 0.03, 0.05)
RECOMMENDED_CAP = 0.01
EPSILON = 1e-12


def ratio(numerator: np.ndarray, denominator: int) -> np.ndarray:
    return numerator.astype(np.float64) / max(1, denominator)


def exact_threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    thresholds = sorted_scores[distinct_ends].astype(np.float64)
    cumulative_tp = np.cumsum(sorted_labels == 1)[distinct_ends].astype(np.int64)
    cumulative_fp = np.cumsum(sorted_labels == 0)[distinct_ends].astype(np.int64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    # Include the valid operating point above the largest observed score.
    thresholds = np.r_[np.nextafter(float(np.max(scores)), np.inf), thresholds]
    tp = np.r_[0, cumulative_tp]
    fp = np.r_[0, cumulative_fp]
    fn = positives - tp
    tn = negatives - fp
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=np.float64), where=(tp + fp) != 0)
    recall = ratio(tp, positives)
    fpr = ratio(fp, negatives)
    fnr = ratio(fn, positives)
    specificity = ratio(tn, negatives)
    accuracy = ratio(tp + tn, len(labels))
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    return {
        "threshold": thresholds, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "fnr": fnr, "specificity": specificity, "accuracy": accuracy,
    }


def row_at_index(sweep: dict[str, np.ndarray], index: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for key, values in sweep.items():
        value = values[index]
        row[key] = int(value) if key in {"tp", "tn", "fp", "fn"} else float(value)
    return row


def best_under_cap(sweep: dict[str, np.ndarray], cap: float) -> dict[str, float | int]:
    eligible = np.flatnonzero(sweep["fpr"] <= cap + EPSILON)
    # Primary objective: lowest FNR. Ties: lower FPR, higher F1, higher threshold.
    order = np.lexsort((
        -sweep["threshold"][eligible],
        -sweep["f1"][eligible],
        sweep["fpr"][eligible],
        sweep["fnr"][eligible],
    ))
    return row_at_index(sweep, int(eligible[order[0]]))


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    matrix = metrics["confusion_matrix"]
    metrics["specificity"] = matrix["tn"] / max(1, matrix["tn"] + matrix["fp"])
    return metrics


def flat_metrics(metrics: dict, threshold: float) -> dict[str, float | int]:
    matrix = metrics["confusion_matrix"]
    return {
        "threshold": threshold,
        "tp": matrix["tp"], "tn": matrix["tn"], "fp": matrix["fp"], "fn": matrix["fn"],
        "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        "fpr": metrics["false_positive_rate"], "fnr": metrics["false_negative_rate"],
        "specificity": metrics["specificity"], "accuracy": metrics["accuracy"],
    }


def pareto_indices(sweep: dict[str, np.ndarray]) -> np.ndarray:
    # One best (lowest FNR) point per observed FPR, then remove dominated points.
    best_for_fpr: dict[float, int] = {}
    for index, (fpr_value, fnr_value) in enumerate(zip(sweep["fpr"], sweep["fnr"])):
        previous = best_for_fpr.get(float(fpr_value))
        if previous is None or fnr_value < sweep["fnr"][previous]:
            best_for_fpr[float(fpr_value)] = index
    ordered = sorted(best_for_fpr.values(), key=lambda index: sweep["fpr"][index])
    frontier: list[int] = []
    best_fnr = float("inf")
    for index in ordered:
        if sweep["fnr"][index] < best_fnr - EPSILON:
            frontier.append(index)
            best_fnr = float(sweep["fnr"][index])
    return np.asarray(frontier, dtype=np.int64)


def useful_pareto(sweep: dict[str, np.ndarray], frontier: np.ndarray, cap_rows: list[dict]) -> list[dict]:
    selected: list[int] = []
    for row in cap_rows:
        distances = np.abs(sweep["threshold"][frontier] - float(row["threshold"]))
        selected.append(int(frontier[int(np.argmin(distances))]))
    # Add a very strict and a high-recall boundary for context without dumping the frontier.
    for target in (0.0025, 0.075):
        valid = frontier[sweep["fpr"][frontier] <= target + EPSILON]
        if len(valid):
            selected.append(int(valid[-1]))
    return [row_at_index(sweep, index) for index in sorted(set(selected), key=lambda i: sweep["fpr"][i])]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct_change(current: float, recommended: float) -> dict[str, float]:
    absolute = recommended - current
    return {
        "absolute_fraction": absolute,
        "absolute_percentage_points": 100.0 * absolute,
        "relative_fraction": absolute / current if current else float("inf"),
        "relative_percent": 100.0 * absolute / current if current else float("inf"),
    }


def score_distribution_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    negative = scores[labels == 0]
    positive = scores[labels == 1]
    def stats(values: np.ndarray) -> dict:
        return {
            "rows": len(values), "minimum": float(np.min(values)),
            "q01": float(np.quantile(values, .01)), "q05": float(np.quantile(values, .05)),
            "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "q75": float(np.quantile(values, .75)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)), "maximum": float(np.max(values)),
        }
    return {"legitimate": stats(negative), "phishing": stats(positive)}


def hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def identical_version_audit(test_scores: np.ndarray, test_labels: np.ndarray) -> dict:
    prior = np.load(POSTFREEZE_RESULTS / "predictions.npz")
    current = np.load(V057_RESULTS / "browser_predictions.npz")
    native_scores = prior["gan_10pct_scores"].astype(np.float64)
    browser_scores = prior["browser_scores"].astype(np.float64)
    v057_scores = current["scores"].astype(np.float64)
    policy_v051 = load_json(Path("browser_extension_user_pilot/models/policy.json"))
    policy_v052 = load_json(Path("browser_extension_balanced_v0_5_2/models/policy.json"))
    policy_v057 = load_json(CURRENT_POLICY)
    native_threshold = float(load_json(
        Path("artifacts/final_browser_policy_v0_5_2_2026-09-01/manifest.json")
    )["conditions"]["gan_10pct"]["primary_threshold"])
    thresholds = {
        "native_gan": native_threshold,
        "extension_v0_5_1": float(policy_v051["primary_binary_rule"]["threshold"]),
        "extension_v0_5_2": float(policy_v052["primary_binary_rule"]["threshold"]),
        "extension_v0_5_7": float(policy_v057["primary_binary_rule"]["threshold"]),
    }
    predictions = {
        "native_gan": native_scores >= thresholds["native_gan"],
        "extension_v0_5_1": browser_scores >= thresholds["extension_v0_5_1"],
        "extension_v0_5_2": browser_scores >= thresholds["extension_v0_5_2"],
        "extension_v0_5_7": v057_scores >= thresholds["extension_v0_5_7"],
    }
    model_paths = [
        Path(f"{root}/models/hyb_gan10_seed{seed}.onnx")
        for root in (
            "browser_extension_user_pilot",
            "browser_extension_balanced_v0_5_2",
            "browser_extension_google_workspace_v0_5_7",
        )
        for seed in (42, 43, 44)
    ]
    calibration_paths = [
        Path(root) / "tests/browser_onnx_calibration_predictions.npz"
        for root in (
            "browser_extension_user_pilot",
            "browser_extension_balanced_v0_5_2",
            "browser_extension_google_workspace_v0_5_7",
        )
    ]
    return {
        "test_rows": len(test_labels),
        "native_vs_v0_5_2": {
            "scores_exactly_identical": bool(np.array_equal(native_scores, browser_scores)),
            "maximum_absolute_score_difference": float(np.max(np.abs(native_scores - browser_scores))),
            "predictions_exactly_identical": bool(np.array_equal(predictions["native_gan"], predictions["extension_v0_5_2"])),
            "prediction_disagreements": int(np.sum(predictions["native_gan"] != predictions["extension_v0_5_2"])),
            "reason": "Same checkpoints and preprocessing; native CUDA and browser FP32 ONNX scores differ slightly and use separately calibrated thresholds.",
        },
        "v0_5_1_vs_v0_5_7": {
            "scores_exactly_identical": bool(np.array_equal(browser_scores, v057_scores)),
            "maximum_absolute_score_difference": float(np.max(np.abs(browser_scores - v057_scores))),
            "predictions_exactly_identical": bool(np.array_equal(predictions["extension_v0_5_1"], predictions["extension_v0_5_7"])),
            "prediction_disagreements": int(np.sum(predictions["extension_v0_5_1"] != predictions["extension_v0_5_7"])),
            "reason": "The packaged model files and primary threshold are intentionally unchanged; v0.5.7 changes reliability, service policy and UI.",
        },
        "thresholds": thresholds,
        "onnx_model_hashes": hashes(model_paths),
        "calibration_prediction_hashes": hashes(calibration_paths),
        "all_three_versions_share_exact_onnx_files": all(
            len({
                sha256_file(Path(root) / f"models/hyb_gan10_seed{seed}.onnx")
                for root in (
                    "browser_extension_user_pilot",
                    "browser_extension_balanced_v0_5_2",
                    "browser_extension_google_workspace_v0_5_7",
                )
            }) == 1 for seed in (42, 43, 44)
        ),
        "all_three_versions_share_exact_calibration_scores": len({
            sha256_file(path) for path in calibration_paths
        }) == 1,
        "cache_finding": (
            "No evaluation-cache bug found. The v0.5.7 evaluator reran ONNX inference into a new artifact; "
            "the resulting scores match because the hashed ONNX graphs and inputs are identical."
        ),
    }


def plot_results(labels: np.ndarray, scores: np.ndarray, sweep: dict[str, np.ndarray],
                 current: dict, recommended: dict) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = {"current": "#B84A4A", "recommended": "#238B76"}
    created: list[Path] = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    axis.plot(100 * sweep["fpr"], 100 * sweep["fnr"], color="#3B6EA8", linewidth=2)
    for name, row in (("Current", current), ("Recommended", recommended)):
        key = name.lower()
        axis.scatter(100 * row["fpr"], 100 * row["fnr"], s=90, color=colors[key], zorder=4,
                     label=f"{name}: FPR {100*row['fpr']:.2f}%, FNR {100*row['fnr']:.2f}%")
    axis.set(xlabel="False-positive rate (%)", ylabel="False-negative rate (%)",
             title="Validation FPR–FNR Trade-off")
    axis.set_xlim(0, 5.25); axis.set_ylim(0, 30)
    axis.grid(alpha=.25); axis.legend(frameon=False)
    path = OUTPUT / "fpr_vs_fnr.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)

    mask = sweep["threshold"] >= .60
    fig, axis = plt.subplots(figsize=(10, 6.5))
    axis.plot(sweep["threshold"][mask], 100 * sweep["fpr"][mask], label="FPR", color="#D68F24", linewidth=2)
    axis.plot(sweep["threshold"][mask], 100 * sweep["fnr"][mask], label="FNR", color="#3B6EA8", linewidth=2)
    for name, row in (("Current", current), ("Recommended", recommended)):
        axis.axvline(row["threshold"], color=colors[name.lower()], linestyle="--", linewidth=1.6, label=name)
    axis.set(xlabel="Decision threshold", ylabel="Rate (%)", title="Validation Error Rates vs Threshold")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    path = OUTPUT / "fpr_fnr_vs_threshold.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.plot(fpr, tpr, color="#3B6EA8", linewidth=2, label=f"ROC-AUC = {auc(fpr, tpr):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="#9AA8B3")
    for name, row in (("Current", current), ("Recommended", recommended)):
        axis.scatter(row["fpr"], row["recall"], s=80, color=colors[name.lower()], label=name, zorder=4)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="Validation ROC Curve")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    path = OUTPUT / "roc_curve.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.plot(recall, precision, color="#6E5AA6", linewidth=2, label=f"PR-AUC = {auc(recall, precision):.4f}")
    for name, row in (("Current", current), ("Recommended", recommended)):
        axis.scatter(row["recall"], row["precision"], s=80, color=colors[name.lower()], label=name, zorder=4)
    axis.set(xlabel="Recall", ylabel="Precision", title="Validation Precision–Recall Curve")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    path = OUTPUT / "precision_recall_curve.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6.5))
    axis.plot(sweep["threshold"][mask], 100 * sweep["f1"][mask], color="#6E5AA6", linewidth=2)
    for name, row in (("Current", current), ("Recommended", recommended)):
        axis.scatter(row["threshold"], 100 * row["f1"], s=80, color=colors[name.lower()], label=name, zorder=4)
    axis.set(xlabel="Decision threshold", ylabel="F1 (%)", title="Validation F1 vs Threshold")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    path = OUTPUT / "f1_vs_threshold.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)

    negative = np.clip(scores[labels == 0], 1e-7, 1 - 1e-7)
    positive = np.clip(scores[labels == 1], 1e-7, 1 - 1e-7)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    axes[0].hist(negative, bins=100, density=True, alpha=.62, color="#3B6EA8", label="Legitimate")
    axes[0].hist(positive, bins=100, density=True, alpha=.58, color="#C65352", label="Phishing")
    axes[0].set(xlabel="Sigmoid ensemble score", ylabel="Density", title="Raw Score Distribution")
    axes[0].set_yscale("log"); axes[0].legend(frameon=False); axes[0].grid(alpha=.2)
    logit_negative = np.log(negative / (1 - negative))
    logit_positive = np.log(positive / (1 - positive))
    axes[1].hist(logit_negative, bins=100, density=True, alpha=.62, color="#3B6EA8", label="Legitimate")
    axes[1].hist(logit_positive, bins=100, density=True, alpha=.58, color="#C65352", label="Phishing")
    axes[1].set(xlabel="Logit of ensemble score", ylabel="Density", title="Tail Overlap (Expanded Scale)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=.2)
    fig.suptitle("Validation Score Separation", fontsize=17, weight="bold", color="#17324D")
    fig.tight_layout()
    path = OUTPUT / "score_distributions.png"; fig.savefig(path, dpi=260, bbox_inches="tight"); created.append(path); plt.close(fig)
    return created


def make_pdf(paths: list[Path], summary: dict) -> None:
    with PdfPages(OUTPUT / "threshold_tradeoff_report.pdf") as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.06, .91, "FPR/FNR operating-point analysis", fontsize=24, weight="bold", color="#17324D")
        fig.text(.06, .855, "HYB-MS +10% MS-HardGAN-C browser ensemble · validation-only selection", fontsize=12, color="#3B6EA8")
        current = summary["comparison"]["validation"]["current"]
        recommended = summary["comparison"]["validation"]["recommended"]
        lines = [
            f"Calibration rows: {summary['calibration']['rows']:,} ({summary['calibration']['legitimate_rows']:,} legitimate, {summary['calibration']['phishing_rows']:,} phishing)",
            f"Current: threshold {current['threshold']:.9f}, FPR {100*current['fpr']:.2f}%, FNR {100*current['fnr']:.2f}%, F1 {100*current['f1']:.2f}%",
            f"Recommended: threshold {recommended['threshold']:.9f}, FPR {100*recommended['fpr']:.2f}%, FNR {100*recommended['fnr']:.2f}%, F1 {100*recommended['f1']:.2f}%",
            f"Validation FNR reduction: {-summary['comparison']['validation']['change']['fnr']['absolute_percentage_points']:.2f} percentage points",
            "",
            "Recommendation: use the 1% validation-FPR operating point for a recall-oriented passive warning.",
            "Do not lower the automatic block threshold to this value.",
            "",
            "Thresholds were selected without using frozen-test labels. The 10K test set was used only",
            "after selection to report confirmation behavior.",
        ]
        fig.text(.06, .73, "\n".join(lines), fontsize=12, color="#263B4B", linespacing=1.55)
        pdf.savefig(fig); plt.close(fig)
        for path in paths:
            image = plt.imread(path)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            axis = fig.add_axes([.04, .04, .92, .92]); axis.imshow(image); axis.axis("off")
            pdf.savefig(fig); plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {OUTPUT}")
    report = load_json(CALIBRATION_REPORT)
    if sha256_file(CALIBRATION_DATASET) != report["dataset_sha256"]:
        raise RuntimeError("Calibration dataset hash mismatch")
    arrays = np.load(CALIBRATION_PREDICTIONS)
    labels = arrays["labels"].astype(np.uint8)
    scores = arrays["scores"].astype(np.float64)
    if len(labels) != report["rows"] or int(labels.sum()) != report["phishing_rows"]:
        raise RuntimeError("Calibration score arrays do not match the frozen cohort")
    policy = load_json(CURRENT_POLICY)
    current_threshold = float(policy["primary_binary_rule"]["threshold"])

    sweep = exact_threshold_sweep(labels, scores)
    cap_rows = [{"fpr_constraint": cap, **best_under_cap(sweep, cap)} for cap in CAPS]
    recommended = next(row for row in cap_rows if row["fpr_constraint"] == RECOMMENDED_CAP)
    current_metrics = metrics_at(labels, scores, current_threshold)
    current = flat_metrics(current_metrics, current_threshold)
    frontier = pareto_indices(sweep)
    pareto_rows = useful_pareto(sweep, frontier, cap_rows)

    test_arrays = np.load(V057_RESULTS / "browser_predictions.npz")
    test_labels = test_arrays["labels"].astype(np.uint8)
    test_scores = test_arrays["scores"].astype(np.float64)
    test_current = flat_metrics(metrics_at(test_labels, test_scores, current_threshold), current_threshold)
    test_recommended = flat_metrics(metrics_at(test_labels, test_scores, float(recommended["threshold"])), float(recommended["threshold"]))

    comparison = {
        "validation": {
            "current": current,
            "recommended": recommended,
            "change": {
                "fpr": pct_change(float(current["fpr"]), float(recommended["fpr"])),
                "fnr": pct_change(float(current["fnr"]), float(recommended["fnr"])),
            },
        },
        "frozen_10k_confirmation_no_selection": {
            "current": test_current,
            "recommended": test_recommended,
            "change": {
                "fpr": pct_change(float(test_current["fpr"]), float(test_recommended["fpr"])),
                "fnr": pct_change(float(test_current["fnr"]), float(test_recommended["fnr"])),
            },
        },
    }
    audit = identical_version_audit(test_scores, test_labels)
    distribution = score_distribution_summary(labels, scores)
    roc_fpr, roc_tpr, _ = roc_curve(labels, scores)
    pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
    payload = {
        "analysis": "Extension v0.5.7 FPR/FNR threshold trade-off",
        "scientific_contract": {
            "threshold_selection_data": str(CALIBRATION_DATASET),
            "test_set_used_for_selection": False,
            "selection_objective": "lowest FNR subject to each FPR constraint; ties use lower FPR, higher F1, then higher threshold",
            "models_retrained": False,
            "deployed_policy_modified": False,
        },
        "calibration": {
            "path": str(CALIBRATION_DATASET), "sha256": report["dataset_sha256"],
            "predictions": str(CALIBRATION_PREDICTIONS),
            "predictions_sha256": sha256_file(CALIBRATION_PREDICTIONS),
            "rows": len(labels), "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "roc_auc": float(auc(roc_fpr, roc_tpr)),
            "pr_auc": float(auc(pr_recall, pr_precision)),
            "distinct_operating_thresholds": len(sweep["threshold"]),
        },
        "implementation_trace": {
            "raw_model_output": "one logit per checkpoint",
            "sigmoid_handling": "sigmoid is embedded once in each exported ONNX graph",
            "ensemble_score": "arithmetic mean of three FP32 sigmoid outputs (seeds 42/43/44)",
            "score_calibration": "not probability-calibrated; treated as a ranking score",
            "lexical_normalization": "(35 raw PSL-aware lexical features - stored center) / stored scale",
            "primary_threshold_path": "browser_extension_google_workspace_v0_5_7/models/policy.json -> primary_binary_rule.threshold",
            "tier_threshold_path": "the same policy.json -> tiers",
            "query_value_view": "diagnostic only; the full URL score controls the decision",
            "post_processing": [
                "exact user-allowed hostname can suppress interruption",
                "reviewed exact service/path rules can suppress URL-text warnings",
                "valid IP literals are review-only rather than automatically blocked",
                "passive mode never interrupts; normal mode interrupts only the block tier",
            ],
        },
        "fpr_constraint_operating_points": cap_rows,
        "pareto_frontier": {
            "total_points": len(frontier),
            "useful_points": pareto_rows,
        },
        "comparison": comparison,
        "score_distributions": distribution,
        "identical_results_audit": audit,
        "conclusion": {
            "threshold_tuning_sufficient_for_requested_shift": True,
            "recommended_validation_fpr_cap": RECOMMENDED_CAP,
            "recommended_threshold": float(recommended["threshold"]),
            "reason": (
                "The 1% validation-FPR point substantially lowers FNR while keeping false positives low. "
                "Use it for warning classification, not automatic blocking."
            ),
            "further_training_justified": (
                "Yes as a later controlled experiment, because score-tail overlap leaves phishing below every low-FPR threshold. "
                "The smallest next experiment is hard-example sampling with the existing BCEWithLogitsLoss pipeline, "
                "using difficult real phishing positives and difficult legitimate login/query negatives, followed by calibration on a separate cohort."
            ),
        },
    }

    OUTPUT.mkdir(parents=True)
    sweep_rows = [row_at_index(sweep, index) for index in range(len(sweep["threshold"]))]
    write_rows(OUTPUT / "dense_threshold_sweep.csv", sweep_rows)
    write_rows(OUTPUT / "fpr_constraint_operating_points.csv", cap_rows)
    write_rows(OUTPUT / "useful_pareto_points.csv", pareto_rows)
    write_json(OUTPUT / "analysis.json", payload)
    paths = plot_results(labels, scores, sweep, current, recommended)
    make_pdf(paths, payload)
    print(json.dumps({
        "calibration_auc": {"roc_auc": payload["calibration"]["roc_auc"], "pr_auc": payload["calibration"]["pr_auc"]},
        "caps": cap_rows,
        "current_vs_recommended_validation": comparison["validation"],
        "frozen_10k_confirmation": comparison["frozen_10k_confirmation_no_selection"],
        "identical_results_audit": audit,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
