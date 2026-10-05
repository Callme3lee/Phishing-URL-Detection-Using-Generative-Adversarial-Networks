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
import torch
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_browser_fresh_snapshot import score_ensemble, tier_for_score
from .evaluate_final_prospective_pair import extended_metrics, score_condition
from .evaluate_frozen_pair_controlled import exact_paired, paired_bootstrap
from .metrics import classification_metrics, select_threshold
from .url_data import psl_aware_lexical_features


DATASET_ROOT = Path("Browser Policy Evaluation Dataset 10K")
COLLECTION_REPORT = DATASET_ROOT / "collection_report.json"
DEVELOPMENT_DATASET = DATASET_ROOT / "policy_development.csv"
FINAL_DATASET = DATASET_ROOT / "frozen_final_test.csv"
POLICY_OUTPUT = Path("artifacts/final_browser_policy_v0_5_2_2026-09-01")
RESULT_OUTPUT = Path("artifacts/results/final_browser_evaluation_2026-09-01_v1")
BROWSER_ROOT = Path("browser_extension_user_pilot")
REAL_POLICY = Path("artifacts/deployment_candidates/conservative_hyb_real_v1/policy.json")
GAN_POLICY = Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
PRIMARY_FPR_CAP = 0.005
TIER_FPR_CAPS = {"block": 0.0025, "warning": 0.005, "caution": 0.01, "secondary_review": 0.05}
METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and run the one-shot Version 3 final browser-policy evaluation."
    )
    parser.add_argument("phase", choices=("freeze-policy", "evaluate-final"))
    parser.add_argument("--policy-output", type=Path, default=POLICY_OUTPUT)
    parser.add_argument("--result-output", type=Path, default=RESULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def condition_from_policy(path: Path, display_name: str) -> dict:
    policy = load_json(path)
    return {
        "display_name": display_name,
        "checkpoints": policy["checkpoints"],
        "source_policy": str(path),
        "source_policy_sha256": sha256_file(path),
    }


def thresholds_at_caps(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    output = {}
    for name, cap in TIER_FPR_CAPS.items():
        selected = select_threshold(labels, scores, maximum_fpr=cap)
        threshold = float(selected["threshold"])
        output[name] = {
            "maximum_development_fpr": cap,
            "threshold": threshold,
            "selection": selected,
            "development_metrics": classification_metrics(labels, scores, threshold),
        }
    return output


def tiers_from_thresholds(points: dict[str, dict]) -> list[dict]:
    secondary = float(points["secondary_review"]["threshold"])
    caution = float(points["caution"]["threshold"])
    warning = float(points["warning"]["threshold"])
    block = float(points["block"]["threshold"])
    if not secondary <= caution <= warning <= block:
        raise RuntimeError("Development-derived browser tier thresholds are not monotonic")
    return [
        {"name": "allow", "lower_inclusive": None, "upper_exclusive": secondary,
         "action": "allow normally"},
        {"name": "secondary_review", "lower_inclusive": secondary, "upper_exclusive": caution,
         "action": "request secondary evidence; do not block from URL score alone"},
        {"name": "caution", "lower_inclusive": caution, "upper_exclusive": warning,
         "action": "show passive caution"},
        {"name": "warning", "lower_inclusive": warning, "upper_exclusive": block,
         "action": "show a strong phishing warning"},
        {"name": "block", "lower_inclusive": block, "upper_exclusive": None,
         "action": "interstitial block with explicit user override"},
    ]


def freeze_policy(args: argparse.Namespace) -> None:
    if args.policy_output.exists():
        raise FileExistsError(f"Frozen policy already exists at {args.policy_output}; refusing to overwrite")
    report = load_json(COLLECTION_REPORT)
    expected = report["splits"]["policy_development"]
    if sha256_file(DEVELOPMENT_DATASET) != expected["sha256"]:
        raise RuntimeError("Policy-development dataset hash mismatch")
    rows = read_rows(DEVELOPMENT_DATASET)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != expected["rows"] or int(labels.sum()) != expected["phishing"]:
        raise RuntimeError("Policy-development row counts do not match the collection report")
    urls = [row["url"] for row in rows]
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()

    specifications = {
        "real_only": condition_from_policy(REAL_POLICY, "Updated HYB-MS real-only"),
        "gan_10pct": condition_from_policy(GAN_POLICY, "Updated HYB-MS +10% MS-HardGAN-C"),
    }
    scores = {}
    conditions = {}
    for key, specification in specifications.items():
        values, verified = score_condition(specification, urls, raw_psl, args.batch_size, device)
        points = thresholds_at_caps(labels, values)
        scores[key] = values.astype(np.float32)
        conditions[key] = {
            **specification,
            "checkpoints": verified,
            "threshold_points": points,
            "primary_threshold": points["warning"]["threshold"],
            "primary_development_metrics": points["warning"]["development_metrics"],
        }

    browser_scores, verified_browser_models = score_ensemble(urls, BROWSER_ROOT, args.batch_size)
    browser_points = thresholds_at_caps(labels, browser_scores)
    current_browser_policy_path = BROWSER_ROOT / "models" / "policy.json"
    current_browser_policy = load_json(current_browser_policy_path)
    current_threshold = float(current_browser_policy["primary_binary_rule"]["threshold"])
    candidate_policy = {
        "policy_version": "final_browser_policy_v0.5.2",
        "project_version": 3,
        "status": "FROZEN BEFORE FINAL TEST INFERENCE",
        "model": "Updated HYB-MS +10% MS-HardGAN-C, three-seed browser ONNX ensemble",
        "maximum_characters": 128,
        "ensemble_rule": "arithmetic mean of FP32 ONNX probabilities from seeds 42/43/44",
        "primary_binary_rule": {
            "threshold": browser_points["warning"]["threshold"],
            "maximum_policy_development_fpr": PRIMARY_FPR_CAP,
            "action": "scores at or above this value are warning-or-block",
        },
        "tiers": tiers_from_thresholds(browser_points),
        "selection_rule": "maximum development F1 subject to the registered FPR cap",
        "development_dataset": str(DEVELOPMENT_DATASET),
        "development_dataset_sha256": expected["sha256"],
        "guardrails": [
            "The frozen final-test labels were not read while selecting these thresholds.",
            "Do not change any threshold, checkpoint, feature, or tier after final-test inference.",
            "Scores are ranking outputs and are not calibrated probabilities of phishing.",
            "This URL-only research prototype does not observe page content, redirects, TLS, DNS, or reputation.",
        ],
    }
    args.policy_output.mkdir(parents=True)
    write_json(args.policy_output / "browser_policy_candidate.json", candidate_policy)
    np.savez_compressed(
        args.policy_output / "policy_development_predictions.npz",
        labels=labels,
        real_only_scores=scores["real_only"],
        gan_10pct_scores=scores["gan_10pct"],
        browser_onnx_scores=browser_scores,
    )
    manifest = {
        "project_version": 3,
        "protocol_version": "final_browser_evaluation_v1",
        "status": "FROZEN BEFORE FINAL TEST INFERENCE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "policy_development": {
            "path": str(DEVELOPMENT_DATASET), "sha256": expected["sha256"],
            "rows": len(rows), "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
        },
        "final_test_contract": {
            "path": str(FINAL_DATASET),
            "expected_sha256_from_collection_report": report["splits"]["frozen_final_test"]["sha256"],
            "expected_rows": report["splits"]["frozen_final_test"]["rows"],
            "expected_legitimate_rows": report["splits"]["frozen_final_test"]["legitimate"],
            "expected_phishing_rows": report["splits"]["frozen_final_test"]["phishing"],
            "labels_read_during_policy_freeze": False,
        },
        "threshold_policy": {
            "primary_fpr_cap": PRIMARY_FPR_CAP,
            "selection": "maximum development F1 subject to the same FPR cap for every condition",
            "tier_fpr_caps": TIER_FPR_CAPS,
        },
        "conditions": conditions,
        "browser_candidate": {
            "root": str(BROWSER_ROOT),
            "verified_models": verified_browser_models,
            "threshold_points": browser_points,
            "primary_threshold": browser_points["warning"]["threshold"],
            "primary_development_metrics": browser_points["warning"]["development_metrics"],
            "candidate_policy": str(args.policy_output / "browser_policy_candidate.json"),
            "candidate_policy_sha256": sha256_file(args.policy_output / "browser_policy_candidate.json"),
            "current_v0_5_1_policy": str(current_browser_policy_path),
            "current_v0_5_1_policy_sha256": sha256_file(current_browser_policy_path),
            "current_v0_5_1_threshold": current_threshold,
            "current_v0_5_1_development_metrics": classification_metrics(labels, browser_scores, current_threshold),
        },
        "deployment_gate": {"maximum_final_fpr": 0.02, "minimum_final_recall": 0.80},
        "candidate_selection_rule": (
            "Among conditions passing the deployment gate, choose the highest final F1; "
            "break ties by higher recall, lower FPR, then higher ROC-AUC. If none pass, "
            "declare that no condition is deployment-ready and retain the highest-F1 model only as a research candidate."
        ),
        "augmentation_support_rule": (
            "GAN must exceed real-only on recall, F1 and ROC-AUC while its FPR is no higher."
        ),
        "statistics": {
            "paired_test": "two-sided exact McNemar",
            "bootstrap": "2,000 label-stratified paired replicates with seed 20260901",
            "confidence_level": 0.95,
        },
        "policy_development_predictions": str(args.policy_output / "policy_development_predictions.npz"),
        "policy_development_predictions_sha256": sha256_file(args.policy_output / "policy_development_predictions.npz"),
        "collection_report": str(COLLECTION_REPORT),
        "collection_report_sha256": sha256_file(COLLECTION_REPORT),
    }
    write_json(args.policy_output / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "device": manifest["device"],
        "primary_thresholds": {
            "real_only": conditions["real_only"]["primary_threshold"],
            "gan_10pct": conditions["gan_10pct"]["primary_threshold"],
            "browser_onnx": browser_points["warning"]["threshold"],
        },
        "development_metrics": {
            "real_only": conditions["real_only"]["primary_development_metrics"],
            "gan_10pct": conditions["gan_10pct"]["primary_development_metrics"],
            "browser_onnx": browser_points["warning"]["development_metrics"],
        },
        "output": str(args.policy_output),
    }, indent=2))


def projected_precision(recall: float, fpr: float, prevalence: float) -> float:
    numerator = recall * prevalence
    return numerator / max(1e-12, numerator + fpr * (1.0 - prevalence))


def slice_summary(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    sliced_labels = labels[mask]
    sliced_scores = scores[mask]
    sliced_predictions = predictions[mask]
    output = {
        "rows": int(mask.sum()),
        "legitimate_rows": int(np.sum(sliced_labels == 0)),
        "phishing_rows": int(np.sum(sliced_labels == 1)),
        "predicted_phishing": int(sliced_predictions.sum()),
        "mean_score": float(np.mean(sliced_scores)) if len(sliced_scores) else None,
        "median_score": float(np.median(sliced_scores)) if len(sliced_scores) else None,
    }
    legitimate = sliced_labels == 0
    phishing = sliced_labels == 1
    if legitimate.any():
        output["false_positives"] = int(np.sum(sliced_predictions[legitimate] == 1))
        output["false_positive_rate"] = output["false_positives"] / int(legitimate.sum())
    if phishing.any():
        output["false_negatives"] = int(np.sum(sliced_predictions[phishing] == 0))
        output["false_negative_rate"] = output["false_negatives"] / int(phishing.sum())
        output["recall"] = 1.0 - output["false_negative_rate"]
    return output


def build_masks(rows: list[dict[str, str]], labels: np.ndarray, frozen_at: str) -> dict[str, np.ndarray]:
    sources = np.asarray([row["source"] for row in rows])
    lengths = np.asarray([int(row["url_length"]) for row in rows])
    login = np.asarray([int(row["is_login_url"]) == 1 for row in rows])
    legitimate = labels == 0
    phishing = labels == 1
    root = sources == "Tranco current root-domain proxy"
    complex_page = np.asarray(["HTML capture" in source for source in sources])
    recent_phishing = np.zeros(len(rows), dtype=bool)
    reference = datetime.fromisoformat(frozen_at)
    for index, row in enumerate(rows):
        if not phishing[index]:
            continue
        try:
            observed = datetime.fromisoformat(row["source_timestamp"].replace("Z", "+00:00"))
            recent_phishing[index] = (reference - observed).total_seconds() <= 30 * 86400
        except (TypeError, ValueError):
            pass
    masks = {
        "all_rows": np.ones(len(rows), dtype=bool),
        "legitimate_root_domains": legitimate & root,
        "legitimate_complex_pages": legitimate & complex_page,
        "legitimate_login_pages": legitimate & login,
        "legitimate_long_over_128": legitimate & (lengths > 128),
        "phishing_all": phishing,
        "phishing_recent_30_days": phishing & recent_phishing,
        "phishing_login_themed": phishing & login,
        "phishing_long_over_128": phishing & (lengths > 128),
    }
    for source in sorted(set(sources)):
        masks["source__" + source.replace(" ", "_").replace("-", "_")] = sources == source
    return masks


def choose_candidate(conditions: dict, gate: dict) -> dict:
    eligible = []
    for key in ("real_only", "gan_10pct"):
        metrics = conditions[key]["metrics"]
        passed = metrics["false_positive_rate"] <= gate["maximum_final_fpr"] and metrics["recall"] >= gate["minimum_final_recall"]
        conditions[key]["deployment_gate_passed"] = passed
        if passed:
            eligible.append(key)
    if not eligible:
        research = max(("real_only", "gan_10pct"), key=lambda key: conditions[key]["metrics"]["f1"])
        return {
            "deployment_ready": False,
            "selected": None,
            "research_candidate": research,
            "reason": "Neither native condition passed the pre-frozen FPR/recall deployment gate.",
        }
    selected = max(
        eligible,
        key=lambda key: (
            conditions[key]["metrics"]["f1"],
            conditions[key]["metrics"]["recall"],
            -conditions[key]["metrics"]["false_positive_rate"],
            conditions[key]["metrics"]["roc_auc"],
        ),
    )
    return {
        "deployment_ready": True,
        "selected": selected,
        "research_candidate": selected,
        "reason": "Selected by the pre-frozen gate and metric ordering; no final threshold tuning was performed.",
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_graph(path: Path, conditions: dict) -> None:
    labels = ["Real-only", "+10% GAN", "Browser ONNX"]
    keys = ["real_only", "gan_10pct", "browser_onnx"]
    colors = ["#3B6EA8", "#E67E22", "#2A9D8F"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.7), facecolor="white")
    for axis, metric in zip(axes.ravel(), METRIC_KEYS):
        values = [100 * conditions[key]["metrics"][metric] for key in keys]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(metric.replace("_", " ").upper(), fontsize=10, weight="bold")
        axis.grid(axis="y", alpha=.20)
        axis.tick_params(axis="x", labelrotation=14, labelsize=8)
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], fontsize=8, padding=2)
        if metric not in ("false_positive_rate", "false_negative_rate"):
            axis.set_ylim(0, max(100, max(values) * 1.12))
    fig.suptitle("Final frozen evaluation: model and browser comparison", fontsize=18, weight="bold", color="#17324D")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_pdf(path: Path, payload: dict, graph_path: Path) -> None:
    conditions = payload["conditions"]
    keys = ("real_only", "gan_10pct", "browser_onnx")
    labels = ("HYB-MS real-only", "HYB-MS +10% GAN", "Browser ONNX v0.5.2")
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "Final frozen browser-policy evaluation", fontsize=24, weight="bold", color="#17324D")
        dataset = payload["dataset"]
        fig.text(.055, .855, f"{dataset['rows']:,} unseen URLs • {pct(dataset['phishing_prevalence'])} phishing • no test tuning", fontsize=12.5, color="#3B6EA8")
        columns = ["Condition", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
        table_rows = []
        for key, label in zip(keys, labels):
            metrics = conditions[key]["metrics"]
            table_rows.append([label] + [pct(metrics[metric]) for metric in METRIC_KEYS])
        axis = fig.add_axes([.025, .48, .95, .25]); axis.axis("off")
        table = axis.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center",
                           colWidths=[.19, .10, .105, .09, .09, .10, .10, .085, .085])
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 2.05)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0:
                cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
            elif row == 3:
                cell.set_facecolor("#E7F3F1")
        decision = payload["deployment_decision"]
        heading = "DEPLOYMENT GATE PASSED" if decision["deployment_ready"] else "DEPLOYMENT GATE NOT PASSED"
        fig.text(.055, .35, heading, fontsize=18, weight="bold", color="#2A9D8F" if decision["deployment_ready"] else "#D9534F")
        fig.text(.055, .285, decision["reason"], fontsize=11.5, color="#526678", wrap=True)
        augmentation = payload["augmentation_verdict"]
        fig.text(.055, .20, f"GAN augmentation: {augmentation['heading']}", fontsize=13, weight="bold", color="#17324D")
        fig.text(.055, .145, augmentation["explanation"], fontsize=10.5, color="#526678", wrap=True)
        fig.text(.055, .07, "All thresholds were frozen on the separate policy-development split before this test file was opened.", fontsize=10.5, weight="bold", color="#17324D")
        pdf.savefig(fig); plt.close(fig)

        image = plt.imread(graph_path)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        axis = fig.add_axes([.025, .04, .95, .92]); axis.imshow(image); axis.axis("off")
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "Error rates by important URL group", fontsize=22, weight="bold", color="#17324D")
        slice_names = [
            ("legitimate_root_domains", "Legitimate roots", "false_positive_rate"),
            ("legitimate_complex_pages", "Complex legit pages", "false_positive_rate"),
            ("legitimate_login_pages", "Legitimate login pages", "false_positive_rate"),
            ("legitimate_long_over_128", "Long legitimate URLs", "false_positive_rate"),
            ("phishing_all", "All phishing", "false_negative_rate"),
            ("phishing_recent_30_days", "Recent phishing", "false_negative_rate"),
            ("phishing_login_themed", "Login-themed phishing", "false_negative_rate"),
            ("phishing_long_over_128", "Long phishing URLs", "false_negative_rate"),
        ]
        rows = []
        for slice_key, label, rate in slice_names:
            first = conditions["real_only"]["slices"][slice_key]
            second = conditions["gan_10pct"]["slices"][slice_key]
            third = conditions["browser_onnx"]["slices"][slice_key]
            rows.append([
                label, str(first["rows"]), pct(float(first.get(rate, 0.0))),
                pct(float(second.get(rate, 0.0))), pct(float(third.get(rate, 0.0))),
            ])
        axis = fig.add_axes([.08, .20, .84, .58]); axis.axis("off")
        table = axis.table(cellText=rows, colLabels=["Group", "Rows", "Real error", "GAN error", "Browser error"],
                           cellLoc="center", loc="center", colWidths=[.34, .12, .18, .18, .18])
        table.auto_set_font_size(False); table.set_fontsize(9.2); table.scale(1, 1.85)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0:
                cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
        fig.text(.07, .10, "For legitimate-only groups the error is FPR; for phishing-only groups it is FNR.", fontsize=10.5, color="#526678")
        pdf.savefig(fig); plt.close(fig)


def evaluate_final(args: argparse.Namespace) -> None:
    if args.result_output.exists():
        raise FileExistsError(f"Final result already exists at {args.result_output}; refusing to rerun")
    manifest_path = args.policy_output / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "FROZEN BEFORE FINAL TEST INFERENCE":
        raise RuntimeError("Policy manifest is not marked frozen before final inference")
    if sha256_file(args.policy_output / "policy_development_predictions.npz") != manifest["policy_development_predictions_sha256"]:
        raise RuntimeError("Policy-development prediction hash mismatch")
    report = load_json(COLLECTION_REPORT)
    contract = manifest["final_test_contract"]
    actual_hash = sha256_file(FINAL_DATASET)
    if actual_hash != contract["expected_sha256_from_collection_report"]:
        raise RuntimeError("Frozen final dataset hash mismatch")
    rows = read_rows(FINAL_DATASET)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != contract["expected_rows"] or int(labels.sum()) != contract["expected_phishing_rows"]:
        raise RuntimeError("Frozen final row counts do not match the pre-frozen contract")
    urls = [row["url"] for row in rows]
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    arrays = {}
    conditions = {}
    masks = build_masks(rows, labels, report["frozen_at_utc"])

    for key in ("real_only", "gan_10pct"):
        frozen = manifest["conditions"][key]
        values, verified = score_condition(frozen, urls, raw_psl, args.batch_size, device)
        threshold = float(frozen["primary_threshold"])
        predictions = (values >= threshold).astype(np.uint8)
        conditions[key] = {
            "display_name": frozen["display_name"], "threshold": threshold,
            "checkpoints": verified, "metrics": extended_metrics(labels, values, threshold),
            "slices": {name: slice_summary(labels, values, predictions, mask) for name, mask in masks.items()},
        }
        arrays[key + "_scores"] = values.astype(np.float32)
        arrays[key + "_predictions"] = predictions

    browser_scores, verified_browser = score_ensemble(urls, BROWSER_ROOT, args.batch_size)
    browser_threshold = float(manifest["browser_candidate"]["primary_threshold"])
    browser_predictions = (browser_scores >= browser_threshold).astype(np.uint8)
    conditions["browser_onnx"] = {
        "display_name": "Packaged browser ONNX v0.5.2 candidate",
        "threshold": browser_threshold,
        "verified_models": verified_browser,
        "metrics": extended_metrics(labels, browser_scores, browser_threshold),
        "slices": {name: slice_summary(labels, browser_scores, browser_predictions, mask) for name, mask in masks.items()},
    }
    arrays["browser_onnx_scores"] = browser_scores
    arrays["browser_onnx_predictions"] = browser_predictions

    current_threshold = float(manifest["browser_candidate"]["current_v0_5_1_threshold"])
    current_metrics = extended_metrics(labels, browser_scores, current_threshold)
    gan = conditions["gan_10pct"]["metrics"]
    real = conditions["real_only"]["metrics"]
    augmentation_supported = (
        gan["recall"] > real["recall"] and gan["f1"] > real["f1"]
        and gan["roc_auc"] > real["roc_auc"]
        and gan["false_positive_rate"] <= real["false_positive_rate"]
    )
    augmentation = {
        "supported": augmentation_supported,
        "heading": "STRICT BENEFIT SUPPORTED" if augmentation_supported else "STRICT BENEFIT RULE NOT MET",
        "rule": manifest["augmentation_support_rule"],
        "explanation": (
            "The GAN condition improved recall, F1 and ROC-AUC without increasing FPR."
            if augmentation_supported else
            "The GAN condition produced a mixed or negative result under the rule frozen before final inference."
        ),
        "deltas_gan_minus_real": {metric: float(gan[metric] - real[metric]) for metric in METRIC_KEYS},
    }
    gate = manifest["deployment_gate"]
    decision = choose_candidate(conditions, gate)
    native_gan_scores = arrays["gan_10pct_scores"]
    native_gan_predictions = arrays["gan_10pct_predictions"]
    parity = {
        "maximum_absolute_score_difference": float(np.max(np.abs(native_gan_scores - browser_scores))),
        "mean_absolute_score_difference": float(np.mean(np.abs(native_gan_scores - browser_scores))),
        "binary_decision_disagreements": int(np.sum(native_gan_predictions != browser_predictions)),
        "binary_decision_disagreement_rate": float(np.mean(native_gan_predictions != browser_predictions)),
        "note": "Native CUDA mixed-precision and browser ONNX FP32 thresholds were each frozen on the same policy-development rows and FPR rule.",
    }
    tier_policy = load_json(args.policy_output / "browser_policy_candidate.json")
    tiers = np.asarray([tier_for_score(float(score), tier_policy["tiers"]) for score in browser_scores])
    tier_counts = []
    for tier in ("allow", "secondary_review", "caution", "warning", "block"):
        mask = tiers == tier
        tier_counts.append({
            "tier": tier, "rows": int(mask.sum()),
            "legitimate": int(np.sum(mask & (labels == 0))),
            "phishing": int(np.sum(mask & (labels == 1))),
        })
    exact = exact_paired(labels, arrays["real_only_predictions"], arrays["gan_10pct_predictions"])
    bootstrap = paired_bootstrap(
        labels, arrays["real_only_scores"], conditions["real_only"]["threshold"],
        arrays["gan_10pct_scores"], conditions["gan_10pct"]["threshold"],
        args.bootstrap_replicates, 20260901,
    )
    for key in ("real_only", "gan_10pct", "browser_onnx"):
        metrics = conditions[key]["metrics"]
        metrics["projected_precision_at_prevalence"] = {
            str(value): projected_precision(metrics["recall"], metrics["false_positive_rate"], value)
            for value in (.001, .01, .05)
        }
    payload = {
        "project_version": 3,
        "evaluation": "one-shot final browser-policy evaluation",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "COMPLETED ONCE; NO POST-TEST TUNING PERMITTED",
        "device": str(device),
        "policy_manifest": str(manifest_path),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "dataset": {
            "path": str(FINAL_DATASET), "sha256": actual_hash, "rows": len(rows),
            "legitimate_rows": int(np.sum(labels == 0)), "phishing_rows": int(labels.sum()),
            "phishing_prevalence": float(labels.mean()),
            "development_final_registered_domain_overlap": report["identity_contract"]["development_final_registered_domain_overlap"],
        },
        "conditions": conditions,
        "current_browser_v0_5_1_diagnostic": {
            "threshold": current_threshold, "metrics": current_metrics,
            "role": "frozen old-policy reference; not used to choose v0.5.2 thresholds",
        },
        "browser_native_parity": parity,
        "browser_tier_counts": tier_counts,
        "paired_exact_comparison": exact,
        "paired_stratified_bootstrap": bootstrap,
        "augmentation_verdict": augmentation,
        "deployment_decision": decision,
        "no_post_test_tuning": True,
        "limitations": report["limitations"] + [
            "The final cohort is an offline proxy-labelled snapshot, not live production certification.",
            "The browser bundle is URL-only and does not include the separate DNS classifier.",
            "Only 300 phishing rows are present, so phishing-slice estimates have wider uncertainty than legitimate FPR estimates.",
        ],
    }
    args.result_output.mkdir(parents=True)
    np.savez_compressed(args.result_output / "frozen_predictions.npz", labels=labels, tiers=tiers, **arrays)
    payload["predictions_sha256"] = sha256_file(args.result_output / "frozen_predictions.npz")
    write_json(args.result_output / "results.json", payload)
    with (args.result_output / "all_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "threshold"] + list(METRIC_KEYS) + ["tn", "fp", "fn", "tp", "deployment_gate_passed"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key in ("real_only", "gan_10pct", "browser_onnx"):
            metrics = conditions[key]["metrics"]; matrix = metrics["confusion_matrix"]
            writer.writerow({
                "condition": key, "threshold": conditions[key]["threshold"],
                **{metric: metrics[metric] for metric in METRIC_KEYS}, **matrix,
                "deployment_gate_passed": conditions[key].get("deployment_gate_passed", "deployment artifact parity condition"),
            })
    with (args.result_output / "slice_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "slice", "rows", "legitimate_rows", "phishing_rows", "false_positives", "false_positive_rate", "false_negatives", "false_negative_rate", "recall", "mean_score"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key in ("real_only", "gan_10pct", "browser_onnx"):
            for name, values in conditions[key]["slices"].items():
                writer.writerow({field: ({"condition": key, "slice": name}.get(field, values.get(field, ""))) for field in fields})
    graph_path = args.result_output / "final_metric_comparison.png"
    make_graph(graph_path, conditions)
    make_pdf(args.result_output / "final_evaluation_report.pdf", payload, graph_path)
    write_json(args.result_output / "browser_policy_candidate.json", tier_policy)
    (args.result_output / "README.md").write_text(
        "# Final frozen browser-policy evaluation\n\n"
        f"Status: **{payload['evaluation_status']}**\n\n"
        f"Deployment decision: **{'passed' if decision['deployment_ready'] else 'not passed'}**. "
        f"{decision['reason']}\n\n"
        f"GAN augmentation: **{augmentation['heading']}**.\n\n"
        "The final dataset may not be used to change thresholds, model checkpoints, preprocessing, or tiers.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evaluation_status": payload["evaluation_status"],
        "conditions": {key: conditions[key]["metrics"] for key in ("real_only", "gan_10pct", "browser_onnx")},
        "augmentation_verdict": augmentation,
        "deployment_decision": decision,
        "browser_native_parity": parity,
        "output": str(args.result_output),
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.phase == "freeze-policy":
        freeze_policy(args)
    else:
        evaluate_final(args)


if __name__ == "__main__":
    main()
