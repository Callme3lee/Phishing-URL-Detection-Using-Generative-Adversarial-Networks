from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json
from .evaluate_browser_fresh_snapshot import score_ensemble, tier_for_score
from .evaluate_final_prospective_pair import extended_metrics


BROWSER_ROOT = Path("browser_extension_balanced_v0_5_2")
OUTPUT = Path("artifacts/results/extension_v0_5_2_evaluation_2026-09-01_v1")
DATASETS = (
    {
        "key": "final_6k_5pct",
        "name": "Frozen final 6K (5% phishing)",
        "path": Path("Final Evaluation Dataset/frozen_final_test.csv"),
        "report": Path("Final Evaluation Dataset/collection_report.json"),
        "report_hash_path": ("splits", "frozen_final_test", "sha256"),
        "expected_rows": 6_000,
        "expected_phishing": 300,
        "cached": Path("artifacts/results/final_browser_evaluation_2026-09-01_v1/frozen_predictions.npz"),
        "cached_score_key": "browser_onnx_scores",
    },
    {
        "key": "postfreeze_10k_3pct",
        "name": "Post-freeze 10K (3% phishing)",
        "path": Path("data/evaluation/postfreeze_10k_3pct_2026-09-01/evaluation_urls.csv"),
        "report": Path("data/evaluation/postfreeze_10k_3pct_2026-09-01/collection_report.json"),
        "report_hash_path": ("dataset_sha256",),
        "expected_rows": 10_000,
        "expected_phishing": 300,
        "cached": Path("artifacts/results/postfreeze_10k_3pct_2026-09-01_v1/predictions.npz"),
        "cached_score_key": "browser_scores",
    },
)
METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)


def nested(mapping: dict, path: tuple[str, ...]):
    value = mapping
    for key in path:
        value = value[key]
    return value


def read_dataset(specification: dict) -> tuple[list[str], np.ndarray, str]:
    report = load_json(specification["report"])
    expected_hash = str(nested(report, specification["report_hash_path"]))
    actual_hash = sha256_file(specification["path"])
    if actual_hash != expected_hash:
        raise RuntimeError(f"Dataset hash mismatch: {specification['path']}")
    with specification["path"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != specification["expected_rows"] or int(labels.sum()) != specification["expected_phishing"]:
        raise RuntimeError(f"Dataset counts changed: {specification['path']}")
    return [row["url"] for row in rows], labels, actual_hash


def tier_counts(labels: np.ndarray, scores: np.ndarray, tiers: list[dict]) -> list[dict]:
    assignments = np.asarray([tier_for_score(float(score), tiers) for score in scores])
    output = []
    for tier in tiers:
        name = tier["name"]
        mask = assignments == name
        legitimate = int(np.sum(mask & (labels == 0)))
        phishing = int(np.sum(mask & (labels == 1)))
        output.append({"tier": name, "legitimate": legitimate, "phishing": phishing,
                       "total": legitimate + phishing})
    return output


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def make_graph(path: Path, results: dict) -> None:
    labels = [results[key]["name"] for key in results]
    short = ["Final 6K\n5% phishing", "Post-freeze 10K\n3% phishing"]
    colors = ["#2A9D8F", "#3B6EA8"]
    display = {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        "false_positive_rate": "FPR", "false_negative_rate": "FNR",
    }
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), facecolor="white")
    fig.suptitle("Browser Extension v0.5.2 — Frozen External Evaluation",
                 fontsize=21, weight="bold", color="#17324D")
    for axis, metric in zip(axes.ravel(), METRICS):
        values = [100 * results[key]["metrics"][metric] for key in results]
        bars = axis.bar(range(len(values)), values, color=colors, width=.62)
        axis.set_title(display[metric], fontsize=12, weight="bold", color="#17324D")
        axis.set_xticks(range(len(short)), short, fontsize=9)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3,
                       fontsize=9, weight="bold")
        if metric == "false_positive_rate":
            axis.set_ylim(0, max(2.0, max(values) * 1.4))
        elif metric == "false_negative_rate":
            axis.set_ylim(0, max(30.0, max(values) * 1.3))
        else:
            axis.set_ylim(0, 108)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.5, w_pad=1.6)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_pdf(path: Path, payload: dict, graph: Path) -> None:
    columns = ["Dataset", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
               "PR-AUC", "FPR", "FNR", "TN/FP/FN/TP"]
    rows = []
    for result in payload["datasets"].values():
        metrics = result["metrics"]
        matrix = metrics["confusion_matrix"]
        rows.append([
            result["name"], *[pct(metrics[key]) for key in METRICS],
            f"{matrix['tn']}/{matrix['fp']}/{matrix['fn']}/{matrix['tp']}",
        ])
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.05, .92, "Browser Extension v0.5.2 Evaluation", fontsize=24,
                 weight="bold", color="#17324D")
        fig.text(.05, .87, "Exact packaged ONNX ensemble • fixed threshold 0.9880644679",
                 fontsize=12.5, color="#3B6EA8")
        axis = fig.add_axes([.02, .48, .96, .28]); axis.axis("off")
        table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                           colWidths=[.17, .085, .09, .08, .075, .09, .085, .07, .07, .13])
        table.auto_set_font_size(False); table.set_fontsize(7.8); table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0:
                cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
        final = payload["datasets"]["final_6k_5pct"]["metrics"]
        later = payload["datasets"]["postfreeze_10k_3pct"]["metrics"]
        fig.text(.05, .36, f"Final 6K: {final['confusion_matrix']['tp']} of 300 phishing URLs detected; "
                 f"{final['confusion_matrix']['fp']} of 5,700 legitimate URLs warned.",
                 fontsize=12, color="#17324D", weight="bold")
        fig.text(.05, .29, f"Later 10K: {later['confusion_matrix']['tp']} of 300 phishing URLs detected; "
                 f"{later['confusion_matrix']['fp']} of 9,700 legitimate URLs warned.",
                 fontsize=12, color="#17324D", weight="bold")
        fig.text(.05, .19, "Interpretation: v0.5.2 improves sensitivity relative to v0.5.1 by moving "
                 "the warning boundary. It does not change model ranking or ROC-AUC, and it trades "
                 "additional warnings for fewer missed phishing URLs.", fontsize=10.5,
                 color="#526678", wrap=True)
        fig.text(.05, .09, "These are frozen proxy-labelled external cohorts, not a production "
                 "certification. Tranco URLs are legitimate proxies and PhishTank rows were verified "
                 "in their recorded source snapshots.", fontsize=9.5, color="#657786", wrap=True)
        pdf.savefig(fig); plt.close(fig)

        image = plt.imread(graph)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        axis = fig.add_axes([.025, .035, .95, .93]); axis.imshow(image); axis.axis("off")
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {OUTPUT}")
    policy_path = BROWSER_ROOT / "models" / "policy.json"
    bundle_path = BROWSER_ROOT / "models" / "bundle_manifest.json"
    policy = load_json(policy_path)
    threshold = float(policy["primary_binary_rule"]["threshold"])
    previous_threshold = float(policy["threshold_revision"]["previous_threshold"])
    if threshold != float(policy["threshold_revision"]["balanced_threshold"]):
        raise RuntimeError("v0.5.2 threshold provenance is inconsistent")

    datasets = {}
    prediction_arrays = {}
    verified_models = None
    for specification in DATASETS:
        urls, labels, dataset_hash = read_dataset(specification)
        scores, current_models = score_ensemble(urls, BROWSER_ROOT, 1024)
        if verified_models is None:
            verified_models = current_models
        elif current_models != verified_models:
            raise RuntimeError("Verified browser model list changed between cohorts")
        cached = np.load(specification["cached"])
        cached_scores = cached[specification["cached_score_key"]].astype(np.float32)
        parity = float(np.max(np.abs(scores - cached_scores)))
        metrics = extended_metrics(labels, scores, threshold)
        previous = extended_metrics(labels, scores, previous_threshold)
        datasets[specification["key"]] = {
            "name": specification["name"], "path": str(specification["path"]),
            "sha256": dataset_hash, "rows": len(labels),
            "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "phishing_prevalence": float(labels.mean()), "metrics": metrics,
            "tier_counts": tier_counts(labels, scores, policy["tiers"]),
            "v0_5_1_same_scores_metrics": previous,
            "v0_5_2_minus_v0_5_1": {key: metrics[key] - previous[key] for key in METRICS},
            "cached_score_maximum_absolute_difference": parity,
        }
        prediction_arrays[f"{specification['key']}_labels"] = labels
        prediction_arrays[f"{specification['key']}_scores"] = scores

    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUTPUT / "predictions.npz", **prediction_arrays)
    payload = {
        "project_version": 3,
        "extension_version": "0.5.2",
        "evaluation": "exact balanced browser extension on two frozen external cohorts",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_provider": "ONNX Runtime CPUExecutionProvider (deployment-faithful exported graphs)",
        "gpu_note": "GPU driver was unavailable; browser deployment uses ONNX Runtime Web/WASM, so CPU ONNX inference is the relevant artifact test.",
        "browser_bundle": {
            "root": str(BROWSER_ROOT), "policy": str(policy_path),
            "policy_sha256": sha256_file(policy_path), "bundle_manifest": str(bundle_path),
            "bundle_manifest_sha256": sha256_file(bundle_path),
            "policy_version": policy["policy_version"], "verified_models": verified_models,
        },
        "thresholds": {"v0_5_2_warning": threshold, "v0_5_1_warning": previous_threshold,
                       "strict_block": float(policy["tiers"][-1]["lower_inclusive"])},
        "datasets": datasets,
        "predictions_sha256": sha256_file(OUTPUT / "predictions.npz"),
        "conclusion": "v0.5.2 raises recall and lowers FNR relative to v0.5.1 on identical scores, at the expected cost of more false warnings. The later cohort remains harder, so the extension is an academic prototype rather than production-certified.",
    }
    write_json(OUTPUT / "results.json", payload)
    with (OUTPUT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "threshold", *METRICS, "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key, result in datasets.items():
            metrics = result["metrics"]
            writer.writerow({"dataset": key, "threshold": threshold,
                             **{name: metrics[name] for name in METRICS},
                             **metrics["confusion_matrix"]})
    graph = OUTPUT / "v0_5_2_metrics.png"
    make_graph(graph, datasets)
    make_pdf(OUTPUT / "v0_5_2_evaluation_report.pdf", payload, graph)
    print(json.dumps({"output": str(OUTPUT), "threshold": threshold,
                      "datasets": {key: value["metrics"] for key, value in datasets.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
