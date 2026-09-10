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

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_browser_fresh_snapshot import score_ensemble
from .evaluate_final_prospective_pair import extended_metrics, score_condition
from .evaluate_frozen_pair_controlled import exact_paired
from .run_final_browser_evaluation import METRIC_KEYS
from .url_data import psl_aware_lexical_features


DATA_ROOT = Path("data/evaluation/postfreeze_10k_3pct_2026-09-01")
DATASET = DATA_ROOT / "evaluation_urls.csv"
COLLECTION_REPORT = DATA_ROOT / "collection_report.json"
POLICY_ROOT = Path("artifacts/final_browser_policy_v0_5_2_2026-09-01")
POLICY_MANIFEST = POLICY_ROOT / "manifest.json"
BROWSER_ROOT = Path("browser_extension_user_pilot")
OUTPUT = Path("artifacts/results/postfreeze_10k_3pct_2026-09-01_v1")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * ((proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def attach_intervals(metrics: dict) -> None:
    matrix = metrics["confusion_matrix"]
    metrics["recall_wilson_95"] = wilson(matrix["tp"], matrix["tp"] + matrix["fn"])
    metrics["fpr_wilson_95"] = wilson(matrix["fp"], matrix["fp"] + matrix["tn"])


def make_graph(path: Path, conditions: dict) -> None:
    keys = ["real_only", "gan_10pct", "browser_candidate", "extension_v0_5_1"]
    labels = ["Real-only", "+10% GAN", "Browser candidate", "Extension v0.5.1"]
    colors = ["#3B6EA8", "#E4A11B", "#2A9D8F", "#B94A48"]
    names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR",
    }
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), facecolor="white")
    fig.suptitle("10,000 URL Evaluation — 300 Phishing URLs", fontsize=21, weight="bold", color="#17324D")
    for axis, metric in zip(axes.ravel(), METRIC_KEYS):
        values = [100.0 * conditions[key]["metrics"][metric] for key in keys]
        bars = axis.bar(range(len(keys)), values, color=colors, width=0.68)
        axis.set_title(names[metric], fontsize=12, weight="bold", color="#17324D")
        axis.set_xticks(range(len(keys)), labels, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.20)
        axis.set_axisbelow(True)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], fontsize=8, padding=2, weight="bold")
        if metric == "false_positive_rate":
            axis.set_ylim(0, max(2.0, max(values) * 1.35))
        elif metric == "false_negative_rate":
            axis.set_ylim(0, max(40.0, max(values) * 1.25))
        else:
            axis.set_ylim(0, 108)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5, w_pad=1.8)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_pdf(path: Path, payload: dict, graph_path: Path) -> None:
    keys = ["real_only", "gan_10pct", "browser_candidate", "extension_v0_5_1"]
    labels = ["HYB-MS real-only", "HYB-MS +10% GAN", "Browser candidate", "Extension v0.5.1"]
    columns = ["Condition", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
    rows = []
    for key, label in zip(keys, labels):
        values = payload["conditions"][key]["metrics"]
        rows.append([label] + [f"{100 * values[metric]:.2f}%" for metric in METRIC_KEYS])
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(0.055, 0.91, "10,000 URL evaluation", fontsize=25, weight="bold", color="#17324D")
        fig.text(0.055, 0.855, "9,700 legitimate URLs • 300 phishing URLs • unchanged thresholds", fontsize=13, color="#3B6EA8")
        axis = fig.add_axes([0.025, 0.45, 0.95, 0.31])
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.18, 0.10, 0.105, 0.09, 0.08, 0.105, 0.10, 0.085, 0.085],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1, 2.05)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0:
                cell.set_facecolor("#3B6EA8")
                cell.set_text_props(color="white", weight="bold")
            elif row == 3:
                cell.set_facecolor("#E7F3F1")
        gan = payload["conditions"]["gan_10pct"]["metrics"]
        real = payload["conditions"]["real_only"]["metrics"]
        extension = payload["conditions"]["extension_v0_5_1"]["metrics"]
        fig.text(0.055, 0.34, f"GAN recall: {100 * gan['recall']:.2f}%   |   GAN FNR: {100 * gan['false_negative_rate']:.2f}%", fontsize=15, weight="bold", color="#17324D")
        fig.text(0.055, 0.275, f"GAN minus real-only F1: {100 * (gan['f1'] - real['f1']):+.2f} points", fontsize=12, color="#526678")
        fig.text(0.055, 0.225, f"Extension v0.5.1 recall: {100 * extension['recall']:.2f}%   |   FNR: {100 * extension['false_negative_rate']:.2f}%", fontsize=12, color="#526678")
        fig.text(0.055, 0.105, "The cohort is identity-disjoint from prior project URL files. Tranco roots are legitimate proxies; PhishTank rows were verified online in the cached source snapshot.", fontsize=9.5, color="#526678", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        image = plt.imread(graph_path)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        axis = fig.add_axes([0.025, 0.035, 0.95, 0.93])
        axis.imshow(image)
        axis.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation at {OUTPUT}")
    report = load_json(COLLECTION_REPORT)
    if report.get("model_inference_performed") is not False or report.get("selection_locked_before_inference") is not True:
        raise RuntimeError("Dataset was not marked locked before inference")
    if sha256_file(DATASET) != report["dataset_sha256"]:
        raise RuntimeError("Evaluation dataset hash mismatch")
    manifest = load_json(POLICY_MANIFEST)
    rows = read_rows(DATASET)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != 10_000 or int(labels.sum()) != 300:
        raise RuntimeError("Unexpected evaluation class counts")

    urls = [row["url"] for row in rows]
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    arrays: dict[str, np.ndarray] = {}
    conditions: dict[str, dict] = {}
    for key in ("real_only", "gan_10pct"):
        frozen = manifest["conditions"][key]
        scores, verified = score_condition(frozen, urls, raw_psl, 1024, device)
        threshold = float(frozen["primary_threshold"])
        predictions = (scores >= threshold).astype(np.uint8)
        metrics = extended_metrics(labels, scores, threshold)
        attach_intervals(metrics)
        conditions[key] = {
            "display_name": frozen["display_name"],
            "threshold": threshold,
            "checkpoints": verified,
            "metrics": metrics,
        }
        arrays[f"{key}_scores"] = scores.astype(np.float32)
        arrays[f"{key}_predictions"] = predictions

    browser_scores, verified_models = score_ensemble(urls, BROWSER_ROOT, 1024)
    browser_threshold = float(manifest["browser_candidate"]["primary_threshold"])
    browser_metrics = extended_metrics(labels, browser_scores, browser_threshold)
    attach_intervals(browser_metrics)
    conditions["browser_candidate"] = {
        "display_name": "Browser ONNX candidate",
        "threshold": browser_threshold,
        "verified_models": verified_models,
        "metrics": browser_metrics,
    }
    arrays["browser_scores"] = browser_scores.astype(np.float32)
    arrays["browser_candidate_predictions"] = (browser_scores >= browser_threshold).astype(np.uint8)

    extension_threshold = float(manifest["browser_candidate"]["current_v0_5_1_threshold"])
    extension_metrics = extended_metrics(labels, browser_scores, extension_threshold)
    attach_intervals(extension_metrics)
    conditions["extension_v0_5_1"] = {
        "display_name": "Installed extension v0.5.1 threshold",
        "threshold": extension_threshold,
        "verified_models": verified_models,
        "metrics": extension_metrics,
    }
    arrays["extension_v0_5_1_predictions"] = (browser_scores >= extension_threshold).astype(np.uint8)

    paired = exact_paired(labels, arrays["real_only_predictions"], arrays["gan_10pct_predictions"])
    parity = {
        "maximum_absolute_score_difference": float(np.max(np.abs(arrays["gan_10pct_scores"] - browser_scores))),
        "candidate_binary_disagreements": int(np.sum(arrays["gan_10pct_predictions"] != arrays["browser_candidate_predictions"])),
    }
    payload = {
        "project_version": 3,
        "evaluation": "post-freeze identity-disjoint 10K evaluation at 3% phishing prevalence",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dataset": {
            "path": str(DATASET),
            "sha256": report["dataset_sha256"],
            "rows": len(rows),
            "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "phishing_prevalence": float(labels.mean()),
            "identity_contract": report["identity_contract"],
        },
        "policy_manifest": str(POLICY_MANIFEST),
        "policy_manifest_sha256": sha256_file(POLICY_MANIFEST),
        "thresholds_changed": False,
        "checkpoints_changed": False,
        "conditions": conditions,
        "paired_real_vs_gan": paired,
        "browser_native_parity": parity,
        "limitations": report["limitations"] + [
            "This post-freeze evaluation does not replace the original one-shot 6K final evaluation.",
            "The manually tested 100 phishing URLs were not available as a URL list and therefore could not be checked for overlap.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUTPUT / "predictions.npz", labels=labels, **arrays)
    payload["predictions_sha256"] = sha256_file(OUTPUT / "predictions.npz")
    write_json(OUTPUT / "results.json", payload)

    with (OUTPUT / "all_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "threshold", *METRIC_KEYS, "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in ("real_only", "gan_10pct", "browser_candidate", "extension_v0_5_1"):
            metrics = conditions[key]["metrics"]
            writer.writerow({
                "condition": key,
                "threshold": conditions[key]["threshold"],
                **{metric: metrics[metric] for metric in METRIC_KEYS},
                **metrics["confusion_matrix"],
            })

    graph = OUTPUT / "metric_comparison.png"
    make_graph(graph, conditions)
    make_pdf(OUTPUT / "evaluation_report.pdf", payload, graph)
    (OUTPUT / "README.md").write_text(
        "# Post-freeze 10K evaluation at 3% phishing prevalence\n\n"
        "The evaluation uses 9,700 legitimate Tranco proxy URLs and 300 PhishTank "
        "verified-online phishing URLs. Frozen thresholds and checkpoints were unchanged.\n",
        encoding="utf-8",
    )
    (DATA_ROOT / "EVALUATION_COMPLETED.txt").write_text(
        f"Evaluation completed at {payload['evaluated_at_utc']} using unchanged frozen models and thresholds.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "device": str(device),
        "dataset": payload["dataset"],
        "conditions": {key: conditions[key]["metrics"] for key in conditions},
        "paired_real_vs_gan": paired,
        "browser_native_parity": parity,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
