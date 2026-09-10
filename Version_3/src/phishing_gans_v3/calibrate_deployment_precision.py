from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics, select_threshold
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


DATA_ROOT = Path("data/deployment/2026-08-15-precision")
OUTPUT = Path("artifacts/results/deployment_precision_2026-08-15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the frozen winner for low-FPR deployment.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--model-id", default="deployment_recovery_hyb_gan10_v1")
    parser.add_argument("--calibration-report-name", default="calibration_report.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def threshold_with_gate(labels: np.ndarray, scores: np.ndarray, maximum_fpr: float,
                        minimum_recall: float) -> tuple[dict[str, float], bool]:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]; s = scores[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    positives = int((labels == 1).sum()); negatives = int((labels == 0).sum())
    precision = tp / np.maximum(1, tp + fp); recall = tp / positives; fpr = fp / negatives
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    distinct = np.r_[s[1:] != s[:-1], True]
    valid = np.flatnonzero(distinct & (fpr <= maximum_fpr) & (recall >= minimum_recall))
    passed = bool(len(valid))
    if passed:
        index = int(valid[np.argmax(f1[valid])])
    else:
        fallback = np.flatnonzero(distinct & (fpr <= maximum_fpr))
        index = int(fallback[np.argmax(recall[fallback])]) if len(fallback) else 0
    return ({"threshold": float(s[index]), "precision": float(precision[index]),
             "recall": float(recall[index]), "f1": float(f1[index]), "fpr": float(fpr[index]),
             "maximum_fpr": maximum_fpr, "minimum_recall": minimum_recall}, passed)


def projected_precision(recall: float, fpr: float, prevalence: float) -> float:
    numerator = recall * prevalence
    return numerator / max(1e-15, numerator + fpr * (1 - prevalence))


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_pdf(path: Path, payload: dict, group_rows: list[dict], tradeoffs: list[dict]) -> None:
    selected = payload["selected_metrics"]
    original = payload["original_threshold_metrics"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        fig.text(.06, .93, "Deployment Precision Calibration", fontsize=25, weight="bold", color="#17324D")
        fig.text(.06, .875, "Frozen HYB-MS +10% ensemble • identity-disjoint calibration only", fontsize=12, color="#526274")
        status = "GATE PASSED" if payload["deployment_gate_passed"] else "GATE FAILED"
        color = "#188977" if payload["deployment_gate_passed"] else "#C94C4C"
        fig.text(.06, .77, status, fontsize=24, weight="bold", color=color)
        fig.text(.06, .70, f"Selected threshold: {payload['selected_threshold']:.6f}", fontsize=15, weight="bold")
        names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
        ax = fig.add_axes([.06, .43, .88, .17]); ax.axis("off")
        table = ax.table(cellText=[[percent(selected[k]) for k in keys]], colLabels=names, cellLoc="center", loc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            cell.set_facecolor("#365F7F" if row == 0 else "#E7F3F1")
            if row == 0: cell.set_text_props(color="white", weight="bold")
        fig.text(.06, .31, f"Gate: FPR ≤ 0.25% and recall ≥ 80%.  Observed: FPR {percent(selected['false_positive_rate'])}, recall {percent(selected['recall'])}.", fontsize=12, weight="bold")
        fig.text(.06, .22, f"Counts: TN {selected['confusion_matrix']['tn']:,} • FP {selected['confusion_matrix']['fp']:,} • FN {selected['confusion_matrix']['fn']:,} • TP {selected['confusion_matrix']['tp']:,}", fontsize=12)
        fig.text(.06, .12, "This threshold was chosen without using the final confirmation set. If the gate fails, the model must be retrained using new development data and recalibrated on a separate cohort.", fontsize=10.5, color="#526274", wrap=True)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        x = np.arange(len(tradeoffs)); width = .23
        ax.bar(x - width, [100*r["metrics"]["recall"] for r in tradeoffs], width, label="Recall", color="#1C9A8B")
        ax.bar(x, [100*r["metrics"]["precision"] for r in tradeoffs], width, label="Precision", color="#4C78A8")
        ax.bar(x + width, [100*r["metrics"]["f1"] for r in tradeoffs], width, label="F1", color="#F2B134")
        ax.set_xticks(x, [f"FPR cap\n{100*r['maximum_fpr']:.2f}%" for r in tradeoffs]); ax.set_ylabel("Percent")
        ax.set_ylim(0, 102); ax.grid(axis="y", alpha=.25); ax.legend(); ax.set_title("Threshold trade-offs on calibration data", fontsize=19, weight="bold", pad=18)
        for i, row in enumerate(tradeoffs):
            ax.text(i, 4, f"actual FPR\n{100*row['metrics']['false_positive_rate']:.3f}%", ha="center", fontsize=9)
        fig.text(.08, .04, f"Original threshold {payload['original_threshold']:.6f}: recall {percent(original['recall'])}, FPR {percent(original['false_positive_rate'])}. Selected deployment threshold is {payload['selected_threshold']:.6f}.", fontsize=10.5)
        fig.tight_layout(rect=[0, .08, 1, 1]); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        fig.text(.06, .93, "Where False Positives Occur", fontsize=24, weight="bold", color="#17324D")
        ax = fig.add_axes([.08, .39, .84, .39])
        labels = [r["group"] for r in group_rows]; values = [100*r["false_positive_rate"] for r in group_rows]
        bars = ax.bar(labels, values, color=["#4C78A8", "#72A0C1", "#9BB7D3", "#B8CADB"][:len(labels)])
        ax.set_ylabel("False-positive rate (%)"); ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.3f}%", ha="center", va="bottom", weight="bold")
        fig.text(.06, .30, "Feature-pattern counts among false positives", fontsize=13, weight="bold")
        flags = payload["false_positive_analysis"]["flag_counts"]
        fig.text(.06, .24, " • ".join(f"{key.replace('_',' ')}: {value}" for key, value in flags.items()) or "No false positives", fontsize=10.5, color="#526274", wrap=True)
        fig.text(.06, .12, "These categories identify likely collection or representation weaknesses; they are diagnostic evidence, not permission to tune against the untouched confirmation set.", fontsize=10.5, color="#526274", wrap=True)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        fig.text(.06, .93, "Operational Interpretation", fontsize=24, weight="bold", color="#17324D")
        y = .82
        for prevalence, precision in payload["projected_precision"].items():
            fig.text(.08, y, f"At {100*float(prevalence):g}% phishing prevalence", fontsize=13, color="#526274")
            fig.text(.58, y, f"projected precision {percent(precision)}", fontsize=14, weight="bold", color="#17324D")
            y -= .10
        fig.text(.08, .42, "Retraining decision", fontsize=14, weight="bold", color="#17324D")
        fig.text(.08, .36, payload["retraining_decision"], fontsize=11.5, color="#526274", wrap=True)
        fig.text(.08, .23, "Next locked action", fontsize=14, weight="bold", color="#17324D")
        fig.text(.08, .17, "Apply this exact ensemble and threshold once to a newly frozen, identity-disjoint low-prevalence confirmation cohort. Do not revise the threshold after viewing those labels.", fontsize=11.5, color="#526274", wrap=True)
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Calibration results already frozen at {args.output}")
    report = load_json(args.data_root / args.calibration_report_name)
    dataset = args.data_root / "calibration_urls.csv"
    if sha256_file(dataset) != report["dataset_sha256"]:
        raise RuntimeError("Calibration dataset hash mismatch")
    freeze_path = args.data_root / "frozen_model_manifest.json"
    if args.checkpoint_root is None:
        freeze = load_json(freeze_path)
    else:
        freeze = load_json(freeze_path) if freeze_path.exists() else {
            "original_threshold": float(np.median([
                load_json(args.checkpoint_root / f"seed_{seed}" / "metrics.json")["threshold"]
                for seed in (42, 43, 44)
            ])),
            "deployment_gate": {"maximum_fpr": .0025, "minimum_recall": .80},
        }
        freeze = {
            **freeze, "model_id": args.model_id,
            "model_display": (("Updated real-only HYB-MS" if "real_only" in args.model_id
                               else "Updated HYB-MS +10% MS-HardGAN-C")
                              + " (three-seed mean-score ensemble)"),
            "checkpoints": [
                {"path": str(args.checkpoint_root / f"seed_{seed}" / "checkpoint.pt"), "seed": seed,
                 "sha256": sha256_file(args.checkpoint_root / f"seed_{seed}" / "checkpoint.pt")}
                for seed in (42, 43, 44)
            ],
        }
    rows = list(csv.DictReader(dataset.open("r", encoding="utf-8", newline="")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    urls = [row["url"] for row in rows]
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    seed_scores = []
    for item in freeze["checkpoints"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_psl - np.asarray(saved["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
        seed_scores.append(neural_score(model, tokens, lexical, args.batch_size, device))
        del model, tokens, lexical; gc.collect(); torch.cuda.empty_cache()
    scores = np.mean(np.stack(seed_scores), axis=0)
    selected_rule, gate = threshold_with_gate(labels, scores, .0025, .80)
    selected_threshold = selected_rule["threshold"]
    selected_metrics = classification_metrics(labels, scores, selected_threshold)
    original_metrics = classification_metrics(labels, scores, float(freeze["original_threshold"]))
    tradeoffs = []
    for maximum_fpr in (.001, .0025, .005, .01):
        rule = select_threshold(labels, scores, maximum_fpr)
        tradeoffs.append({"maximum_fpr": maximum_fpr, "selection": rule,
                          "metrics": classification_metrics(labels, scores, rule["threshold"])})

    predictions = scores >= selected_threshold
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row["label"]) == 0:
            groups[row["structure_category"]].append(index)
    group_rows = []
    for name in ("root", "shallow", "deep", "query"):
        indexes = np.asarray(groups.get(name, []), dtype=int)
        fp = int(predictions[indexes].sum()) if len(indexes) else 0
        group_rows.append({"group": name, "rows": int(len(indexes)), "false_positives": fp,
                           "false_positive_rate": fp / max(1, len(indexes)),
                           "mean_score": float(scores[indexes].mean()) if len(indexes) else 0.0})
    flag_counts: Counter = Counter()
    fp_rows = []
    for index, row in enumerate(rows):
        if int(row["label"]) or not predictions[index]:
            continue
        url = row["url"]; _, host, _ = canonicalize_url(url); flags = []
        tests = {
            "long_url": len(url) >= 100, "query_string": "?" in url,
            "deep_path": structure_from_row(row) == "deep", "many_subdomains": host.count(".") >= 3,
            "punycode": "xn--" in host, "explicit_port": bool(urlsplit_safe(url).port),
            "percent_encoding": "%" in url,
            "security_words": any(word in url.lower() for word in ("login", "secure", "account", "verify", "password", "update")),
        }
        for name, active in tests.items():
            if active: flags.append(name); flag_counts[name] += 1
        if not flags: flag_counts["none_of_registered_flags"] += 1
        fp_rows.append({**row, "ensemble_score": float(scores[index]), "flags": ";".join(flags)})
    fp_rows.sort(key=lambda row: float(row["ensemble_score"]), reverse=True)

    prevalence_values = (.001, .01, .05, .10)
    payload = {
        "model": freeze["model_display"], "model_id": freeze["model_id"],
        "calibration_dataset": str(dataset), "calibration_dataset_sha256": report["dataset_sha256"],
        "rows": len(rows), "legitimate_rows": int((labels == 0).sum()), "phishing_rows": int((labels == 1).sum()),
        "original_threshold": freeze["original_threshold"], "original_threshold_metrics": original_metrics,
        "selected_threshold": selected_threshold, "selected_rule": selected_rule,
        "selected_metrics": selected_metrics, "deployment_gate": freeze["deployment_gate"],
        "deployment_gate_passed": gate,
        "tradeoffs": tradeoffs,
        "projected_precision": {str(p): projected_precision(selected_metrics["recall"], selected_metrics["false_positive_rate"], p) for p in prevalence_values},
        "false_positive_analysis": {"by_structure": group_rows, "flag_counts": dict(flag_counts), "rows": len(fp_rows)},
        "retraining_decision": ("Gate passed; preserve the frozen model. Retraining is not justified before untouched confirmation."
                                if gate else "Gate failed; threshold adjustment alone is insufficient. Retrain with newly collected, structurally diverse legitimate development rows, then recalibrate on a separate unused cohort."),
        "confirmation_contract": "model, checkpoint hashes, ensemble rule, preprocessing, and selected threshold are locked before final confirmation",
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "calibration_analysis.json", payload)
    np.savez_compressed(args.output / "calibration_predictions.npz", labels=labels, scores=scores,
                        seed_scores=np.stack(seed_scores), threshold=np.asarray([selected_threshold]))
    with (args.output / "false_positives.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys()) + ["ensemble_score", "flags"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(fp_rows)
    make_pdf(args.output / "deployment_precision_calibration.pdf", payload, group_rows, tradeoffs)
    print(json.dumps({"gate_passed": gate, "threshold": selected_threshold, "metrics": selected_metrics,
                      "false_positives": len(fp_rows)}, indent=2))


def structure_from_row(row: dict[str, str]) -> str:
    return row.get("structure_category", "unknown")


def urlsplit_safe(url: str):
    from urllib.parse import urlsplit
    try:
        return urlsplit(url if "://" in url else "http://" + url.lstrip("/"))
    except ValueError:
        return urlsplit("http://invalid/")


if __name__ == "__main__":
    main()
