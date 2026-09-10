from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .url_data import encode_url, psl_aware_lexical_features


DATASET = Path("data/external/2026-08-13-browser-prevalence-final/external_urls.csv")
COLLECTION = Path("data/external/2026-08-13-browser-prevalence-final/collection_report.json")
SELECTION = Path("artifacts/results/all_models_equal_calibration_v1/definitive_summary.json")
OUTPUT = Path("artifacts/results/final_browser_prevalence_2026-08-13")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot evaluation of the selected model on the final browser cohort.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--collection-report", type=Path, default=COLLECTION)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [centre - half, centre + half]


def read_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["url"] for row in rows], np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def title(fig, heading: str, subtitle: str) -> None:
    fig.text(.06, .94, heading, fontsize=23, fontweight="bold", color="#15294B", va="top")
    fig.text(.06, .875, subtitle, fontsize=11.5, color="#526274", va="top")
    fig.add_artist(plt.Line2D([.06, .94], [.84, .84], color="#CAD5E1", lw=1))


def make_pdf(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    cm = metrics["confusion_matrix"]
    old = payload["prior_external_reference"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.2)); fig.patch.set_facecolor("white")
        title(fig, "Unseen Deployment-Style Evaluation", f"One frozen model • one frozen threshold • {payload['dataset']['legitimate_rows']:,} legitimate + {payload['dataset']['phishing_rows']:,} phishing URLs")
        fig.text(.06, .76, payload["model_display"], fontsize=20, fontweight="bold", color="#15294B")
        fig.text(.06, .70, "FINAL RESULT", fontsize=11, fontweight="bold", color="#1C9A8B")
        columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
        ax = fig.add_axes([.06, .51, .88, .14]); ax.axis("off")
        table = ax.table(cellText=[[pct(metrics[k]) for k in keys]], colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 2.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1"); cell.set_linewidth(.6)
            if row == 0: cell.set_facecolor("#365F7F"); cell.set_text_props(color="white", fontweight="bold")
            else: cell.set_facecolor("#E7F3F1")
        verdict = payload["verdict"]
        fig.text(.06, .42, verdict["heading"], fontsize=16, fontweight="bold", color=verdict["color"])
        fig.text(.06, .35, textwrap.fill(verdict["text"], 125), fontsize=11.5, color="#15294B", va="top")
        fig.text(.06, .20, f"Confusion counts: TN {cm['tn']:,} • FP {cm['fp']:,} • FN {cm['fn']:,} • TP {cm['tp']:,}", fontsize=12, fontweight="bold", color="#15294B")
        fig.text(.06, .14, "This is confirmation evidence, not permission to tune. The model and threshold must remain unchanged after seeing it.", fontsize=10.5, color="#526274")
        fig.text(.94, .035, "1", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Why This Resembles Browser Deployment", "Low prevalence exposes alert burden that balanced accuracy can hide")
        labels = ["Legitimate", "Phishing"]
        values = [payload["dataset"]["legitimate_rows"], payload["dataset"]["phishing_rows"]]
        ax = fig.add_axes([.08, .21, .42, .55]); bars = ax.bar(labels, values, color=["#4C78A8", "#E45756"])
        ax.set_ylabel("URLs"); ax.set_title("Final cohort composition", fontweight="bold")
        ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:,}", ha="center", va="bottom", fontweight="bold")
        fig.text(.56, .72, "Observed prevalence", fontsize=12, color="#526274")
        fig.text(.56, .65, pct(payload["dataset"]["phishing_prevalence"]), fontsize=28, fontweight="bold", color="#15294B")
        fig.text(.56, .53, "Operational view", fontsize=14, fontweight="bold", color="#15294B")
        operational = payload["operational"]
        lines = [
            f"False alerts per 10,000 visits: {operational['false_alerts_per_10000']:.1f}",
            f"Detected phish per 10,000 visits: {operational['detected_phish_per_10000']:.1f}",
            f"Missed phish per 10,000 visits: {operational['missed_phish_per_10000']:.1f}",
            f"Alerts that are true phishing: {pct(metrics['precision'])}",
        ]
        fig.text(.56, .47, "\n\n".join(lines), fontsize=11.5, color="#15294B", va="top")
        fig.text(.06, .09, textwrap.fill(f"Accuracy alone is misleading here: predicting every URL as legitimate would score {pct(payload['always_legitimate_accuracy'])} accuracy while detecting zero phishing. Recall, FPR, precision, F1, and ranking metrics must be read together.", 145), fontsize=10.5, color="#526274")
        fig.text(.94, .035, "2", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Confusion Matrix and Uncertainty", "Exact counts and 95% Wilson intervals")
        ax = fig.add_axes([.08, .31, .38, .43]); matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred. legitimate", "Pred. phishing"]); ax.set_yticks([0, 1], ["True legitimate", "True phishing"])
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", fontsize=19, fontweight="bold", color="white" if matrix[i,j] > matrix.max()/2 else "#15294B")
        ax.set_title("Final frozen decisions", fontweight="bold")
        cis = payload["confidence_intervals_95"]
        fig.text(.54, .70, "Metric", fontweight="bold", color="#526274")
        fig.text(.72, .70, "Estimate", fontweight="bold", color="#526274")
        fig.text(.84, .70, "95% CI", fontweight="bold", color="#526274")
        y = .64
        for name, key in [("Recall", "recall"), ("FPR", "false_positive_rate"), ("Precision", "precision"), ("Accuracy", "accuracy")]:
            lo, hi = cis[key]
            fig.text(.54, y, name, fontsize=11); fig.text(.72, y, pct(metrics[key]), fontsize=11, fontweight="bold")
            fig.text(.84, y, f"{pct(lo)}–{pct(hi)}", fontsize=11); y -= .09
        fig.text(.54, .27, f"Specificity: {pct(metrics['specificity'])}\nNPV: {pct(metrics['negative_predictive_value'])}\nBalanced accuracy: {pct(metrics['balanced_accuracy'])}\nMCC: {metrics['matthews_correlation_coefficient']:.4f}", fontsize=11.5, color="#15294B", va="top", linespacing=1.7)
        fig.text(.06, .07, f"FPR is estimated from {payload['dataset']['legitimate_rows']:,} legitimate URLs; recall is estimated from {payload['dataset']['phishing_rows']:,} phishing URLs.", fontsize=10.2, color="#526274")
        fig.text(.94, .035, "3", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Previous Stress Test vs Final Low-Prevalence Test", "Same selected model and threshold; different class prevalence and fresh identities")
        names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FPR"]
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]
        x = np.arange(len(names)); width = .36
        ax = fig.add_axes([.08, .28, .86, .48])
        ax.bar(x-width/2, [100*old[k] for k in keys], width, label="Earlier 30K (30.07% phish)", color="#9BB7D3")
        ax.bar(x+width/2, [100*metrics[k] for k in keys], width,
               label=f"Current {payload['dataset']['rows']/1000:g}K ({pct(payload['dataset']['phishing_prevalence'])} phish)", color="#1C9A8B")
        ax.set_xticks(x, names); ax.set_ylabel("Percent"); ax.grid(axis="y", alpha=.25); ax.legend(loc="upper right")
        ax.set_title("Transfer under deployment-like class imbalance", fontweight="bold")
        fig.text(.08, .16, textwrap.fill(payload["comparison_interpretation"], 125), fontsize=11, color="#15294B")
        fig.text(.08, .08, "Precision is prevalence-sensitive; ROC-AUC, recall, and FPR are more directly comparable across these two cohorts.", fontsize=10.5, color="#526274")
        fig.text(.94, .035, "4", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Final Interpretation and Deployment Boundary", "What the one-shot confirmation supports")
        fig.text(.07, .76, payload["verdict"]["heading"], fontsize=18, fontweight="bold", color=payload["verdict"]["color"])
        sections = [
            ("Supported", payload["claims"]["supported"]),
            ("Not supported", payload["claims"]["not_supported"]),
            ("Operational next step", payload["claims"]["next_step"]),
        ]
        y = .67
        colors = ["#1C9A8B", "#E45756", "#365F7F"]
        for (heading, items), color in zip(sections, colors):
            fig.text(.07, y, heading.upper(), fontsize=11, fontweight="bold", color=color); y -= .045
            for item in items:
                wrapped = textwrap.wrap(item, 125)
                fig.text(.09, y, "• " + wrapped[0], fontsize=10.5, color="#15294B")
                for extra in wrapped[1:]:
                    y -= .027; fig.text(.105, y, extra, fontsize=10.5, color="#15294B")
                y -= .055
            y -= .02
        fig.text(.06, .045, f"Dataset SHA-256: {payload['dataset']['sha256']}", fontsize=7.5, color="#728398")
        fig.text(.94, .035, "5", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Final evaluation is already frozen at {args.output}; refusing to rerun or overwrite it.")
    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen final dataset hash mismatch")
    if not collection.get("selection_locked_before_inference"):
        raise RuntimeError("Dataset was not declared locked before inference")
    selection = load_json(args.selection)
    winner = selection["winner"]
    if winner["id"] != "experiments__fair__hybrid__gan__v1__gan__10pct__hyb__ms":
        raise RuntimeError("The selected model changed unexpectedly")

    urls, labels = read_rows(args.dataset)
    if len(urls) != collection["rows"] or int(labels.sum()) != collection["phishing_rows"]:
        raise RuntimeError("Dataset count mismatch")
    device = require_cuda()
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    seed_scores = []
    verified_checkpoints = []
    for checkpoint in winner["checkpoints"]:
        path = Path(checkpoint["path"])
        actual_hash = sha256_file(path)
        if actual_hash != checkpoint["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_psl - np.asarray(saved["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
        seed_scores.append(neural_score(model, tokens, lexical, args.batch_size, device))
        verified_checkpoints.append({**checkpoint, "verified_sha256": actual_hash})
        del model, tokens, lexical; gc.collect(); torch.cuda.empty_cache()
    scores = np.mean(np.stack(seed_scores), axis=0)
    threshold = float(winner["threshold"])
    metrics = classification_metrics(labels, scores, threshold)
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(cm[k]) for k in ("tn", "fp", "fn", "tp"))
    metrics.update({
        "specificity": tn / (tn + fp),
        "negative_predictive_value": tn / (tn + fn),
        "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
        "matthews_correlation_coefficient": (tp * tn - fp * fn) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
    })
    confidence = {
        "recall": wilson(tp, tp + fn),
        "false_positive_rate": wilson(fp, fp + tn),
        "precision": wilson(tp, tp + fp),
        "accuracy": wilson(tp + tn, len(labels)),
    }
    per_10000 = 10_000 / len(labels)
    operational = {
        "false_alerts_per_10000": fp * per_10000,
        "detected_phish_per_10000": tp * per_10000,
        "missed_phish_per_10000": fn * per_10000,
        "total_alerts_per_10000": (tp + fp) * per_10000,
    }
    passed = metrics["false_positive_rate"] <= .02 and metrics["recall"] >= .80
    verdict = {
        "status": "confirmed_offline_candidate" if passed else "confirmation_failed",
        "heading": "OFFLINE CANDIDATE CONFIRMED" if passed else "FINAL CONFIRMATION DID NOT PASS",
        "color": "#1C9A8B" if passed else "#E45756",
        "text": ("The locked model retained at least 80% phishing recall with no more than 2% false-positive rate on a completely identity-disjoint, low-prevalence cohort."
                 if passed else "The locked model failed at least one predeclared practical guardrail: recall of 80% or FPR of 2%. It must not be described as confirmed for deployment."),
    }
    old = winner["evaluations"]["external_30k"]
    comparison = (f"On the final cohort, recall changed from {pct(old['recall'])} to {pct(metrics['recall'])} and FPR from "
                  f"{pct(old['false_positive_rate'])} to {pct(metrics['false_positive_rate'])}. Precision changed from "
                  f"{pct(old['precision'])} to {pct(metrics['precision'])}; much of that movement is expected when phishing prevalence changes from 30.07% to {pct(labels.mean())}.")
    payload = {
        "project_version": 3,
        "methodological_status": collection["purpose"] + "; model, checkpoints, ensemble, calibration, and threshold frozen before inference",
        "evaluated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "model_id": winner["id"], "model_display": "Weighted HYB-MS +10% MS-HardGAN-C (seeds 42/43/44)",
        "threshold": threshold, "threshold_policy": winner["threshold_policy"],
        "checkpoints": verified_checkpoints,
        "dataset": {"path": str(args.dataset), "sha256": collection["dataset_sha256"],
                    "rows": len(labels), "legitimate_rows": int((labels == 0).sum()),
                    "phishing_rows": int(labels.sum()), "phishing_prevalence": float(labels.mean()),
                    "overlap": collection["measured_overlap"]},
        "metrics": metrics, "confidence_intervals_95": confidence, "operational": operational,
        "always_legitimate_accuracy": float((labels == 0).mean()),
        "predeclared_confirmation_guardrails": {"maximum_fpr": .02, "minimum_recall": .80},
        "verdict": verdict, "prior_external_reference": old,
        "comparison_interpretation": comparison,
        "claims": {
            "supported": [
                f"The selected model transfers to a new identity-disjoint cohort at {pct(labels.mean())} phishing prevalence." if passed else "The cohort provides an honest estimate of the selected model's prevalence-sensitive failure mode.",
                "The result used no retraining, threshold tuning, checkpoint selection, or comparison against the other 105 models.",
            ],
            "not_supported": [
                "Production-ready status: latency, memory, privacy, browser integration, monitoring, and live alert adjudication remain untested.",
                "Universal phishing coverage: two phishing feeds and a Tranco legitimate proxy cannot represent the whole web.",
            ],
            "next_step": [
                "Package only this frozen model for a shadow-mode browser pilot; log decisions without blocking users.",
                "Measure latency and manually review false alerts and missed phish before choosing any production action threshold.",
            ],
        },
        "source_limitations": collection["limitations"],
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "frozen_predictions.npz", labels=labels, scores=scores,
                        threshold=np.asarray([threshold], dtype=np.float32))
    payload["predictions_sha256"] = sha256_file(args.output / "frozen_predictions.npz")
    write_json(args.output / "final_evaluation.json", payload)
    make_pdf(args.output / "final_browser_deployment_evaluation.pdf", payload)
    (args.output / "README.md").write_text(
        "# Final browser-prevalence evaluation\n\n"
        f"Verdict: **{verdict['heading']}**\n\n"
        f"Rows: {len(labels):,}; phishing prevalence: {pct(labels.mean())}. "
        f"F1: {pct(metrics['f1'])}; recall: {pct(metrics['recall'])}; "
        f"precision: {pct(metrics['precision'])}; FPR: {pct(metrics['false_positive_rate'])}.\n\n"
        "The run is immutable by design. The evaluator refuses to overwrite this directory.\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict["status"], "metrics": metrics, "operational": operational,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
