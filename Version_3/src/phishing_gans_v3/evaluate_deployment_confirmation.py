from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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


DATA = Path("data/deployment/2026-08-15-confirmation")
OUTPUT = Path("artifacts/results/deployment_confirmation_2026-08-15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single locked deployment confirmation inference.")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials; denominator = 1 + z*z/trials
    centre = (p + z*z/(2*trials)) / denominator
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [centre-half, centre+half]


def pct(value: float) -> str:
    return f"{100*value:.2f}%"


def make_pdf(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]; cm = metrics["confusion_matrix"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        fig.text(.06, .93, "One-Shot Deployment Confirmation", fontsize=25, weight="bold", color="#17324D")
        fig.text(.06, .875, "Post-threshold collection • 1% phishing prevalence • 14,600 unseen domains", fontsize=12, color="#526274")
        status = "CONFIRMATION PASSED" if payload["gate_passed"] else "CONFIRMATION FAILED"
        fig.text(.06, .76, status, fontsize=24, weight="bold", color="#188977" if payload["gate_passed"] else "#C94C4C")
        names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
        ax = fig.add_axes([.06, .47, .88, .17]); ax.axis("off")
        table = ax.table(cellText=[[pct(metrics[key]) for key in keys]], colLabels=names, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.2)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1"); cell.set_facecolor("#365F7F" if row == 0 else "#E7F3F1")
            if row == 0: cell.set_text_props(color="white", weight="bold")
        fig.text(.06, .35, f"TN {cm['tn']:,} • FP {cm['fp']:,} • FN {cm['fn']:,} • TP {cm['tp']:,}", fontsize=13, weight="bold")
        fig.text(.06, .27, f"False alerts per 10,000 visits: {payload['operational']['false_alerts_per_10000']:.1f}   •   detected phishing per 10,000: {payload['operational']['detected_phish_per_10000']:.1f}", fontsize=11.5)
        fig.text(.06, .16, payload["verdict"], fontsize=11.5, color="#526274", wrap=True)
        fig.text(.06, .07, "No threshold, feature, checkpoint, or data-selection rule was changed after inference.", fontsize=10.5, color="#526274")
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")
        fig.text(.06, .93, "Uncertainty and Deployment Meaning", fontsize=24, weight="bold", color="#17324D")
        intervals = payload["confidence_intervals_95"]
        y=.79
        for label, key in (("Recall", "recall"), ("False-positive rate", "false_positive_rate"), ("Precision", "precision"), ("Accuracy", "accuracy")):
            lo, hi = intervals[key]
            fig.text(.08, y, label, fontsize=13, color="#526274"); fig.text(.40, y, pct(metrics[key]), fontsize=14, weight="bold")
            fig.text(.60, y, f"95% CI {pct(lo)}–{pct(hi)}", fontsize=12); y -= .11
        fig.text(.08, .29, "Interpretation boundary", fontsize=14, weight="bold", color="#17324D")
        fig.text(.08, .23, "Passing supports an offline deployment candidate under this root-heavy, proxy-labelled cohort. It does not prove production readiness; latency, live browser integration, privacy, adversarial resistance, and ongoing drift monitoring remain required.", fontsize=11.5, color="#526274", wrap=True)
        fig.text(.08, .10, "Because only 146 phishing domains were available after strict overlap removal, recall uncertainty is wider than the FPR uncertainty estimated from 14,454 legitimate domains.", fontsize=10.5, color="#526274", wrap=True)
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Confirmation already evaluated at {args.output}; refusing to rerun")
    report = load_json(args.data_root / "collection_report.json")
    dataset = args.data_root / "confirmation_urls.csv"
    if not report.get("selection_locked_before_inference") or sha256_file(dataset) != report["dataset_sha256"]:
        raise RuntimeError("Confirmation lock or dataset hash is invalid")
    rows = list(csv.DictReader(dataset.open("r", encoding="utf-8", newline="")))
    urls = [row["url"] for row in rows]; labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda(); seed_scores = []
    for item in report["checkpoints"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]: raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32); scale = np.where(scale == 0, 1, scale)
        lexical = ((raw_psl - np.asarray(saved["lexical_scaler_center"], dtype=np.float32))/scale).astype(np.float32)
        model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
        seed_scores.append(neural_score(model, tokens, lexical, args.batch_size, device))
        del model, tokens, lexical; gc.collect(); torch.cuda.empty_cache()
    scores = np.mean(np.stack(seed_scores), axis=0); threshold = float(report["threshold"])
    metrics = classification_metrics(labels, scores, threshold); cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(cm[key]) for key in ("tn", "fp", "fn", "tp"))
    gate = metrics["false_positive_rate"] <= .0025 and metrics["recall"] >= .80
    scale10k = 10_000 / len(rows)
    payload = {
        "project_version": 3, "evaluation_type": "one-shot post-threshold confirmation",
        "model_id": report["model_id"], "threshold": threshold, "dataset_sha256": report["dataset_sha256"],
        "rows": len(rows), "legitimate_rows": int((labels==0).sum()), "phishing_rows": int((labels==1).sum()),
        "phishing_prevalence": float(labels.mean()), "metrics": metrics,
        "gate": {"maximum_fpr": .0025, "minimum_recall": .80}, "gate_passed": gate,
        "confidence_intervals_95": {"recall": wilson(tp, tp+fn), "false_positive_rate": wilson(fp, fp+tn),
                                    "precision": wilson(tp, tp+fp), "accuracy": wilson(tp+tn, len(rows))},
        "operational": {"false_alerts_per_10000": fp*scale10k, "detected_phish_per_10000": tp*scale10k,
                        "missed_phish_per_10000": fn*scale10k, "alerts_per_10000": (tp+fp)*scale10k},
        "verdict": ("The predeclared recall/FPR gate passed on the untouched cohort; the system remains an offline deployment candidate, not production-ready."
                    if gate else "The predeclared gate failed on untouched confirmation. The system must not be presented as confirmed for deployment, and this cohort must not be used for further tuning."),
        "one_shot_contract_honored": True, "collection_report_sha256": sha256_file(args.data_root / "collection_report.json"),
    }
    args.output.mkdir(parents=True); write_json(args.output / "confirmation_metrics.json", payload)
    np.savez_compressed(args.output / "confirmation_predictions.npz", labels=labels, scores=scores,
                        predictions=(scores>=threshold).astype(np.uint8), seed_scores=np.stack(seed_scores))
    make_pdf(args.output / "deployment_confirmation_report.pdf", payload)
    print(json.dumps({"gate_passed": gate, "metrics": metrics, "operational": payload["operational"]}, indent=2))


if __name__ == "__main__":
    main()
