from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from datetime import datetime, timezone
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot evaluation of the frozen real-only and +10% HYB-MS pair.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/final_candidate_2026-08-22/manifest.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-22-prospective-2pct/external_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-22-prospective-2pct/collection_report.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/prospective_2026-08-22_2pct"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [centre - half, centre + half]


def score_condition(condition: dict, urls: list[str], raw_psl: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, list[dict]]:
    outputs = []
    verified = []
    for item in condition["checkpoints"]:
        path = Path(item["path"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_psl - np.asarray(saved["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(saved).to(device)
        model.load_state_dict(saved["state_dict"])
        outputs.append(neural_score(model, tokens, lexical, batch_size, device))
        verified.append({**item, "verified_sha256": actual})
        del model, tokens, lexical
        gc.collect(); torch.cuda.empty_cache()
    return np.mean(np.stack(outputs), axis=0), verified


def extended_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(cm[key]) for key in ("tn", "fp", "fn", "tp"))
    metrics["confidence_intervals_95"] = {
        "recall": wilson(tp, tp + fn),
        "false_positive_rate": wilson(fp, fp + tn),
        "precision": wilson(tp, tp + fp),
        "accuracy": wilson(tp + tn, len(labels)),
    }
    scale = 10_000 / len(labels)
    metrics["per_10000"] = {
        "false_alerts": fp * scale,
        "detected_phishing": tp * scale,
        "missed_phishing": fn * scale,
        "total_alerts": (tp + fp) * scale,
    }
    return metrics


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def make_pdf(path: Path, payload: dict) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "Prospective low-prevalence evaluation", fontsize=25, weight="bold", color="#17324D")
        fig.text(.055, .855, f"{payload['dataset']['rows']:,} new URLs • {pct(payload['dataset']['phishing_prevalence'])} phishing • frozen models and thresholds", fontsize=13, color="#3B6EA8")
        rows = []
        for key, label in (("real_only", "Weighted HYB-MS real-only"), ("gan_10pct", "Weighted HYB-MS +10% GAN")):
            rows.append([label] + [pct(payload["conditions"][key]["metrics"][metric]) for metric in keys])
        ax = fig.add_axes([.035, .48, .93, .25]); ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["Condition"] + names, cellLoc="center", loc="center",
                         colWidths=[.23, .095, .105, .09, .09, .10, .10, .085, .085])
        table.auto_set_font_size(False); table.set_fontsize(8.8); table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0:
                cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
            elif row == 2:
                cell.set_facecolor("#E7F3F1")
        verdict = payload["augmentation_verdict"]
        fig.text(.055, .35, verdict["heading"], fontsize=18, weight="bold", color="#2A9D8F" if verdict["supported"] else "#D9534F")
        fig.text(.055, .28, verdict["explanation"], fontsize=12, color="#526678", wrap=True)
        fig.text(.055, .16, "No training, calibration, checkpoint choice or threshold was changed after this dataset was frozen.", fontsize=11.5, color="#17324D", weight="bold")
        fig.text(.055, .08, f"Dataset SHA-256: {payload['dataset']['sha256']}", fontsize=8, color="#657786")
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "What the result means in browser-like traffic", fontsize=24, weight="bold", color="#17324D")
        x = np.arange(4); width = .35
        labels = ["Recall", "F1", "ROC-AUC", "FPR"]
        real = payload["conditions"]["real_only"]["metrics"]
        gan = payload["conditions"]["gan_10pct"]["metrics"]
        ax = fig.add_axes([.08, .36, .84, .40])
        ax.bar(x-width/2, [100*real[k] for k in ("recall", "f1", "roc_auc", "false_positive_rate")], width, label="Real-only", color="#3B6EA8")
        ax.bar(x+width/2, [100*gan[k] for k in ("recall", "f1", "roc_auc", "false_positive_rate")], width, label="+10% GAN", color="#E67E22")
        ax.set_xticks(x, labels); ax.set_ylabel("Percent"); ax.grid(axis="y", alpha=.22); ax.legend(frameon=False)
        fig.text(.07, .25, f"Real-only false alerts per 10,000 visits: {real['per_10000']['false_alerts']:.1f}", fontsize=12, color="#526678")
        fig.text(.07, .19, f"+10% GAN false alerts per 10,000 visits: {gan['per_10000']['false_alerts']:.1f}", fontsize=12, color="#526678")
        fig.text(.07, .11, "This is an identity-disjoint, proxy-labelled evaluation. It is useful evidence, but it is still not a live production deployment test.", fontsize=11.5, color="#17324D")
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Prospective result already exists at {args.output}; refusing to rerun")
    manifest = load_json(args.manifest)
    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen prospective dataset hash mismatch")
    if not collection.get("selection_locked_before_inference"):
        raise RuntimeError("Dataset was not locked before inference")
    rows = list(csv.DictReader(args.dataset.open("r", encoding="utf-8", newline="")))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != collection["rows"] or int(labels.sum()) != collection["phishing_rows"]:
        raise RuntimeError("Prospective dataset counts do not match its collection report")
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    results = {}; predictions = {}
    for key in ("real_only", "gan_10pct"):
        condition = manifest["conditions"][key]
        scores, verified = score_condition(condition, urls, raw_psl, args.batch_size, device)
        metrics = extended_metrics(labels, scores, float(condition["threshold"]))
        practical = metrics["false_positive_rate"] <= .02 and metrics["recall"] >= .80
        results[key] = {"display_name": condition["display_name"], "threshold": condition["threshold"],
                        "checkpoints": verified, "metrics": metrics, "practical_gate_passed": practical}
        predictions[f"{key}_scores"] = scores
        predictions[f"{key}_predictions"] = (scores >= float(condition["threshold"])).astype(np.uint8)
    real = results["real_only"]["metrics"]; gan = results["gan_10pct"]["metrics"]
    supported = (gan["recall"] > real["recall"] and gan["f1"] > real["f1"]
                 and gan["roc_auc"] > real["roc_auc"]
                 and gan["false_positive_rate"] <= real["false_positive_rate"])
    deltas = {metric: gan[metric] - real[metric] for metric in
              ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")}
    verdict = {
        "supported": supported,
        "heading": "AUGMENTATION BENEFIT SUPPORTED" if supported else "STRICT AUGMENTATION RULE NOT MET",
        "rule": manifest["prospective_contract"]["augmentation_support_rule"],
        "explanation": (
            "The +10% model improved recall, F1 and ROC-AUC without increasing FPR on this new cohort."
            if supported else
            "The +10% model did not improve recall, F1 and ROC-AUC while also keeping FPR no higher than real-only. The result is kept as a valid negative or mixed finding."
        ),
    }
    payload = {
        "project_version": 3,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "one-shot prospective real-only versus 10% GAN comparison",
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "dataset": {"path": str(args.dataset), "sha256": collection["dataset_sha256"],
                    "rows": len(rows), "legitimate_rows": int((labels == 0).sum()),
                    "phishing_rows": int(labels.sum()), "phishing_prevalence": float(labels.mean()),
                    "overlap": collection["measured_overlap"]},
        "conditions": results, "deltas_gan_minus_real": deltas,
        "augmentation_verdict": verdict,
        "one_shot_contract_honored": True,
        "limitations": collection["limitations"],
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "frozen_predictions.npz", labels=labels, **predictions)
    payload["predictions_sha256"] = sha256_file(args.output / "frozen_predictions.npz")
    write_json(args.output / "results.json", payload)
    make_pdf(args.output / "prospective_comparison.pdf", payload)
    (args.output / "README.md").write_text(
        "# Prospective low-prevalence comparison\n\n"
        f"Verdict: **{verdict['heading']}**\n\n"
        f"Dataset: {len(rows):,} URLs at {pct(labels.mean())} phishing prevalence. "
        "The real-only and +10% weighted HYB-MS ensembles were frozen before collection and evaluated once.\n",
        encoding="utf-8",
    )
    print(json.dumps({"augmentation_verdict": verdict, "conditions": {
        key: results[key]["metrics"] for key in results}, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
