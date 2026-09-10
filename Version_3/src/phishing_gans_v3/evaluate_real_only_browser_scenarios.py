from __future__ import annotations

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


SELECTION = Path("artifacts/results/all_models_equal_calibration_v1/definitive_summary.json")
OUTPUT = Path("artifacts/results/real_only_vs_gan10_browser_scenarios_2026-08-14")
SCENARIOS = {
    "browser_0_99pct": {
        "label": "0.99% phishing",
        "dataset": Path("data/external/2026-08-13-browser-prevalence-final/external_urls.csv"),
        "report": Path("data/external/2026-08-13-browser-prevalence-final/collection_report.json"),
        "gan10_result": Path("artifacts/results/final_browser_prevalence_2026-08-13/final_evaluation.json"),
    },
    "browser_5pct": {
        "label": "5.00% phishing",
        "dataset": Path("data/external/2026-08-13-browser-prevalence-5pct/external_urls.csv"),
        "report": Path("data/external/2026-08-13-browser-prevalence-5pct/collection_report.json"),
        "gan10_result": Path("artifacts/results/browser_prevalence_5pct_2026-08-13/final_evaluation.json"),
    },
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def read_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["url"] for row in rows], np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def derived(metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(cm[k]) for k in ("tn", "fp", "fn", "tp"))
    result = dict(metrics)
    result.update({
        "specificity": tn / (tn + fp),
        "negative_predictive_value": tn / (tn + fn),
        "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
        "matthews_correlation_coefficient": (tp * tn - fp * fn) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
    })
    return result


def title(fig, heading: str, subtitle: str) -> None:
    fig.text(.06, .94, heading, fontsize=22, fontweight="bold", color="#15294B", va="top")
    fig.text(.06, .875, subtitle, fontsize=11.5, color="#526274", va="top")
    fig.add_artist(plt.Line2D([.06, .94], [.84, .84], color="#CAD5E1", lw=1))


def metric_table(ax, rows: list[list[str]], columns: list[str], font_size: float = 8.8) -> None:
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(font_size); table.scale(1, 2.05)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CAD5E1"); cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor("#365F7F"); cell.set_text_props(color="white", fontweight="bold")
        elif row % 2:
            cell.set_facecolor("#E7F3F1")


def make_pdf(path: Path, payload: dict) -> None:
    colors = {"real_only": "#4C78A8", "gan_10pct": "#1C9A8B"}
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Real-Only vs 10% GAN in Browser Scenarios", "Same architecture family • frozen three-seed ensembles • separately frozen equal-calibration thresholds")
        fig.text(.06, .76, "RESULT", color="#1C9A8B", fontsize=11, fontweight="bold")
        fig.text(.06, .70, payload["conclusion"]["heading"], color="#15294B", fontsize=18, fontweight="bold")
        fig.text(.06, .63, textwrap.fill(payload["conclusion"]["text"], 125), color="#15294B", fontsize=11.5, va="top")
        columns = ["Scenario", "Condition", "Acc.", "Prec.", "Recall", "F1", "AUC", "PR-AUC", "FPR", "FNR"]
        rows = []
        for scenario in SCENARIOS:
            for condition in ("real_only", "gan_10pct"):
                m = payload["scenarios"][scenario][condition]["metrics"]
                rows.append([payload["scenarios"][scenario]["label"], "Real-only" if condition == "real_only" else "+10% GAN"] + [pct(m[k]) for k in METRICS])
        ax = fig.add_axes([.04, .27, .92, .27]); metric_table(ax, rows, columns, 7.7)
        fig.text(.06, .15, "No weights or thresholds were changed. The 10% condition refers to GAN augmentation during training—not 10% phishing prevalence.", fontsize=10.5, color="#526274")
        fig.text(.94, .035, "1", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        for page, scenario in enumerate(SCENARIOS, start=2):
            record = payload["scenarios"][scenario]
            fig = plt.figure(figsize=(11, 8.2)); title(fig, f"Scenario: {record['label']}", f"{record['rows']:,} unseen URLs • {record['legitimate_rows']:,} legitimate • {record['phishing_rows']:,} phishing")
            names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
            x = np.arange(len(names)); width = .35
            ax = fig.add_axes([.07, .31, .88, .46])
            for offset, condition in [(-width/2, "real_only"), (width/2, "gan_10pct")]:
                m = record[condition]["metrics"]
                ax.bar(x + offset, [100*m[k] for k in METRICS], width,
                       label="Real-only" if condition == "real_only" else "+10% GAN", color=colors[condition])
            ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_ylabel("Percent")
            ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.25); ax.legend(loc="lower left")
            ax.set_title("Frozen-condition metric comparison", fontweight="bold")
            real = record["real_only"]["metrics"]; gan = record["gan_10pct"]["metrics"]
            summary = (f"GAN − real-only: F1 {100*(gan['f1']-real['f1']):+.2f} points; recall {100*(gan['recall']-real['recall']):+.2f}; "
                       f"precision {100*(gan['precision']-real['precision']):+.2f}; ROC-AUC {100*(gan['roc_auc']-real['roc_auc']):+.2f}; "
                       f"FPR {100*(gan['false_positive_rate']-real['false_positive_rate']):+.2f} points.")
            fig.text(.07, .19, textwrap.fill(summary, 130), fontsize=11.5, color="#15294B")
            fig.text(.07, .10, "Positive deltas favor GAN for F1/recall/precision/AUC; a negative delta favors GAN for FPR/FNR.", fontsize=10, color="#526274")
            fig.text(.94, .035, str(page), ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11, 8.2)); title(fig, "Operational Counts and Decision", "Prevalence-sensitive alert burden")
        columns = ["Scenario", "Condition", "TN", "FP", "FN", "TP", "False alerts / 10K", "True alerts / 10K"]
        rows = []
        for scenario in SCENARIOS:
            record = payload["scenarios"][scenario]
            for condition in ("real_only", "gan_10pct"):
                cm = record[condition]["metrics"]["confusion_matrix"]
                op = record[condition]["operational"]
                rows.append([record["label"], "Real-only" if condition == "real_only" else "+10% GAN",
                             f"{cm['tn']:,}", f"{cm['fp']:,}", f"{cm['fn']:,}", f"{cm['tp']:,}",
                             f"{op['false_alerts_per_10000']:.1f}", f"{op['detected_phish_per_10000']:.1f}"])
        ax = fig.add_axes([.06, .48, .88, .25]); metric_table(ax, rows, columns, 8.5)
        fig.text(.07, .38, "INTERPRETATION", fontsize=11, fontweight="bold", color="#1C9A8B")
        for index, line in enumerate(payload["conclusion"]["points"]):
            fig.text(.09, .32-index*.075, "• " + textwrap.fill(line, 115), fontsize=11, color="#15294B", va="top")
        fig.text(.07, .065, textwrap.fill("Because both deployment cohorts have now been used for model comparison, they are retrospective evidence and must not be reused for further threshold tuning.", 120), fontsize=10.3, color="#E45756")
        fig.text(.94, .035, "4", ha="right", color="#728398"); pdf.savefig(fig); plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Comparison already frozen at {OUTPUT}")
    selection = load_json(SELECTION)
    model_spec = selection["best_no_gan"]
    if model_spec["id"] != "experiments__hybrid__cnn__coral__v1":
        raise RuntimeError("Unexpected real-only comparison model")
    datasets = {}
    all_urls = []
    for name, spec in SCENARIOS.items():
        report = load_json(spec["report"])
        if sha256_file(spec["dataset"]) != report["dataset_sha256"]:
            raise RuntimeError(f"Dataset hash mismatch for {name}")
        urls, labels = read_dataset(spec["dataset"])
        datasets[name] = {"urls": urls, "labels": labels, "report": report}
        all_urls.extend(urls)

    device = require_cuda()
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in all_urls]).astype(np.float32)
    seed_scores = []
    checkpoints = []
    for checkpoint in model_spec["checkpoints"]:
        path = Path(checkpoint["path"])
        actual_hash = sha256_file(path)
        if actual_hash != checkpoint["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in all_urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = ((raw_psl - np.asarray(saved["lexical_scaler_center"], dtype=np.float32)) / scale).astype(np.float32)
        model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
        seed_scores.append(neural_score(model, tokens, lexical, 1024, device))
        checkpoints.append({**checkpoint, "verified_sha256": actual_hash})
        del model, tokens, lexical; gc.collect(); torch.cuda.empty_cache()
    scores = np.mean(np.stack(seed_scores), axis=0)

    threshold = float(model_spec["threshold"])
    offset = 0; scenarios = {}
    for name, spec in SCENARIOS.items():
        labels = datasets[name]["labels"]; length = len(labels)
        real_metrics = derived(classification_metrics(labels, scores[offset:offset+length], threshold)); offset += length
        gan_payload = load_json(spec["gan10_result"])
        if gan_payload["dataset"]["sha256"] != datasets[name]["report"]["dataset_sha256"]:
            raise RuntimeError(f"GAN comparison dataset mismatch for {name}")
        gan_metrics = gan_payload["metrics"]
        def operational(m: dict) -> dict:
            cm = m["confusion_matrix"]; multiplier = 10_000 / length
            return {"false_alerts_per_10000": cm["fp"] * multiplier,
                    "detected_phish_per_10000": cm["tp"] * multiplier,
                    "missed_phish_per_10000": cm["fn"] * multiplier}
        scenarios[name] = {
            "label": spec["label"], "dataset": str(spec["dataset"]),
            "dataset_sha256": datasets[name]["report"]["dataset_sha256"],
            "rows": length, "legitimate_rows": int((labels == 0).sum()),
            "phishing_rows": int(labels.sum()), "phishing_prevalence": float(labels.mean()),
            "real_only": {"metrics": real_metrics, "operational": operational(real_metrics)},
            "gan_10pct": {"metrics": gan_metrics, "operational": operational(gan_metrics)},
        }

    gan_wins = sum(scenarios[s]["gan_10pct"]["metrics"]["f1"] > scenarios[s]["real_only"]["metrics"]["f1"] for s in scenarios)
    conclusion = {
        "heading": "+10% GAN WINS BOTH DEPLOYMENT-SCENARIO F1 COMPARISONS" if gan_wins == 2 else "RESULTS ARE MIXED ACROSS PREVALENCE SCENARIOS",
        "text": "The low-dose GAN-augmented weighted hybrid is compared against its real-only counterpart with the same architecture family, seed ensemble, external datasets, and equal-calibration protocol. Thresholds remain the separately frozen values selected before these scenarios.",
        "points": [
            "Use F1, recall, FPR, and ROC-AUC together; precision changes substantially with phishing prevalence.",
            "Any gain belongs to the complete +10% training condition, not automatically to GAN novelty alone.",
            "The next honest test would require a new untouched temporal cohort; do not tune either threshold on these two datasets.",
        ],
    }
    payload = {
        "project_version": 3,
        "status": "retrospective controlled comparison on previously evaluated deployment cohorts",
        "real_only_model": model_spec["id"], "real_only_threshold": threshold,
        "real_only_threshold_policy": model_spec["threshold_policy"], "checkpoints": checkpoints,
        "gan_10pct_model": selection["winner"]["id"], "gan_10pct_threshold": selection["winner"]["threshold"],
        "scenarios": scenarios, "conclusion": conclusion,
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "comparison.json", payload)
    make_pdf(OUTPUT / "real_only_vs_gan10_browser_scenarios.pdf", payload)
    print(json.dumps({"output": str(OUTPUT), "scenarios": {name: {
        "real_only": record["real_only"]["metrics"], "gan_10pct": record["gan_10pct"]["metrics"]
    } for name, record in scenarios.items()}}, indent=2))


if __name__ == "__main__":
    main()
