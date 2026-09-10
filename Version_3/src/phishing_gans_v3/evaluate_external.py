from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .metrics import classification_metrics
from .models import create_classifier
from .url_data import encode_url, lexical_features


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
NEURAL = {"cnn", "lstm", "rnn"}
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "rnn": "RNN", "random_forest": "Random Forest", "linear_svm": "Linear SVM"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Version 3 real-only models on the frozen external snapshot.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/external/2026-08-11/external_urls.csv"))
    parser.add_argument("--collection-report", type=Path, default=Path("data/external/2026-08-11/collection_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/external/2026-08-11_topdomains_phishtank"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


@torch.inference_mode()
def neural_scores(model: torch.nn.Module, tokens: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    results = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.from_numpy(tokens[start:start + batch_size].astype(np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
        results.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(results)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    report = load_json(args.collection_report)
    if sha256_file(args.dataset) != report["dataset_sha256"]:
        raise RuntimeError("External snapshot hash does not match its frozen collection report")
    legacy_overlap = [report.get(key, 0) for key in (
        "master_exact_overlap", "master_encoded_overlap", "master_hostname_overlap", "prior_external_overlap"
    )]
    measured_overlap = list(report.get("measured_overlap", {}).values())
    if any(value != 0 for value in (*legacy_overlap, *measured_overlap)):
        raise RuntimeError("External collection report contains overlap")
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != report["rows"] or int((labels == 0).sum()) != report["legitimate_rows"] or int((labels == 1).sum()) != report["phishing_rows"]:
        raise RuntimeError("External row or label counts differ from the frozen collection report")
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(project["data"]["maximum_encoded_length"])
    tokens = np.stack([encode_url(row["url"], vocabulary, maximum) for row in rows])
    features = np.stack([lexical_features(row["url"]) for row in rows]).astype(np.float32)
    device = require_cuda()
    per_run = []
    prediction_columns: dict[str, list] = {
        "url": [row["url"] for row in rows], "label": labels.tolist(),
        "source": [row["source"] for row in rows], "source_id": [row["source_id"] for row in rows],
    }
    for model_name in MODELS:
        for seed in project["seeds"]:
            run_dir = args.checkpoint_dir / model_name / f"seed_{seed}"
            if model_name in NEURAL:
                checkpoint_path = run_dir / "checkpoint.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model = create_classifier(model_name, len(checkpoint["vocabulary"]), checkpoint["model_config"]).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                threshold = float(checkpoint["threshold"])
                scores = neural_scores(model, tokens, args.batch_size, device)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                checkpoint_path = run_dir / "model.joblib"
                checkpoint = joblib.load(checkpoint_path)
                model = checkpoint["model"]
                threshold = float(checkpoint["threshold"])
                scores = (
                    model.predict_proba(features)[:, 1]
                    if model_name == "random_forest" else model.decision_function(features)
                ).astype(np.float32)
            metrics = classification_metrics(labels, scores, threshold)
            record = {
                "model": model_name, "seed": int(seed), "threshold": threshold,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                "metrics": metrics,
            }
            per_run.append(record)
            key = f"{model_name}__seed_{seed}"
            prediction_columns[f"{key}__score"] = scores.tolist()
            prediction_columns[f"{key}__prediction"] = (scores >= threshold).astype(np.uint8).tolist()
            print(json.dumps({"model": model_name, "seed": seed, **{metric: metrics[metric] for metric in METRICS}}), flush=True)

    grouped = {}
    for model_name in MODELS:
        selected = [row for row in per_run if row["model"] == model_name]
        grouped[model_name] = {
            "seeds": [row["seed"] for row in selected],
            "metrics": {
                metric: {**mean_sd([float(row["metrics"][metric]) for row in selected]),
                         "values": [float(row["metrics"][metric]) for row in selected]}
                for metric in METRICS
            },
            "mean_confusion_matrix": {
                key: float(np.mean([row["metrics"]["confusion_matrix"][key] for row in selected]))
                for key in ("tn", "fp", "fn", "tp")
            },
        }
    source_names = {
        key: value.get("name", key) for key, value in report.get("sources", {}).items()
    }
    summary = {
        "project_version": 3,
        "protocol": report.get("purpose", "frozen external evaluation"),
        "threshold_policy": "unchanged internal-validation threshold from each checkpoint; no external calibration",
        "dataset": str(args.dataset), "dataset_sha256": report["dataset_sha256"],
        "rows": len(rows), "legitimate_rows": int((labels == 0).sum()),
        "phishing_rows": int((labels == 1).sum()), "models": list(MODELS),
        "seeds": list(project["seeds"]), "source_names": source_names, "collection_report": report,
        "grouped": grouped, "per_run": per_run,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_run.json", per_run)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(prediction_columns)
        writer.writerows(zip(*prediction_columns.values()))
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in MODELS:
            writer.writerow({
                "model": model_name,
                **{f"{metric}_mean": grouped[model_name]["metrics"][metric]["mean"] for metric in METRICS},
                **{f"{metric}_sd": grouped[model_name]["metrics"][metric]["standard_deviation"] for metric in METRICS},
            })

    x = np.arange(len(MODELS))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        means = [100 * grouped[model]["metrics"][metric]["mean"] for model in MODELS]
        errors = [100 * grouped[model]["metrics"][metric]["standard_deviation"] for model in MODELS]
        axis.bar(x, means, yerr=errors, capsize=4, color=colors)
        axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in MODELS], rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric not in {"false_positive_rate", "false_negative_rate"}:
            axis.set_ylim(0, 100)
    fig.suptitle(
        f"Version 3 external evaluation — {args.checkpoint_dir.name}\n"
        f"{report['legitimate_rows']} legitimate + {report['phishing_rows']} phishing; mean ± SD",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "all_models_all_metrics.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_models_all_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Version 3 External Evaluation", "",
        f"This frozen evaluation contains {report['legitimate_rows']} legitimate and {report['phishing_rows']} phishing proxy-labelled rows.", "",
        "Sources: " + "; ".join(f"{key}: {value}" for key, value in source_names.items()) + ".", "",
        "Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.", "",
        "Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.", "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        metrics = grouped[model_name]["metrics"]
        lines.append("| " + DISPLAY[model_name] + " | " + " | ".join(
            f'{metrics[metric]["mean"]:.4f} ± {metrics[metric]["standard_deviation"]:.4f}' for metric in METRICS
        ) + " |")
    lines.extend(("", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "evaluations": len(per_run), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
