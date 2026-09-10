from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import load_json, require_cuda, write_json
from .metrics import classification_metrics
from .models import create_classifier
from .url_data import encode_url


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen LSTM models on an external snapshot.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/external/2026-08-09"))
    return parser.parse_args()


@torch.inference_mode()
def score_checkpoint(path: Path, tokens: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_classifier("lstm", len(checkpoint["vocabulary"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens.astype(np.int64))), batch_size=512)
    scores = []
    for (batch,) in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scores.append(torch.sigmoid(model(batch.to(device))).float().cpu().numpy())
    return np.concatenate(scores), checkpoint


def main() -> None:
    args = parse_args(); device = require_cuda(); project = load_json(Path("configs/project.json"))
    vocabulary = load_json(Path("data/processed/vocabulary.json")); maximum = int(project["data"]["maximum_encoded_length"])
    rows = list(csv.DictReader(args.dataset.open("r", encoding="utf-8", newline="")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    tokens = np.stack([encode_url(row["url"], vocabulary, maximum) for row in rows])
    families = {
        "Real-only LSTM": Path("artifacts/baselines/lstm"),
        "SeqGAN 10% LSTM": Path("artifacts/augmentation/seqgan/generated_10_percent/lstm"),
    }
    results = []; prediction_columns = {"url": [row["url"] for row in rows], "label": labels.tolist()}
    for family, base in families.items():
        for seed in (42, 43, 44):
            path = base / f"seed_{seed}" / "checkpoint.pt"
            scores, checkpoint = score_checkpoint(path, tokens, device)
            threshold = float(checkpoint["threshold"]); metrics = classification_metrics(labels, scores, threshold)
            results.append({"family": family, "seed": seed, "threshold": threshold, **metrics})
            key = family.lower().replace(" ", "_").replace("-", "_").replace("%", "percent")
            prediction_columns[f"{key}_seed_{seed}_score"] = scores.tolist()
            prediction_columns[f"{key}_seed_{seed}_prediction"] = (scores >= threshold).astype(np.uint8).tolist()
    summary = {"project_version": 2, "dataset": str(args.dataset), "rows": len(rows), "device": torch.cuda.get_device_name(device), "families": {}}
    grouped = defaultdict(list)
    for row in results: grouped[row["family"]].append(row)
    for family, family_rows in grouped.items():
        summary["families"][family] = {
            "seeds": [row["seed"] for row in family_rows],
            "metrics": {metric: {"mean": float(np.mean([row[metric] for row in family_rows])), "std": float(np.std([row[metric] for row in family_rows], ddof=1)), "values": [row[metric] for row in family_rows]} for metric in METRICS},
            "confusion_matrix_mean": {key: float(np.mean([row["confusion_matrix"][key] for row in family_rows])) for key in ("tn", "fp", "fn", "tp")},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True); write_json(args.output_dir / "summary.json", summary); write_json(args.output_dir / "runs.json", results)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(prediction_columns.keys()); writer.writerows(zip(*prediction_columns.values()))
    x = np.arange(len(METRICS)); width = .36; fig, axis = plt.subplots(figsize=(10.5, 5.8))
    colors = ("#4a5568", "#2b6cb0")
    for index, (family, payload) in enumerate(summary["families"].items()):
        means = [100 * payload["metrics"][metric]["mean"] for metric in METRICS]
        errors = [100 * payload["metrics"][metric]["std"] for metric in METRICS]
        bounded_errors = np.asarray([
            [min(mean, error) for mean, error in zip(means, errors)],
            [min(100 - mean, error) for mean, error in zip(means, errors)],
        ])
        axis.bar(x + (index - .5) * width, means, width, yerr=bounded_errors, capsize=4, label=family, color=colors[index])
    axis.set_xticks(x, [metric.replace("_", " ").upper() for metric in METRICS]); axis.set_ylabel("Score (%)")
    axis.set_title("External snapshot — frozen three-seed LSTM comparison"); axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "external_metrics.png", dpi=220, bbox_inches="tight"); fig.savefig(args.output_dir / "external_metrics.pdf", bbox_inches="tight"); plt.close(fig)
    lines = ["# External Snapshot Evaluation", "", f"Dataset: **{len(rows)} URLs** ({int((labels==0).sum())} legitimate, {int((labels==1).sum())} phishing).", "",
             "Frozen internal-validation thresholds were used; no retraining or external tuning was performed.", "", "| Model family | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for family, payload in summary["families"].items():
        values = [payload["metrics"][metric]["mean"] for metric in METRICS]
        lines.append(f"| {family} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines.extend(("", "## Per-seed results", "", "| Model family | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|"))
    for row in results:
        lines.append(f'| {row["family"]} | {row["seed"]} | ' + " | ".join(f'{row[metric]:.6f}' for metric in METRICS) + " |")
    lines.extend(("", "Values in the first table are means across seeds 42, 43, and 44; error bars in the graph are one standard deviation.", "",
                  "Caution: Tranco root URLs are legitimate proxies and are structurally easier than arbitrary benign deep links. This is a small temporal snapshot, not a population-wide estimate."))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
