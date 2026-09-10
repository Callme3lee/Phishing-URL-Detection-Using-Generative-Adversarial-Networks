from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v1_newest_frozen_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import encode_url, require_cuda
from .models import create_character_model
from .train_classifier import metrics


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DATASET = PROJECT_ROOT / "Version_3/data/external/2026-08-20-long-url-raw/evaluation_urls.csv"
COLLECTION_REPORT = PROJECT_ROOT / "Version_3/data/external/2026-08-20-long-url-raw/collection_report.json"
OUTPUT = ROOT / "artifacts/newest_frozen_evaluation_2026-08-20/real_only"
CHECKPOINTS = {
    42: ROOT / "artifacts/robust_augmentation/results/real_only/cnn/best.pt",
    43: ROOT / "artifacts/robust_augmentation/results_seed43/real_only/cnn/best.pt",
    44: ROOT / "artifacts/robust_augmentation/results_seed44/real_only/cnn/best.pt",
}
METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def score_checkpoint(checkpoint_path: Path, urls: list[str], labels: np.ndarray,
                     device: torch.device, batch_size: int = 1024) -> tuple[dict, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    tokens = np.stack([
        encode_url(url, metadata["vocab"], int(metadata["max_length"])) for url in urls
    ]).astype(np.int64)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    model = create_character_model(checkpoint["model_type"], **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scores = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    probabilities = np.concatenate(scores)
    threshold = float(checkpoint["threshold"])
    result = metrics(labels, probabilities, threshold)
    result["false_negative_rate"] = 1.0 - float(result["recall"])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, probabilities


def main() -> None:
    collection = json.loads(COLLECTION_REPORT.read_text(encoding="utf-8"))
    actual_hash = sha256_file(DATASET)
    if actual_hash != collection["dataset_sha256"]:
        raise RuntimeError("The newest frozen dataset hash does not match its collection report")
    with DATASET.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    device = require_cuda()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict] = {}
    prediction_columns: dict[str, list] = {"url": urls, "label": labels.tolist()}
    for seed, checkpoint_path in CHECKPOINTS.items():
        result, probabilities = score_checkpoint(checkpoint_path, urls, labels, device)
        per_seed[str(seed)] = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics": result,
        }
        prediction_columns[f"seed_{seed}_score"] = probabilities.tolist()
        prediction_columns[f"seed_{seed}_prediction"] = (
            probabilities >= result["threshold"]
        ).astype(np.uint8).tolist()
        print(json.dumps({"seed": seed, **result}), flush=True)

    summary: dict[str, dict] = {}
    for metric in METRIC_KEYS:
        values = np.asarray([per_seed[str(seed)]["metrics"][metric] for seed in CHECKPOINTS])
        summary[metric] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    report = {
        "project_version": 1,
        "condition": "real-only CNN",
        "evaluation_role": "frozen Version 1 checkpoints evaluated on the newest frozen Version 3 URL set",
        "dataset": str(DATASET.relative_to(PROJECT_ROOT)),
        "dataset_sha256": actual_hash,
        "rows": len(rows),
        "legitimate_rows": int((labels == 0).sum()),
        "phishing_rows": int((labels == 1).sum()),
        "phishing_prevalence": float(labels.mean()),
        "checkpoints_retrained": False,
        "thresholds_retuned": False,
        "device": str(device),
        "per_seed": per_seed,
        "mean_across_seeds": summary,
        "interpretation_limit": collection["limitations"][2],
    }
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUTPUT / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(prediction_columns)
        writer.writerows(zip(*prediction_columns.values()))

    display = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
               "false_positive_rate", "false_negative_rate")
    titles = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR")
    values = [100 * summary[key]["mean"] for key in display]
    errors = [100 * summary[key]["sample_standard_deviation"] for key in display]
    colors = ["#3B6EA8"] * 6 + ["#D9534F", "#E67E22"]
    fig, axis = plt.subplots(figsize=(13, 6.5))
    bars = axis.bar(titles, values, yerr=errors, capsize=4, color=colors)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=9)
    axis.set_ylim(0, max(105, max(value + error for value, error in zip(values, errors)) + 8))
    axis.set_ylabel("Mean across seeds 42/43/44 (%)")
    axis.set_title("Version 1 real-only CNN on newest frozen long-URL dataset", fontweight="bold")
    axis.grid(axis="y", alpha=.25)
    fig.text(.5, .015, "1,400 legitimate + 150 phishing URLs; frozen checkpoints and thresholds; error bars = seed SD",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT / "metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), "mean_across_seeds": summary}, indent=2))


if __name__ == "__main__":
    main()
