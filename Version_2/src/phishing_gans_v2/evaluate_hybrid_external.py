from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import load_json, require_cuda, write_json
from .metrics import classification_metrics
from .models import HybridCharacterCNN
from .url_data import encode_url, lexical_features


def main() -> None:
    dataset_path = Path("data/external/2026-08-11/external_urls.csv")
    checkpoint_dir = Path("artifacts/hybrid_cnn")
    existing_path = Path("artifacts/results/external/2026-08-11_phishtank_topmillion/summary.json")
    output_dir = Path("artifacts/results/hybrid_cnn/external_2026-08-11")
    rows = list(csv.DictReader(dataset_path.open(encoding="utf-8")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    device = require_cuda(); runs = []
    for seed in (42, 43, 44):
        checkpoint = torch.load(checkpoint_dir / f"seed_{seed}" / "checkpoint.pt", map_location="cpu", weights_only=False)
        vocabulary = checkpoint["vocabulary"]; config = checkpoint["model_config"]
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        tokens = np.stack([encode_url(row["url"], vocabulary, 128) for row in rows]).astype(np.int64)
        lexical = np.stack([lexical_features(row["url"]) for row in rows]).astype(np.float32)
        lexical = (lexical - mean) / std
        loader = DataLoader(TensorDataset(
            torch.from_numpy(tokens), torch.from_numpy(lexical)
        ), batch_size=256)
        model = HybridCharacterCNN(
            vocabulary_size=len(vocabulary), lexical_feature_count=len(mean),
            embedding_dim=config["embedding_dim"], channels_per_kernel=config["channels_per_kernel"],
            kernel_sizes=tuple(config["kernel_sizes"]), lexical_hidden_dim=config["lexical_hidden_dim"],
            fusion_hidden_dim=config["fusion_hidden_dim"], dropout=config["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval(); scores = []
        with torch.inference_mode():
            for batch_tokens, batch_lexical in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch_tokens.to(device), batch_lexical.to(device))
                scores.append(torch.sigmoid(logits).float().cpu().numpy())
        scores_array = np.concatenate(scores)
        metrics = classification_metrics(labels, scores_array, float(checkpoint["threshold"]))
        matrix = metrics["confusion_matrix"]
        metrics["false_negative_rate"] = float(
            matrix["fn"] / max(1, matrix["fn"] + matrix["tp"])
        )
        runs.append({"seed": seed, "threshold": float(checkpoint["threshold"]), **metrics})
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")
    grouped = {
        metric: {
            "mean": float(np.mean([run[metric] for run in runs])),
            "sd": float(np.std([run[metric] for run in runs], ddof=1)),
            "values": [float(run[metric]) for run in runs],
        }
        for metric in metric_names
    }
    summary = {
        "project_version": 2, "model": "hybrid_cnn", "condition": "real_only",
        "dataset": str(dataset_path), "rows": len(rows),
        "threshold_source": "unchanged internal validation", "runs": runs, "metrics": grouped,
    }
    output_dir.mkdir(parents=True, exist_ok=True); write_json(output_dir / "summary.json", summary)
    existing = load_json(existing_path)["grouped"]
    models = ["cnn", "lstm", "random_forest", "linear_svm", "hybrid_cnn"]
    display = ["CNN", "LSTM", "Random Forest", "Linear SVM", "Hybrid CNN"]
    matrix = []
    for model in models[:-1]:
        matrix.append([100 * existing[model]["real_only"]["native_metrics"][metric]["mean"] for metric in metric_names])
    matrix.append([100 * grouped[metric]["mean"] for metric in metric_names])
    matrix = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(metric_names)), [m.replace("_", " ").upper() for m in metric_names], rotation=25, ha="right")
    axis.set_yticks(range(len(models)), display)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center")
    axis.set_title("Real-only models — frozen PhishTank/Top-Million snapshot (%)")
    fig.colorbar(image, ax=axis, label="Metric (%)"); fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
