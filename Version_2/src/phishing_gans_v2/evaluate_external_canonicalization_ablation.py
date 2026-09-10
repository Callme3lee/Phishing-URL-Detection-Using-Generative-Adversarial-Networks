from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_json, write_json
from .metrics import classification_metrics
from .models import HybridCharacterCNN, create_classifier
from .url_data import encode_url, lexical_features


MODELS = ("cnn", "lstm", "random_forest", "linear_svm", "hybrid_cnn")
DISPLAY = ("CNN", "LSTM", "Random Forest", "Linear SVM", "Hybrid CNN")
VARIANTS = ("as_collected", "scheme_stripped", "training_style_canonical")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")


def variants(url: str) -> dict[str, str]:
    parsed = urlsplit(url)
    host = parsed.netloc
    stripped = host + (parsed.path or "") + (f"?{parsed.query}" if parsed.query else "") + (f"#{parsed.fragment}" if parsed.fragment else "")
    path = "" if parsed.path == "/" else (parsed.path or "")
    training_style = host + path + (f"?{parsed.query}" if parsed.query else "") + (f"#{parsed.fragment}" if parsed.fragment else "")
    return {"as_collected": url, "scheme_stripped": stripped, "training_style_canonical": training_style}


@torch.inference_mode()
def neural_score(model: torch.nn.Module, tokens: np.ndarray, lexical: np.ndarray | None = None) -> np.ndarray:
    model.eval(); output = []
    for start in range(0, len(tokens), 256):
        batch_tokens = torch.from_numpy(tokens[start:start + 256].astype(np.int64, copy=False))
        if lexical is None:
            logits = model(batch_tokens)
        else:
            batch_lexical = torch.from_numpy(lexical[start:start + 256].astype(np.float32, copy=False))
            logits = model(batch_tokens, batch_lexical)
        output.append(torch.sigmoid(logits).numpy())
    return np.concatenate(output)


def add_fnr(metrics: dict) -> dict:
    return {**metrics, "false_negative_rate": 1.0 - float(metrics["recall"])}


def main() -> None:
    project = load_json(Path("configs/project.json")); master_hash = project["data"]["master_sha256"]
    rows = list(csv.DictReader(Path("data/external/2026-08-11/external_urls.csv").open(encoding="utf-8")))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    urls = {name: [variants(row["url"])[name] for row in rows] for name in VARIANTS}
    vocabulary = load_json(Path("data/processed/vocabulary.json")); maximum = project["data"]["maximum_encoded_length"]
    tokens = {name: np.stack([encode_url(url, vocabulary, maximum) for url in values]) for name, values in urls.items()}
    features = {name: np.stack([lexical_features(url) for url in values]).astype(np.float32) for name, values in urls.items()}
    records = []
    for model_name in MODELS:
        for seed in (42, 43, 44):
            if model_name in ("cnn", "lstm"):
                checkpoint = torch.load(Path("artifacts/baselines") / model_name / f"seed_{seed}" / "checkpoint.pt", map_location="cpu", weights_only=False)
                model = create_classifier(model_name, len(vocabulary), checkpoint["model_config"])
                model.load_state_dict(checkpoint["state_dict"]); threshold = float(checkpoint["threshold"])
            elif model_name in ("random_forest", "linear_svm"):
                checkpoint = joblib.load(Path("artifacts/baselines") / model_name / f"seed_{seed}" / "model.joblib")
                model = checkpoint["model"]; threshold = float(checkpoint["threshold"])
            else:
                checkpoint = torch.load(Path("artifacts/hybrid_cnn") / f"seed_{seed}" / "checkpoint.pt", map_location="cpu", weights_only=False)
                config = checkpoint["model_config"]
                model = HybridCharacterCNN(
                    vocabulary_size=len(vocabulary), lexical_feature_count=len(checkpoint["feature_names"]),
                    embedding_dim=config["embedding_dim"], channels_per_kernel=config["channels_per_kernel"],
                    kernel_sizes=tuple(config["kernel_sizes"]), lexical_hidden_dim=config["lexical_hidden_dim"],
                    fusion_hidden_dim=config["fusion_hidden_dim"], dropout=config["dropout"],
                )
                model.load_state_dict(checkpoint["state_dict"]); threshold = float(checkpoint["threshold"])
            if checkpoint["master_sha256"] != master_hash:
                raise RuntimeError("Checkpoint master hash mismatch")
            for variant in VARIANTS:
                if model_name in ("cnn", "lstm"):
                    scores = neural_score(model, tokens[variant])
                elif model_name == "random_forest":
                    scores = model.predict_proba(features[variant])[:, 1]
                elif model_name == "linear_svm":
                    scores = model.decision_function(features[variant])
                else:
                    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
                    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
                    scores = neural_score(model, tokens[variant], (features[variant] - mean) / std)
                records.append({
                    "model": model_name, "seed": seed, "variant": variant,
                    "threshold": threshold, **add_fnr(classification_metrics(labels, scores, threshold)),
                })
            del model, checkpoint; gc.collect()
    grouped = {}
    for model in MODELS:
        grouped[model] = {}
        for variant in VARIANTS:
            selected = [row for row in records if row["model"] == model and row["variant"] == variant]
            grouped[model][variant] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in selected])),
                    "sd": float(np.std([row[metric] for row in selected], ddof=1)),
                    "values": [float(row[metric]) for row in selected],
                } for metric in METRICS
            }
    output_dir = Path("artifacts/results/distribution_shift")
    write_json(output_dir / "canonicalization_ablation.json", {
        "project_version": 2,
        "purpose": "diagnostic no-retraining ablation; not model tuning",
        "dataset_rows": len(rows), "variants": list(VARIANTS), "records": records, "grouped": grouped,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    x = np.arange(len(MODELS)); width = .25
    colors = ("#c53030", "#dd6b20", "#319795")
    for axis, metric in zip(axes, ("roc_auc", "recall", "false_positive_rate")):
        for index, variant in enumerate(VARIANTS):
            values = [100 * grouped[model][variant][metric]["mean"] for model in MODELS]
            axis.bar(x + (index - 1) * width, values, width, label=variant.replace("_", " "), color=colors[index])
        axis.set_xticks(x, DISPLAY, rotation=25, ha="right"); axis.set_ylim(0, 100)
        axis.set_title(metric.replace("_", " ").upper()); axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Metric (%)"); axes[0].legend(fontsize=8)
    fig.suptitle("External canonicalization ablation — frozen real-only models, no retraining")
    fig.tight_layout(); fig.savefig(output_dir / "canonicalization_ablation.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "canonicalization_ablation.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps({model: {variant: {m: round(grouped[model][variant][m]["mean"], 6) for m in ("roc_auc", "recall", "false_positive_rate")} for variant in VARIANTS} for model in MODELS}, indent=2))


if __name__ == "__main__":
    main()
