from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_generated_challenge import MODELS, NEURAL, DISPLAY, neural_scores
from .metrics import classification_metrics
from .models import create_classifier


CONDITIONS = {
    "real_only": Path("artifacts/baselines"),
    "random_10_percent": Path("artifacts/augmentation/seqgan/generated_10_percent"),
    "realism_filtered_10_percent": Path("artifacts/augmentation/seqgan/realism_filtered_10_percent"),
    "realism_challenging_10_percent": Path("artifacts/augmentation/seqgan/realism_challenging_10_percent"),
}
CONDITION_DISPLAY = {
    "real_only": "Real only",
    "random_10_percent": "Random 10%",
    "realism_filtered_10_percent": "Realism-filtered 10%",
    "realism_challenging_10_percent": "Filtered + challenging 10%",
}
METRICS = ("recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate", "accuracy", "precision")
PRIMARY = ("recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the controlled quality-selection experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--challenge-manifest", type=Path, default=Path("data/challenges/seqgan_quality_holdout/challenge.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/quality_experiment"))
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def checkpoint_path(root: Path, model: str, seed: int) -> Path:
    suffix = "checkpoint.pt" if model in NEURAL else "model.joblib"
    return root / model / f"seed_{seed}" / suffix


def load_model(path: Path, model_name: str, device: torch.device) -> tuple[object, float]:
    if model_name in NEURAL:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = create_classifier(
            model_name, len(checkpoint["vocabulary"]), checkpoint["model_config"]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        checkpoint = joblib.load(path); model = checkpoint["model"]
    return model, float(checkpoint["threshold"])


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def paired(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean()); sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return {"mean_delta": mean, "sd_delta": sd, "paired_95_ci": [mean - half, mean + half]}


def add_fnr(metrics: dict) -> dict:
    return {**metrics, "false_negative_rate": 1.0 - float(metrics["recall"])}


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    seeds = tuple(int(value) for value in project["seeds"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    challenge = load_json(args.challenge_manifest)
    challenge_tokens = np.load(challenge["tokens"], mmap_mode="r")
    challenge_features = np.load(challenge["features"], mmap_mode="r")
    if len(challenge_tokens) != int(challenge["rows"]):
        raise RuntimeError("Challenge row count mismatch")
    if sha256_file(Path(challenge["tokens"])) != challenge["tokens_sha256"]:
        raise RuntimeError("Frozen challenge token hash mismatch")
    device = require_cuda()

    records: list[dict] = []
    challenge_predictions: dict[str, np.ndarray] = {}
    for model_name in MODELS:
        for seed in seeds:
            real_model_path = checkpoint_path(CONDITIONS["real_only"], model_name, seed)
            baseline_model, locked_threshold = load_model(real_model_path, model_name, device)
            # Release the temporary real-only load; each condition is loaded
            # independently below, including real-only itself.
            del baseline_model
            gc.collect(); torch.cuda.empty_cache()
            for condition, root in CONDITIONS.items():
                path = checkpoint_path(root, model_name, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                model, native_threshold = load_model(path, model_name, device)
                prediction_path = root / model_name / f"seed_{seed}" / "test_predictions.npz"
                with np.load(prediction_path) as stored:
                    labels = stored["labels"].astype(np.uint8)
                    real_scores = stored["scores"]
                locked_metrics = add_fnr(classification_metrics(labels, real_scores, locked_threshold))
                native_metrics = add_fnr(classification_metrics(labels, real_scores, native_threshold))
                if model_name in NEURAL:
                    challenge_scores = neural_scores(model, challenge_tokens, args.batch_size, device)
                elif model_name == "random_forest":
                    challenge_scores = model.predict_proba(challenge_features)[:, 1].astype(np.float32)
                else:
                    challenge_scores = model.decision_function(challenge_features).astype(np.float32)
                challenge_detection_locked = float((challenge_scores >= locked_threshold).mean())
                challenge_detection_native = float((challenge_scores >= native_threshold).mean())
                key = f"{condition}__{model_name}__seed_{seed}"
                challenge_predictions[key] = (challenge_scores >= locked_threshold).astype(np.uint8)
                records.append({
                    "condition": condition,
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256_file(path),
                    "locked_real_only_threshold": locked_threshold,
                    "native_validation_threshold": native_threshold,
                    "real_test_locked": locked_metrics,
                    "real_test_native": native_metrics,
                    "challenge_detection_recall_locked": challenge_detection_locked,
                    "challenge_fnr_locked": 1.0 - challenge_detection_locked,
                    "challenge_detection_recall_native": challenge_detection_native,
                    "challenge_fnr_native": 1.0 - challenge_detection_native,
                })
                print(json.dumps({
                    "condition": condition,
                    "model": model_name,
                    "seed": seed,
                    "real_test_f1_locked": locked_metrics["f1"],
                    "challenge_recall_locked": challenge_detection_locked,
                }), flush=True)
                del model, challenge_scores
                gc.collect(); torch.cuda.empty_cache()

    grouped: dict[str, dict] = {}
    for model_name in MODELS:
        grouped[model_name] = {}
        for condition in CONDITIONS:
            selected = [r for r in records if r["model"] == model_name and r["condition"] == condition]
            result = {
                "seeds": list(seeds),
                "real_test_locked": {
                    metric: mean_sd([float(r["real_test_locked"][metric]) for r in selected])
                    for metric in METRICS
                },
                "real_test_native": {
                    metric: mean_sd([float(r["real_test_native"][metric]) for r in selected])
                    for metric in METRICS
                },
                "synthetic_challenge_locked": {
                    "recall": mean_sd([r["challenge_detection_recall_locked"] for r in selected]),
                    "false_negative_rate": mean_sd([r["challenge_fnr_locked"] for r in selected]),
                },
                "synthetic_challenge_native": {
                    "recall": mean_sd([r["challenge_detection_recall_native"] for r in selected]),
                    "false_negative_rate": mean_sd([r["challenge_fnr_native"] for r in selected]),
                },
            }
            result["paired_deltas"] = {}
            for reference in ("real_only", "random_10_percent"):
                reference_rows = [r for r in records if r["model"] == model_name and r["condition"] == reference]
                by_seed = {int(r["seed"]): r for r in reference_rows}
                result["paired_deltas"][reference] = {
                    "real_test_locked": {
                        metric: paired([
                            float(r["real_test_locked"][metric]) - float(by_seed[int(r["seed"])]["real_test_locked"][metric])
                            for r in selected
                        ]) for metric in METRICS
                    },
                    "synthetic_challenge_locked": {
                        "recall": paired([
                            r["challenge_detection_recall_locked"] - by_seed[int(r["seed"])]["challenge_detection_recall_locked"]
                            for r in selected
                        ]),
                        "false_negative_rate": paired([
                            r["challenge_fnr_locked"] - by_seed[int(r["seed"])]["challenge_fnr_locked"]
                            for r in selected
                        ]),
                    },
                }
            grouped[model_name][condition] = result

    payload = {
        "project_version": 2,
        "protocol": "four-condition controlled synthetic quality-selection experiment",
        "conditions": list(CONDITIONS),
        "source_generator_for_all_augmented_conditions": "seqgan",
        "generated_rows_per_augmented_condition": 228425,
        "seeds": list(seeds),
        "primary_threshold_protocol": "same frozen real-only model/seed threshold applied across all four conditions",
        "secondary_threshold_protocol": "each condition's existing validation-selected threshold",
        "real_test": "unchanged frozen real test split",
        "synthetic_challenge": challenge,
        "synthetic_metric_warning": "phishing-only synthetic challenge supports recall/FNR, not accuracy, precision, FPR, or ROC-AUC",
        "grouped": grouped,
        "per_run": records,
    }
    write_json(args.output_dir / "summary.json", payload)
    np.savez_compressed(args.output_dir / "challenge_predictions.npz", **challenge_predictions)

    with (args.output_dir / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "model", "seed", "locked_real_only_threshold", "native_validation_threshold"]
        fields += [f"real_locked_{metric}" for metric in METRICS]
        fields += ["challenge_recall_locked", "challenge_fnr_locked"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({
                "condition": row["condition"], "model": row["model"], "seed": row["seed"],
                "locked_real_only_threshold": row["locked_real_only_threshold"],
                "native_validation_threshold": row["native_validation_threshold"],
                **{f"real_locked_{metric}": row["real_test_locked"][metric] for metric in METRICS},
                "challenge_recall_locked": row["challenge_detection_recall_locked"],
                "challenge_fnr_locked": row["challenge_fnr_locked"],
            })

    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    x = np.arange(len(CONDITIONS)); labels = [CONDITION_DISPLAY[c] for c in CONDITIONS]
    for model_name in MODELS:
        fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.2)); axes = axes.ravel()
        for axis, metric in zip(axes, METRICS):
            means = [grouped[model_name][c]["real_test_locked"][metric]["mean"] for c in CONDITIONS]
            errors = [grouped[model_name][c]["real_test_locked"][metric]["sd"] for c in CONDITIONS]
            axis.bar(x, means, yerr=errors, capsize=4, color=colors)
            axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
            axis.set_xticks(x, labels, rotation=22, ha="right")
            axis.grid(axis="y", alpha=.25)
        challenge_axis = axes[-1]
        means = [grouped[model_name][c]["synthetic_challenge_locked"]["recall"]["mean"] for c in CONDITIONS]
        errors = [grouped[model_name][c]["synthetic_challenge_locked"]["recall"]["sd"] for c in CONDITIONS]
        challenge_axis.bar(x, means, yerr=errors, capsize=4, color=colors)
        challenge_axis.set_title("HELD-OUT SYNTHETIC RECALL", fontweight="bold")
        challenge_axis.set_xticks(x, labels, rotation=22, ha="right"); challenge_axis.grid(axis="y", alpha=.25)
        fig.suptitle(f"{DISPLAY[model_name]} — controlled 10% quality-selection experiment\nmean ± SD, common real-only thresholds", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.output_dir / f"{model_name}_all_metrics.png", dpi=220, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{model_name}_all_metrics.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6)); width = .19
    for model_index, model_name in enumerate(MODELS):
        means = [100 * grouped[model_name][c]["synthetic_challenge_locked"]["recall"]["mean"] for c in CONDITIONS]
        errors = [100 * grouped[model_name][c]["synthetic_challenge_locked"]["recall"]["sd"] for c in CONDITIONS]
        axis.bar(x + (model_index - 1.5) * width, means, width, yerr=errors, capsize=3, label=DISPLAY[model_name])
    axis.set_xticks(x, labels); axis.set_ylabel("Detection recall on held-out generated phishing (%)")
    axis.set_title("Synthetic robustness: frozen disjoint 20,000-URL challenge", fontweight="bold")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "synthetic_challenge_recall.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "synthetic_challenge_recall.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6.5))
    markers = {"realism_filtered_10_percent": "o", "realism_challenging_10_percent": "s"}
    for condition, marker in markers.items():
        for model_name, color in zip(MODELS, colors):
            f1_delta = 100 * grouped[model_name][condition]["paired_deltas"]["random_10_percent"]["real_test_locked"]["f1"]["mean_delta"]
            challenge_delta = 100 * grouped[model_name][condition]["paired_deltas"]["random_10_percent"]["synthetic_challenge_locked"]["recall"]["mean_delta"]
            axis.scatter(f1_delta, challenge_delta, marker=marker, s=95, color=color)
            axis.annotate(f"{DISPLAY[model_name]} / {'filtered' if marker == 'o' else 'challenging'}", (f1_delta, challenge_delta), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=.8); axis.axvline(0, color="black", linewidth=.8)
    axis.set_xlabel("Real-test F1 change vs random 10% (percentage points)")
    axis.set_ylabel("Held-out synthetic recall change vs random 10% (percentage points)")
    axis.set_title("Quality selection trade-off relative to random SeqGAN augmentation", fontweight="bold")
    axis.grid(alpha=.25); fig.tight_layout()
    fig.savefig(args.output_dir / "real_vs_synthetic_tradeoff.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "real_vs_synthetic_tradeoff.pdf", bbox_inches="tight"); plt.close(fig)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "runs": len(records),
        "primary_threshold_protocol": payload["primary_threshold_protocol"],
    }, indent=2))


if __name__ == "__main__":
    main()
