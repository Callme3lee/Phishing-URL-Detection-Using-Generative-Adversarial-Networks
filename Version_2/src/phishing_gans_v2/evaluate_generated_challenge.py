from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t
from sklearn.neighbors import NearestNeighbors

from .common import load_json, require_cuda, set_seed, write_json
from .models import create_classifier
from .url_data import LEXICAL_FEATURE_NAMES


GENERATORS = ("gumbel_lsgan", "seqgan", "relgan")
MODELS = ("cnn", "lstm", "random_forest", "linear_svm")
NEURAL = {"cnn", "lstm"}
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "random_forest": "Random Forest", "linear_svm": "Linear SVM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge frozen real-only classifiers with accepted GAN URLs.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--sample-count", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/generated_challenge"))
    return parser.parse_args()


def neural_scores(model: torch.nn.Module, tokens: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    output = np.empty(len(tokens), dtype=np.float32); model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            stop = min(len(tokens), start + batch_size)
            batch = torch.from_numpy(np.asarray(tokens[start:stop], dtype=np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.float16): logits = model(batch)
            output[start:stop] = torch.sigmoid(logits).float().cpu().numpy()
    return output


def probability_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6); threshold = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    return np.log(clipped / (1 - clipped)) - math.log(threshold / (1 - threshold))


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = (.01, .05, .25, .5, .75, .95, .99)
    return {f"q{round(100*p):02d}": float(value) for p, value in zip(points, np.quantile(values, points))}


def paired_seed_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float); mean = float(array.mean())
    half = float(t.ppf(.975, len(array)-1) * array.std(ddof=1) / math.sqrt(len(array)))
    return [mean-half, mean+half]


def real_feature_statistics(feature_dir: Path, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.load(feature_dir / "train_features.npy", mmap_mode="r")
    indices = np.load(data_dir / "train_malicious_indices.npy", mmap_mode="r")
    total = np.zeros(features.shape[1], dtype=np.float64); squared = np.zeros_like(total); count = 0
    for start in range(0, len(indices), 100000):
        values = np.asarray(features[np.asarray(indices[start:start+100000])], dtype=np.float64)
        total += values.sum(0); squared += np.square(values).sum(0); count += len(values)
    mean = total/count; variance = np.maximum(1e-12, squared/count - np.square(mean))
    return mean, np.sqrt(variance), np.asarray(indices)


def read_urls(path: Path, count: int) -> tuple[list[str], list[str]]:
    urls, hosts = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            urls.append(row["url"]); hosts.append(row["hostname"])
            if len(urls) == count: break
    if len(urls) != count: raise RuntimeError(f"Expected {count:,} challenge URLs in {path}")
    return urls, hosts


def tld_like(hostname: str) -> bool:
    labels = hostname.rstrip(".").split(".")
    return len(labels) >= 2 and 2 <= len(labels[-1]) <= 24 and labels[-1].isalpha()


def top_shifts(mean: np.ndarray, real_mean: np.ndarray, real_std: np.ndarray, count: int = 8) -> list[dict]:
    standardized = (mean-real_mean)/np.maximum(real_std, 1e-6)
    order = np.argsort(-np.abs(standardized))[:count]
    return [{"feature": LEXICAL_FEATURE_NAMES[int(i)], "standardized_mean_difference": float(standardized[i])}
            for i in order]


def main() -> None:
    args = parse_args(); project = load_json(args.config); seeds = tuple(int(v) for v in project["seeds"])
    if args.sample_count <= 0: raise ValueError("sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True); device = require_cuda(); set_seed(int(project["seed"]))
    token_pools = {}; feature_pools = {}
    for generator in GENERATORS:
        token_pools[generator] = np.load(f"data/augmented/{generator}/generated_pool_tokens.npy", mmap_mode="r")[:args.sample_count]
        feature_pools[generator] = np.load(f"data/augmented/{generator}/generated_pool_features.npy", mmap_mode="r")[:args.sample_count]
        if len(token_pools[generator]) != args.sample_count: raise RuntimeError(f"Insufficient {generator} samples")

    score_arrays: dict[str, np.ndarray] = {}; prediction_arrays: dict[str, np.ndarray] = {}; records = []
    started = time.perf_counter()
    for model_name in MODELS:
        for seed in seeds:
            if model_name in NEURAL:
                checkpoint = torch.load(f"artifacts/baselines/{model_name}/seed_{seed}/checkpoint.pt",
                                        map_location=device, weights_only=False)
                if checkpoint["master_sha256"] != project["data"]["master_sha256"]: raise RuntimeError("Checkpoint hash mismatch")
                model = create_classifier(model_name, len(checkpoint["vocabulary"]), checkpoint["model_config"]).to(device)
                model.load_state_dict(checkpoint["state_dict"]); threshold = float(checkpoint["threshold"])
            else:
                checkpoint = joblib.load(f"artifacts/baselines/{model_name}/seed_{seed}/model.joblib")
                if checkpoint["master_sha256"] != project["data"]["master_sha256"]: raise RuntimeError("Checkpoint hash mismatch")
                model = checkpoint["model"]; threshold = float(checkpoint["threshold"])
            for generator in GENERATORS:
                if model_name in NEURAL:
                    scores = neural_scores(model, token_pools[generator], args.batch_size, device)
                elif model_name == "random_forest":
                    scores = model.predict_proba(feature_pools[generator])[:, 1].astype(np.float32)
                else:
                    scores = model.decision_function(feature_pools[generator]).astype(np.float32)
                predictions = scores >= threshold; key = f"{generator}__{model_name}__seed_{seed}"
                score_arrays[key] = scores; prediction_arrays[key] = predictions
                margin = (scores-threshold) if model_name == "linear_svm" else probability_margin(scores, threshold)
                detected = int(predictions.sum()); detection_rate = detected/args.sample_count
                record = {"generator": generator, "model": model_name, "seed": seed, "sample_count": args.sample_count,
                          "threshold": threshold, "detected_as_phishing": detected,
                          "detection_rate_recall": detection_rate, "false_negative_evasion_rate": 1-detection_rate,
                          "score_mean": float(scores.mean()), "score_quantiles": quantiles(scores),
                          "threshold_margin_mean": float(margin.mean()), "threshold_margin_quantiles": quantiles(margin)}
                records.append(record); print(json.dumps({k: record[k] for k in ("generator","model","seed","detection_rate_recall","false_negative_evasion_rate")}), flush=True)
            del model, checkpoint; gc.collect(); torch.cuda.empty_cache()

    grouped = {}
    for generator in GENERATORS:
        grouped[generator] = {}
        for model_name in MODELS:
            selected = [row for row in records if row["generator"] == generator and row["model"] == model_name]
            detection = [row["detection_rate_recall"] for row in selected]; evasion = [row["false_negative_evasion_rate"] for row in selected]
            grouped[generator][model_name] = {
                "seeds": list(seeds), "detection_rate_mean": float(np.mean(detection)),
                "detection_rate_95_t_interval": paired_seed_interval(detection),
                "evasion_fnr_mean": float(np.mean(evasion)), "evasion_fnr_95_t_interval": paired_seed_interval(evasion),
                "per_seed": selected,
            }

    real_mean, real_std, malicious_indices = real_feature_statistics(Path("data/features"), Path("data/processed"))
    train_features = np.load("data/features/train_features.npy", mmap_mode="r")
    rng = np.random.default_rng(int(project["seed"])); real_sample = rng.choice(np.asarray(malicious_indices), 12000, replace=False)
    reference = (np.asarray(train_features[real_sample[:10000]], dtype=np.float32)-real_mean)/np.maximum(real_std, 1e-6)
    real_query = (np.asarray(train_features[real_sample[10000:]], dtype=np.float32)-real_mean)/np.maximum(real_std, 1e-6)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(reference)
    real_nn_mean = float(nearest.kneighbors(real_query, return_distance=True)[0].mean())
    realism = {}; hard_rows = []
    for generator in GENERATORS:
        prediction_matrix = np.stack([prediction_arrays[f"{generator}__{model}__seed_{seed}"]
                                      for model in MODELS for seed in seeds])
        votes = prediction_matrix.sum(0); majority_evaded = votes <= len(MODELS)*len(seeds)//2 - 1
        unanimous_evaded = votes == 0; disagreement = (votes > 0) & (votes < prediction_matrix.shape[0])
        features = np.asarray(feature_pools[generator], dtype=np.float64)
        z = np.abs((features-real_mean)/np.maximum(real_std, 1e-6)); lexical_inlier = z.max(1) <= 5
        urls, hosts = read_urls(Path(f"data/augmented/{generator}/generated_pool_manifest.csv"), args.sample_count)
        tld_mask = np.fromiter((tld_like(host) for host in hosts), dtype=bool, count=args.sample_count)
        plausibly_hard = majority_evaded & lexical_inlier & tld_mask
        vote_counts = Counter(int(value) for value in votes)
        query_indices = rng.choice(args.sample_count, min(2000, args.sample_count), replace=False)
        generated_query = (features[query_indices]-real_mean)/np.maximum(real_std, 1e-6)
        generated_nn_mean = float(nearest.kneighbors(generated_query, return_distance=True)[0].mean())
        hard_indices = np.flatnonzero(majority_evaded)
        if len(hard_indices):
            hard_query_indices = rng.choice(hard_indices, min(2000, len(hard_indices)), replace=False)
            hard_query = (features[hard_query_indices]-real_mean)/np.maximum(real_std, 1e-6)
            hard_nn_mean = float(nearest.kneighbors(hard_query, return_distance=True)[0].mean())
        else: hard_nn_mean = None
        hard_mean = features[majority_evaded].mean(0) if majority_evaded.any() else np.full(features.shape[1], np.nan)
        realism[generator] = {
            "all_classifier_seed_votes": int(prediction_matrix.shape[0]),
            "detection_vote_histogram": {str(k): vote_counts.get(k, 0) for k in range(prediction_matrix.shape[0]+1)},
            "majority_evasion_rate": float(majority_evaded.mean()), "unanimous_evasion_rate": float(unanimous_evaded.mean()),
            "classifier_disagreement_rate": float(disagreement.mean()), "tld_like_rate": float(tld_mask.mean()),
            "lexical_inlier_rate_max_abs_z_le_5": float(lexical_inlier.mean()),
            "plausibly_hard_rate": float(plausibly_hard.mean()),
            "mean_nearest_lexical_distance_to_10000_real_phishing": generated_nn_mean,
            "hard_mean_nearest_lexical_distance_to_10000_real_phishing": hard_nn_mean,
            "real_to_real_reference_nearest_distance": real_nn_mean,
            "largest_all_generated_feature_shifts": top_shifts(features.mean(0), real_mean, real_std),
            "largest_hard_feature_shifts": top_shifts(hard_mean, real_mean, real_std) if majority_evaded.any() else [],
        }
        ranked = np.argsort(votes)[:200]
        for index in ranked:
            hard_rows.append({"generator": generator, "pool_index": int(index), "url": urls[int(index)],
                              "hostname": hosts[int(index)], "detection_votes_of_12": int(votes[index]),
                              "majority_evaded": bool(majority_evaded[index]), "tld_like": bool(tld_mask[index]),
                              "lexical_inlier": bool(lexical_inlier[index])})

    payload = {"project_version": 2, "protocol": "frozen real-only classifier challenge evaluation",
               "sample_count_per_generator": args.sample_count, "sample_slice": [0, args.sample_count],
               "thresholds": "unchanged from each real-only validation run", "synthetic_label_semantics": "intended phishing",
               "metric_warning": "phishing-only synthetic sets support recall/FNR, not accuracy, precision, ROC-AUC, or FPR",
               "elapsed_seconds": time.perf_counter()-started, "classifier_results": grouped, "realism_and_consensus": realism}
    write_json(args.output_dir / "summary.json", payload)
    with (args.output_dir / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("generator","model","seed","sample_count","threshold","detected_as_phishing","detection_rate_recall",
                  "false_negative_evasion_rate","score_mean","threshold_margin_mean")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records: writer.writerow({field: row[field] for field in fields})
    with (args.output_dir / "hard_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(hard_rows[0])); writer.writeheader(); writer.writerows(hard_rows)
    np.savez_compressed(args.output_dir / "predictions.npz", **{key: value.astype(np.uint8) for key, value in prediction_arrays.items()})

    matrix = np.asarray([[100*grouped[g][m]["evasion_fnr_mean"] for g in GENERATORS] for m in MODELS])
    fig, axis = plt.subplots(figsize=(8.6, 5)); image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(1, matrix.max()), aspect="auto")
    axis.set_xticks(range(len(GENERATORS)), ("Gumbel-LSGAN","SeqGAN","RelGAN")); axis.set_yticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]): axis.text(column,row,f"{matrix[row,column]:.2f}%",ha="center",va="center")
    axis.set_title("Generated phishing evasion rate (FNR)\nFrozen real-only classifiers, mean across seeds 42/43/44", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Generated URLs predicted legitimate (%)"); fig.tight_layout()
    fig.savefig(args.output_dir / "evasion_heatmap.png", dpi=220, bbox_inches="tight"); fig.savefig(args.output_dir / "evasion_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    x = np.arange(len(GENERATORS)); width = .2; fig, axis = plt.subplots(figsize=(10, 5.3))
    for model_index, model_name in enumerate(MODELS):
        axis.bar(x+(model_index-1.5)*width, [100*grouped[g][model_name]["evasion_fnr_mean"] for g in GENERATORS],
                 width, label=DISPLAY[model_name])
    axis.set_xticks(x, ("Gumbel-LSGAN","SeqGAN","RelGAN")); axis.set_ylabel("Evasion / false-negative rate (%)")
    axis.set_title("Do frozen real-only classifiers recognize generated phishing URLs?", fontweight="bold")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "evasion_by_classifier.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "evasion_by_classifier.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "sample_count_per_generator": args.sample_count,
        "elapsed_seconds": payload["elapsed_seconds"],
        "mean_evasion_fnr": {
            generator: {
                model_name: grouped[generator][model_name]["evasion_fnr_mean"]
                for model_name in MODELS
            }
            for generator in GENERATORS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
