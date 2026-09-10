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
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .common import load_json, require_cuda, set_seed, write_json
from .evaluate_external import neural_scores
from .models import create_classifier
from .url_data import encode_url, psl_aware_lexical_features


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
NEURAL = {"cnn", "lstm", "rnn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GAN realism, diversity, shift, and classifier challenge rate.")
    parser.add_argument("--model", choices=("seqgan", "relgan", "urlgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--generated-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/results/gan_quality"))
    parser.add_argument("--sample-size", type=int, default=20000)
    return parser.parse_args()


def source_auc(real: np.ndarray, fake: np.ndarray, seeds: list[int]) -> list[float]:
    values = np.vstack((real, fake)); labels = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    output = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            values, labels, test_size=.3, stratify=labels, random_state=seed)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
        model.fit(x_train, y_train); output.append(float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])))
    return output


def mmd(real: np.ndarray, fake: np.ndarray, rng: np.random.Generator) -> float:
    count = min(2000, len(real), len(fake)); x = real[rng.choice(len(real), count, False)]; y = fake[rng.choice(len(fake), count, False)]
    scaler = RobustScaler().fit(x); x = scaler.transform(x); y = scaler.transform(y)
    probe = np.vstack((x[:500], y[:500])); distances = ((probe[:, None] - probe[None]) ** 2).sum(2)
    nonzero = distances[distances > 0]; gamma = 1 / max(1e-8, float(np.median(nonzero)))
    return float(rbf_kernel(x, x, gamma=gamma).mean() + rbf_kernel(y, y, gamma=gamma).mean()
                 - 2 * rbf_kernel(x, y, gamma=gamma).mean())


def char_distances(real_urls: list[str], fake_urls: list[str], rng: np.random.Generator) -> dict:
    count = min(3000, len(real_urls), len(fake_urls))
    real = np.asarray(real_urls, dtype=object)[rng.choice(len(real_urls), count, False)].tolist()
    fake = np.asarray(fake_urls, dtype=object)[rng.choice(len(fake_urls), count, False)].tolist()
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**16,
                                  alternate_sign=False, norm="l2")
    real_matrix = vectorizer.transform(real); fake_matrix = vectorizer.transform(fake)
    cross = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1).fit(real_matrix).kneighbors(fake_matrix)[0][:, 0]
    self_distance = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1).fit(fake_matrix).kneighbors(fake_matrix)[0][:, 1]
    return {"sample_rows": count, "generated_to_nearest_real_mean": float(cross.mean()),
            "generated_to_nearest_real_p05": float(np.quantile(cross, .05)),
            "generated_to_nearest_generated_mean": float(self_distance.mean()),
            "generated_to_nearest_generated_p05": float(np.quantile(self_distance, .05))}


def challenge(urls: list[str], project: dict, vocabulary: dict[str, int], checkpoint_dir: Path,
              device: torch.device) -> dict:
    tokens = np.stack([encode_url(url, vocabulary, int(project["data"]["maximum_encoded_length"])) for url in urls])
    from .url_data import lexical_features
    features = np.stack([lexical_features(url) for url in urls]).astype(np.float32)
    decisions = []; per_run = []
    for name in MODELS:
        for seed in project["seeds"]:
            run = checkpoint_dir / name / f"seed_{seed}"
            if name in NEURAL:
                saved = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
                classifier = create_classifier(name, len(saved["vocabulary"]), saved["model_config"]).to(device)
                classifier.load_state_dict(saved["state_dict"]); scores = neural_scores(classifier, tokens, 512, device)
                threshold = float(saved["threshold"]); del classifier; gc.collect(); torch.cuda.empty_cache()
            else:
                saved = joblib.load(run / "model.joblib"); classifier = saved["model"]
                scores = classifier.predict_proba(features)[:, 1] if name == "random_forest" else classifier.decision_function(features)
                threshold = float(saved["threshold"])
            predicted = scores >= threshold; decisions.append(predicted)
            per_run.append({"model": name, "seed": seed, "phishing_rate": float(predicted.mean()),
                            "challenging_rate": float((~predicted).mean())})
    matrix = np.stack(decisions)
    return {"rows": len(urls), "per_run": per_run,
            "majority_challenging_rate": float((matrix.sum(0) < (len(matrix) / 2)).mean()),
            "challenging_to_at_least_one_rate": float((~matrix).any(0).mean()),
            "challenging_to_all_rate": float((~matrix).all(0).mean()),
            "mean_classifier_phishing_vote_fraction": float(matrix.mean())}


def main() -> None:
    args = parse_args(); project = load_json(args.config); set_seed(int(project["seed"])); rng = np.random.default_rng(int(project["seed"]))
    with (args.generated_root / args.model / "accepted.csv").open("r", encoding="utf-8", newline="") as handle:
        fake_all = [row["url"] for row in csv.DictReader(handle)]
    with (args.data_dir / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        real_all = [row["url"] for row in csv.DictReader(handle) if int(row["label"]) == 1]
    count = min(args.sample_size, len(real_all), len(fake_all))
    real_urls = np.asarray(real_all, dtype=object)[rng.choice(len(real_all), count, False)].tolist()
    fake_urls = np.asarray(fake_all, dtype=object)[rng.choice(len(fake_all), count, False)].tolist()
    real = np.stack([psl_aware_lexical_features(url) for url in real_urls]); fake = np.stack([psl_aware_lexical_features(url) for url in fake_urls])
    pooled_std = np.std(real, axis=0); smd = np.abs((fake.mean(0) - real.mean(0)) / np.maximum(pooled_std, 1e-6))
    scaler = RobustScaler().fit(real); real_scaled = scaler.transform(real); fake_scaled = scaler.transform(fake)
    clusters = MiniBatchKMeans(n_clusters=50, random_state=int(project["seed"]), batch_size=2048, n_init=10).fit(real_scaled)
    real_counts = np.bincount(clusters.labels_, minlength=50) + 1e-9; fake_counts = np.bincount(clusters.predict(fake_scaled), minlength=50) + 1e-9
    aucs = source_auc(real, fake, list(project["seeds"])); device = require_cuda(); vocabulary = load_json(args.data_dir / "vocabulary.json")
    report = {"project_version": 3, "generator": args.model, "sample_rows_per_source": count,
              "source_classifier_roc_auc": {"values": aucs, "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1))},
              "lexical_standardized_mean_difference": {"mean_absolute": float(smd.mean()), "median_absolute": float(np.median(smd)), "maximum_absolute": float(smd.max())},
              "rbf_mmd_squared": mmd(real, fake, rng),
              "real_cluster_coverage": float((fake_counts > 1e-8).mean()),
              "cluster_js_distance": float(jensenshannon(real_counts / real_counts.sum(), fake_counts / fake_counts.sum())),
              "character_ngram_distances": char_distances(real_urls, fake_urls, rng),
              "real_only_classifier_challenge": challenge(fake_urls, project, vocabulary, args.checkpoint_dir, device)}
    output = args.output_root / args.model; output.mkdir(parents=True, exist_ok=True); write_json(output / "quality.json", report)
    labels = ["Source AUC", "Cluster coverage", "Majority challenge", "At least one challenge"]
    values = [report["source_classifier_roc_auc"]["mean"], report["real_cluster_coverage"],
              report["real_only_classifier_challenge"]["majority_challenging_rate"],
              report["real_only_classifier_challenge"]["challenging_to_at_least_one_rate"]]
    fig, axis = plt.subplots(figsize=(9, 5)); bars = axis.bar(labels, values, color=("#E45756", "#54A24B", "#4C78A8", "#72B7B2"))
    axis.bar_label(bars, fmt="%.3f"); axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.25)
    axis.set_title(f"{args.model.upper()} quality and challenge audit", fontweight="bold"); fig.tight_layout()
    fig.savefig(output / "quality_summary.png", dpi=220); fig.savefig(output / "quality_summary.pdf"); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
