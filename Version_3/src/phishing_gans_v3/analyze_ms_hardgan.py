from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from .analyze_gan_quality import char_distances, challenge, mmd, source_auc
from .common import load_json, require_cuda, set_seed, write_json
from .url_data import psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MS-HardGAN realism, diversity, and classifier challenge.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated/ms_hardgan_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments/ms_hardgan_v1"))
    parser.add_argument("--classifier-data-dir", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/multisource_baselines_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/ms_hardgan_v1/quality"))
    parser.add_argument("--sample-size", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); project = load_json(args.config); seed = int(project["seed"])
    set_seed(seed); rng = np.random.default_rng(seed)
    with (args.generated_dir / "accepted.csv").open("r", encoding="utf-8", newline="") as handle:
        fake_all = [row["url"] for row in csv.DictReader(handle)]
    real_all = (args.data_dir / "train_urls.txt").read_text(encoding="utf-8").splitlines()
    count = min(args.sample_size, len(real_all), len(fake_all))
    real_urls = np.asarray(real_all, dtype=object)[rng.choice(len(real_all), count, False)].tolist()
    fake_urls = np.asarray(fake_all, dtype=object)[rng.choice(len(fake_all), count, False)].tolist()
    real = np.stack([psl_aware_lexical_features(url) for url in real_urls])
    fake = np.stack([psl_aware_lexical_features(url) for url in fake_urls])
    standardized = np.abs((fake.mean(0) - real.mean(0)) / np.maximum(real.std(0), 1e-6))
    scaler = RobustScaler().fit(real)
    real_scaled, fake_scaled = scaler.transform(real), scaler.transform(fake)
    clusterer = MiniBatchKMeans(n_clusters=50, random_state=seed, batch_size=2048, n_init=10).fit(real_scaled)
    real_counts = np.bincount(clusterer.labels_, minlength=50) + 1e-9
    fake_counts = np.bincount(clusterer.predict(fake_scaled), minlength=50) + 1e-9
    device = require_cuda(); vocabulary = load_json(args.classifier_data_dir / "vocabulary.json")
    audit = load_json(args.generated_dir / "audit.json")
    report = {
        "project_version": 3, "generator": audit["generator"], "sample_rows_per_distribution": count,
        "generation_audit": {
            "accepted_rows": audit["accepted_rows"], "accepted_fraction": audit["accepted_fraction"],
            "category_agreement": audit["category_agreement"],
            "held_out_challenge_rows": audit["held_out_challenge_rows"],
            "augmentation_pool_rows": audit["augmentation_pool_rows"],
        },
        "source_classifier_roc_auc": {},
        "lexical_standardized_mean_difference": {
            "mean_absolute": float(standardized.mean()), "median_absolute": float(np.median(standardized)),
            "maximum_absolute": float(standardized.max()),
        },
        "rbf_mmd_squared": mmd(real, fake, rng),
        "real_cluster_coverage": float((fake_counts > 1e-8).mean()),
        "cluster_js_distance": float(jensenshannon(real_counts / real_counts.sum(), fake_counts / fake_counts.sum())),
        "character_ngram_distances": char_distances(real_urls, fake_urls, rng),
        "frozen_ms_classifier_challenge": challenge(
            fake_urls, project, vocabulary, args.checkpoint_dir, device
        ),
        "interpretation": "Novel and difficult samples are zero-day-like stress tests, not predictions of future attacks.",
    }
    aucs = source_auc(real, fake, list(project["seeds"]))
    report["source_classifier_roc_auc"] = {
        "values": aucs, "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "quality.json", report)
    challenge_report = report["frozen_ms_classifier_challenge"]
    labels = ("Validity", "Uniqueness", "Pattern\nagreement", "Source\nagreement",
              "Cluster\ncoverage", "Majority\nchallenge")
    values = (
        audit["accepted_quality"]["validity"], audit["accepted_quality"]["uniqueness"],
        audit["category_agreement"]["requested_pattern_agreement"],
        audit["category_agreement"]["requested_source_agreement"],
        report["real_cluster_coverage"], challenge_report["majority_challenging_rate"],
    )
    fig, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(labels, values, color=("#54A24B", "#72B7B2", "#4C78A8", "#5F9ED1", "#F2CF5B", "#E45756"))
    axis.bar_label(bars, fmt="%.3f", padding=3); axis.set_ylim(0, 1.08); axis.grid(axis="y", alpha=.25)
    axis.set_ylabel("Fraction"); axis.set_title("MS-HardGAN defensive quality audit", fontweight="bold")
    fig.tight_layout(); fig.savefig(args.output_dir / "quality_summary.png", dpi=220)
    fig.savefig(args.output_dir / "quality_summary.pdf"); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
