from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .evaluate_generated_challenge import (
    MODELS,
    NEURAL,
    neural_scores,
    real_feature_statistics,
    tld_like,
)
from .models import create_classifier
from .url_data import LEXICAL_FEATURE_NAMES


SOURCE_GENERATOR = "seqgan"
STRUCTURAL_FEATURES = (
    "url_length",
    "hostname_length",
    "estimated_subdomains",
    "double_slash_count",
    "double_dot_count",
    "slash_count",
    "percent_count",
    "at_count",
)
CONDITIONS = {
    "realism_filtered_10_percent": "random",
    "realism_challenging_10_percent": "challenging",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze quality-controlled SeqGAN 10% augmentation conditions."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/augmented/seqgan"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/augmented/seqgan_quality"))
    parser.add_argument("--challenge-dir", type=Path, default=Path("data/challenges/seqgan_quality_holdout"))
    parser.add_argument("--reserve-size", type=int, default=100000)
    parser.add_argument("--challenge-count", type=int, default=20000)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--cluster-max-share", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def read_pool_manifest(path: Path, expected: int) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    hosts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            if int(row["pool_index"]) != expected_index:
                raise RuntimeError("Source generated-pool manifest is not index-aligned")
            urls.append(row["url"])
            hosts.append(row["hostname"])
    if len(urls) != expected:
        raise RuntimeError(f"Expected {expected:,} source rows, found {len(urls):,}")
    if len(set(urls)) != len(urls):
        raise RuntimeError("Strict accepted source pool unexpectedly contains duplicates")
    return urls, hosts


def real_upper_quantiles(
    features: np.ndarray, malicious_indices: np.ndarray, columns: list[int]
) -> np.ndarray:
    selected = np.asarray(features[np.asarray(malicious_indices)][:, columns], dtype=np.float32)
    return np.quantile(selected, 0.995, axis=0)


def empirical_percentiles(scores: np.ndarray, calibration_mask: np.ndarray) -> tuple[np.ndarray, float]:
    calibration = np.sort(np.asarray(scores[calibration_mask], dtype=np.float32))
    if not len(calibration):
        raise RuntimeError("No candidates available for score calibration")
    percentiles = np.searchsorted(calibration, scores, side="right").astype(np.float32)
    percentiles /= float(len(calibration))
    return percentiles, float(len(calibration))


def allocate_diverse(
    candidates: np.ndarray,
    priority: np.ndarray,
    clusters: np.ndarray,
    target: int,
    maximum_share: float,
) -> tuple[np.ndarray, dict]:
    if len(candidates) < target:
        raise RuntimeError(f"Need {target:,} eligible candidates but only have {len(candidates):,}")
    cap = max(1, int(math.floor(target * maximum_share)))
    ordered = candidates[np.argsort(-priority[candidates], kind="stable")]
    chosen: list[int] = []
    counts: dict[int, int] = {}
    rejected_for_cap: list[int] = []
    for index in ordered:
        cluster = int(clusters[int(index)])
        if counts.get(cluster, 0) < cap:
            chosen.append(int(index)); counts[cluster] = counts.get(cluster, 0) + 1
            if len(chosen) == target:
                break
        else:
            rejected_for_cap.append(int(index))
    fallback = 0
    if len(chosen) < target:
        needed = target - len(chosen)
        chosen.extend(rejected_for_cap[:needed]); fallback = needed
        for index in rejected_for_cap[:needed]:
            cluster = int(clusters[index]); counts[cluster] = counts.get(cluster, 0) + 1
    selected = np.asarray(chosen, dtype=np.int64)
    return selected, {
        "cluster_cap_rows": cap,
        "fallback_rows_after_cluster_cap": fallback,
        "represented_clusters": len(counts),
        "largest_cluster_rows": max(counts.values()),
        "largest_cluster_share": max(counts.values()) / target,
        "cluster_counts": {str(key): value for key, value in sorted(counts.items())},
    }


def materialize(
    output_dir: Path,
    condition: str,
    selected: np.ndarray,
    source_tokens: np.ndarray,
    source_features: np.ndarray,
    urls: list[str],
    hosts: list[str],
    selection_score: np.ndarray,
    evasion_votes: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / f"{condition}_tokens.npy"
    feature_path = output_dir / f"{condition}_features.npy"
    index_path = output_dir / f"{condition}_source_indices.npy"
    manifest_path = output_dir / f"{condition}_manifest.csv"
    token_output = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=source_tokens.dtype,
        shape=(len(selected), source_tokens.shape[1]),
    )
    feature_output = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=source_features.dtype,
        shape=(len(selected), source_features.shape[1]),
    )
    for start in range(0, len(selected), 50000):
        stop = min(len(selected), start + 50000)
        rows = selected[start:stop]
        token_output[start:stop] = source_tokens[rows]
        feature_output[start:stop] = source_features[rows]
    token_output.flush(); feature_output.flush(); np.save(index_path, selected)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "selected_index", "source_pool_index", "url", "label", "generator", "hostname",
            "selection_score", "evasion_votes_of_12", "lexical_cluster",
        ))
        for selected_index, source_index in enumerate(selected):
            i = int(source_index)
            writer.writerow((
                selected_index, i, urls[i], 1, SOURCE_GENERATOR, hosts[i],
                float(selection_score[i]), int(evasion_votes[i]), int(clusters[i]),
            ))
    return {
        "tokens": token_path,
        "features": feature_path,
        "indices": index_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    set_seed(int(project["seed"]))
    rng = np.random.default_rng(int(project["seed"]))
    source_index = load_json(args.source_dir / "index.json")
    existing_manifest = load_json(args.source_dir / "generated_10_percent.json")
    target = int(existing_manifest["generated_rows"])
    source_tokens = np.load(args.source_dir / "generated_pool_tokens.npy", mmap_mode="r")
    source_features = np.load(args.source_dir / "generated_pool_features.npy", mmap_mode="r")
    row_count = len(source_tokens)
    if len(source_features) != row_count:
        raise RuntimeError("Source token and feature pools have different lengths")
    if args.reserve_size <= args.challenge_count or args.reserve_size >= row_count - target:
        raise ValueError("Reserve must exceed challenge count and leave enough training candidates")
    urls, hosts = read_pool_manifest(args.source_dir / "generated_pool_manifest.csv", row_count)

    # Reserve is frozen before any augmentation ranking. It is outside the
    # existing random-10% prefix, so no four-condition model trains on it.
    reserve_mask = np.zeros(row_count, dtype=bool)
    reserve_mask[row_count - args.reserve_size:] = True
    if reserve_mask[:target].any():
        raise RuntimeError("Synthetic challenge reserve overlaps existing random 10% prefix")

    real_mean, real_std, malicious_indices = real_feature_statistics(args.feature_dir, args.data_dir)
    generated = np.asarray(source_features, dtype=np.float32)
    lexical_z = np.abs((generated - real_mean) / np.maximum(real_std, 1e-6))
    lexical_inlier = lexical_z.max(axis=1) <= 5.0
    tld_mask = np.fromiter((tld_like(host) for host in hosts), dtype=bool, count=row_count)
    structural_columns = [LEXICAL_FEATURE_NAMES.index(name) for name in STRUCTURAL_FEATURES]
    real_features = np.load(args.feature_dir / "train_features.npy", mmap_mode="r")
    structural_upper = real_upper_quantiles(real_features, malicious_indices, structural_columns)
    structural_inlier = np.all(generated[:, structural_columns] <= structural_upper, axis=1)
    realism_mask = lexical_inlier & tld_mask & structural_inlier
    training_eligible = realism_mask & ~reserve_mask
    challenge_eligible = realism_mask & reserve_mask
    if training_eligible.sum() < target or challenge_eligible.sum() < args.challenge_count:
        raise RuntimeError("Frozen realism rule leaves too few candidates")

    device = require_cuda()
    seeds = tuple(int(value) for value in project["seeds"])
    percentile_sum = np.zeros(row_count, dtype=np.float32)
    boundary_sum = np.zeros(row_count, dtype=np.float32)
    evasion_votes = np.zeros(row_count, dtype=np.uint8)
    checkpoint_records = []
    for model_name in MODELS:
        for seed in seeds:
            if model_name in NEURAL:
                checkpoint_path = Path(f"artifacts/baselines/{model_name}/seed_{seed}/checkpoint.pt")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model = create_classifier(
                    model_name, len(checkpoint["vocabulary"]), checkpoint["model_config"]
                ).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                scores = neural_scores(model, source_tokens, args.batch_size, device)
            else:
                checkpoint_path = Path(f"artifacts/baselines/{model_name}/seed_{seed}/model.joblib")
                checkpoint = joblib.load(checkpoint_path)
                model = checkpoint["model"]
                if model_name == "random_forest":
                    scores = model.predict_proba(source_features)[:, 1].astype(np.float32)
                else:
                    scores = model.decision_function(source_features).astype(np.float32)
            if checkpoint["master_sha256"] != project["data"]["master_sha256"]:
                raise RuntimeError(f"Master hash mismatch in {checkpoint_path}")
            threshold = float(checkpoint["threshold"])
            percentile, calibration_count = empirical_percentiles(scores, training_eligible)
            threshold_percentile = float((scores[training_eligible] <= threshold).mean())
            percentile_sum += percentile
            boundary_sum += np.exp(-np.abs(percentile - threshold_percentile) / 0.10).astype(np.float32)
            evasion_votes += (scores < threshold).astype(np.uint8)
            checkpoint_records.append({
                "model": model_name,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "threshold": threshold,
                "threshold_percentile_within_training_eligible": threshold_percentile,
                "calibration_rows": int(calibration_count),
            })
            print(json.dumps({
                "scored": f"{model_name}/seed_{seed}",
                "training_evasion_rate": float((scores[training_eligible] < threshold).mean()),
                "challenge_reserve_evasion_rate": float((scores[challenge_eligible] < threshold).mean()),
            }), flush=True)
            del scores, percentile, model, checkpoint
            gc.collect(); torch.cuda.empty_cache()

    checkpoint_count = len(MODELS) * len(seeds)
    mean_percentile = percentile_sum / checkpoint_count
    mean_boundary = boundary_sum / checkpoint_count
    consensus_evasion = evasion_votes.astype(np.float32) / checkpoint_count
    selection_score = (
        0.50 * (1.0 - mean_percentile)
        + 0.30 * consensus_evasion
        + 0.20 * mean_boundary
    ).astype(np.float32)

    standardized = ((generated - real_mean) / np.maximum(real_std, 1e-6)).astype(np.float32)
    fit_candidates = np.flatnonzero(training_eligible)
    fit_sample = rng.choice(fit_candidates, min(200000, len(fit_candidates)), replace=False)
    clusterer = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=int(project["seed"]),
        batch_size=8192,
        n_init=3,
    ).fit(standardized[fit_sample])
    cluster_ids = np.empty(row_count, dtype=np.int16)
    for start in range(0, row_count, 100000):
        stop = min(row_count, start + 100000)
        cluster_ids[start:stop] = clusterer.predict(standardized[start:stop]).astype(np.int16)

    random_priority = rng.random(row_count).astype(np.float32)
    filtered_indices, filtered_diversity = allocate_diverse(
        np.flatnonzero(training_eligible), random_priority, cluster_ids,
        target, args.cluster_max_share,
    )
    challenging_indices, challenging_diversity = allocate_diverse(
        np.flatnonzero(training_eligible), selection_score, cluster_ids,
        target, args.cluster_max_share,
    )
    challenge_indices, challenge_diversity = allocate_diverse(
        np.flatnonzero(challenge_eligible), selection_score, cluster_ids,
        args.challenge_count, args.cluster_max_share,
    )
    if set(challenge_indices).intersection(filtered_indices) or set(challenge_indices).intersection(challenging_indices):
        raise RuntimeError("Held-out synthetic challenge overlaps an augmentation condition")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rule = {
        "project_version": 2,
        "status": "frozen_before_augmented_model_test_evaluation",
        "source_generator": SOURCE_GENERATOR,
        "source_pool_index_sha256": sha256_file(args.source_dir / "index.json"),
        "master_sha256": project["data"]["master_sha256"],
        "target_generated_rows": target,
        "generator_composition": {SOURCE_GENERATOR: 1.0},
        "realism": {
            "strict_accepted_pool": True,
            "tld_like": "at least two hostname labels; alphabetic final label length 2-24",
            "lexical_inlier": "maximum absolute z-score <= 5 using real malicious training statistics",
            "structural_upper_quantile": 0.995,
            "structural_features": list(STRUCTURAL_FEATURES),
            "structural_upper_values": {
                name: float(value) for name, value in zip(STRUCTURAL_FEATURES, structural_upper)
            },
        },
        "challenging_score": {
            "raw_score_combination": False,
            "per_checkpoint_calibration": "empirical percentile within non-reserved realism-eligible candidates",
            "formula": "0.50*(1-mean_percentile)+0.30*evasion_vote_fraction+0.20*mean_boundary_proximity",
            "boundary_bandwidth_percentile": 0.10,
            "checkpoints": checkpoint_records,
        },
        "diversity": {
            "method": "64-cluster MiniBatchKMeans in real-standardized 28-feature lexical space",
            "fit_rows": len(fit_sample),
            "maximum_cluster_share": args.cluster_max_share,
        },
        "challenge_reserve": {
            "source_slice": [row_count - args.reserve_size, row_count],
            "reserve_rows": args.reserve_size,
            "selected_rows": args.challenge_count,
            "outside_existing_random_10_percent_prefix": True,
            "excluded_from_both_new_augmentation_conditions": True,
        },
        "validation_role": "inherited only through the already-frozen real-only checkpoints and thresholds",
        "selection_does_not_use": ["real test data", "augmented model results"],
    }
    rule_path = args.output_dir / "selection_rule.json"
    write_json(rule_path, rule)
    rule_sha256 = sha256_file(rule_path)

    condition_indices = {
        "realism_filtered_10_percent": (filtered_indices, filtered_diversity),
        "realism_challenging_10_percent": (challenging_indices, challenging_diversity),
    }
    condition_reports = {}
    real_count = int(existing_manifest["real_train_rows"])
    for condition, (indices, diversity) in condition_indices.items():
        paths = materialize(
            args.output_dir, condition, indices, source_tokens, source_features,
            urls, hosts, selection_score, evasion_votes, cluster_ids,
        )
        manifest = {
            "project_version": 2,
            "condition": condition,
            "generator": SOURCE_GENERATOR,
            "selection_kind": CONDITIONS[condition],
            "selection_rule": str(rule_path),
            "selection_rule_sha256": rule_sha256,
            "real_train_rows": real_count,
            "generated_rows": target,
            "final_training_rows": real_count + target,
            "generated_fraction": target / (real_count + target),
            "generated_pool_slice": [0, target],
            "real_tokens": str(args.data_dir / "train_tokens.npy"),
            "real_labels": str(args.data_dir / "train_labels.npy"),
            "generated_tokens": str(paths["tokens"]),
            "generated_features": str(paths["features"]),
            "generated_label": 1,
            "master_sha256": project["data"]["master_sha256"],
            "source_accepted_csv_sha256": source_index["conditions"]["generated_10_percent"]["accepted_csv_sha256"],
            "source_indices_sha256": sha256_file(paths["indices"]),
            "generated_tokens_sha256": sha256_file(paths["tokens"]),
            "generated_features_sha256": sha256_file(paths["features"]),
            "generated_manifest_sha256": sha256_file(paths["manifest"]),
            "generator_composition": {SOURCE_GENERATOR: target},
            "diversity": diversity,
            "mean_selection_score": float(selection_score[indices].mean()),
            "mean_evasion_votes_of_12": float(evasion_votes[indices].mean()),
            "majority_evasion_rate": float((evasion_votes[indices] >= 7).mean()),
        }
        manifest_path = args.output_dir / f"{condition}.json"
        write_json(manifest_path, manifest)
        condition_reports[condition] = {"manifest": str(manifest_path), **manifest}

    challenge_paths = materialize(
        args.challenge_dir, "challenge", challenge_indices, source_tokens,
        source_features, urls, hosts, selection_score, evasion_votes, cluster_ids,
    )
    challenge_manifest = {
        "project_version": 2,
        "role": "evaluation_only_synthetic_phishing_challenge",
        "frozen_before_augmented_model_training": True,
        "source_generator": SOURCE_GENERATOR,
        "rows": args.challenge_count,
        "label": 1,
        "tokens": str(challenge_paths["tokens"]),
        "features": str(challenge_paths["features"]),
        "manifest": str(challenge_paths["manifest"]),
        "source_indices": str(challenge_paths["indices"]),
        "selection_rule_sha256": rule_sha256,
        "master_sha256": project["data"]["master_sha256"],
        "tokens_sha256": sha256_file(challenge_paths["tokens"]),
        "features_sha256": sha256_file(challenge_paths["features"]),
        "manifest_sha256": sha256_file(challenge_paths["manifest"]),
        "source_indices_sha256": sha256_file(challenge_paths["indices"]),
        "diversity": challenge_diversity,
        "mean_selection_score": float(selection_score[challenge_indices].mean()),
        "mean_evasion_votes_of_12": float(evasion_votes[challenge_indices].mean()),
        "majority_evasion_rate": float((evasion_votes[challenge_indices] >= 7).mean()),
        "metric_warning": "intended-phishing-only set supports recall/FNR, not accuracy, precision, FPR, or ROC-AUC",
    }
    args.challenge_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.challenge_dir / "challenge.json", challenge_manifest)
    report = {
        "project_version": 2,
        "selection_rule": str(rule_path),
        "selection_rule_sha256": rule_sha256,
        "source_rows": row_count,
        "target_rows_per_condition": target,
        "realism_rates": {
            "tld_like": float(tld_mask.mean()),
            "lexical_inlier": float(lexical_inlier.mean()),
            "structural_inlier": float(structural_inlier.mean()),
            "combined": float(realism_mask.mean()),
        },
        "eligible_training_rows": int(training_eligible.sum()),
        "eligible_reserved_challenge_rows": int(challenge_eligible.sum()),
        "conditions": condition_reports,
        "challenge": challenge_manifest,
        "condition_overlap_rows": int(len(set(filtered_indices).intersection(challenging_indices))),
    }
    write_json(args.output_dir / "experiment_manifest.json", report)
    print(json.dumps({
        "selection_rule_sha256": rule_sha256,
        "realism_combined_rate": report["realism_rates"]["combined"],
        "eligible_training_rows": report["eligible_training_rows"],
        "conditions": {
            key: {
                "rows": value[0].shape[0],
                "mean_selection_score": condition_reports[key]["mean_selection_score"],
                "majority_evasion_rate": condition_reports[key]["majority_evasion_rate"],
            } for key, value in condition_indices.items()
        },
        "challenge_rows": args.challenge_count,
    }, indent=2))


if __name__ == "__main__":
    main()
