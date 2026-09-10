from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, set_seed, write_json
from .datasets import AugmentedEncodedURLDataset, EncodedURLDataset
from .metrics import classification_metrics, select_threshold
from .models import create_classifier


NEURAL = {"cnn", "lstm"}
TRADITIONAL = {"random_forest", "linear_svm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version 2 real-only baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--models", default="cnn,lstm,random_forest,linear_svm")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--real-indices", type=Path, help="Optional frozen subset of real training row indices.")
    parser.add_argument(
        "--augmentation-manifest", type=Path,
        help="Optional generated dose JSON. Validation and test remain real-only.",
    )
    return parser.parse_args()


@torch.inference_mode()
def neural_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval(); probabilities = []; labels = []
    start = time.perf_counter()
    for tokens, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(tokens)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy())
    torch.cuda.synchronize()
    return np.concatenate(probabilities), np.concatenate(labels).astype(np.uint8), time.perf_counter() - start


def loaders(
    data_dir: Path, batch_size: int, workers: int, augmentation: dict | None = None,
    real_indices: np.ndarray | None = None,
) -> dict[str, DataLoader]:
    result = {}
    for split in ("train", "validation", "test"):
        dataset = EncodedURLDataset(data_dir, split, indices=real_indices if split == "train" else None)
        if split == "train" and augmentation is not None:
            dataset = AugmentedEncodedURLDataset(
                data_dir, Path(augmentation["generated_tokens"]), augmentation["generated_rows"],
                real_indices=real_indices,
                synthetic_weight=float(augmentation.get("synthetic_weight", 1.0)),
            )
        result[split] = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=split == "train", num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
        )
    return result


def train_neural(
    name: str, seed: int, project: dict, data_dir: Path, output: Path, workers: int,
    augmentation: dict | None, real_indices: np.ndarray | None,
) -> dict:
    device = require_cuda(); set_seed(seed)
    vocabulary = load_json(data_dir / "vocabulary.json")
    model_config = project["baseline_models"][name]
    training = project["baseline_training"]
    model = create_classifier(name, len(vocabulary), model_config).to(device)
    data_loaders = loaders(data_dir, int(model_config["batch_size"]), workers, augmentation, real_indices)
    all_train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    train_labels = np.asarray(all_train_labels[real_indices]) if real_indices is not None else all_train_labels
    generated_count = int(augmentation["generated_rows"]) if augmentation else 0
    # The real subset alone defines class balance. Synthetic rows are supplemental
    # and must not reduce the contribution of real phishing examples.
    positive_weight = float((train_labels == 0).sum() / max(1, (train_labels == 1).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device), reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []; best_pr_auc = -np.inf; best_state = None; best_epoch = 0; stale = 0
    total_training_time = 0.0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); running_loss = 0.0; seen = 0
        started = time.perf_counter()
        for batch in data_loaders["train"]:
            if len(batch) == 3:
                tokens, targets, sample_weights = batch
            else:
                tokens, targets = batch; sample_weights = torch.ones_like(targets)
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                losses = criterion(model(tokens), targets)
                loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            scaler.step(optimizer); scaler.update()
            running_loss += float(loss.item()) * len(targets); seen += len(targets)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        total_training_time += elapsed
        validation_scores, validation_labels, _ = neural_scores(
            model, data_loaders["validation"], device
        )
        validation_pr_auc = float(average_precision_score(validation_labels, validation_scores))
        record = {
            "epoch": epoch, "train_loss": running_loss / max(1, seen),
            "validation_pr_auc": validation_pr_auc, "training_seconds": elapsed,
        }
        history.append(record); print(json.dumps({"model":name,"seed":seed,**record}), flush=True)
        if validation_pr_auc > best_pr_auc:
            best_pr_auc = validation_pr_auc; best_state = deepcopy(model.state_dict())
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Neural training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_scores, validation_labels, validation_time = neural_scores(
        model, data_loaders["validation"], device
    )
    threshold = select_threshold(validation_labels, validation_scores, maximum_fpr=0.01)
    test_scores, test_labels, test_time = neural_scores(model, data_loaders["test"], device)
    metrics = classification_metrics(test_labels, test_scores, threshold["threshold"])
    run_dir = output / name / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "project_version": 2, "condition": augmentation["condition"] if augmentation else "real_only", "model": name, "seed": seed,
        "model_config": model_config, "vocabulary": vocabulary,
        "state_dict": best_state, "threshold": threshold["threshold"], "best_epoch": best_epoch,
        "master_sha256": project["data"]["master_sha256"],
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
        predictions=(test_scores >= threshold["threshold"]).astype(np.uint8),
    )
    payload = {
        **metrics, "project_version": 2, "condition": augmentation["condition"] if augmentation else "real_only", "model": name,
        "seed": seed, "threshold": threshold["threshold"], "threshold_selection": threshold,
        "best_epoch": best_epoch, "validation_pr_auc": best_pr_auc,
        "training_time_seconds": total_training_time,
        "validation_inference_seconds": validation_time,
        "test_inference_seconds": test_time,
        "inference_urls_per_second": len(test_labels) / test_time,
        "model_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
        "device": torch.cuda.get_device_name(device), "mixed_precision": True,
        "master_sha256": project["data"]["master_sha256"],
        "generator": augmentation["generator"] if augmentation else None,
        "real_training_rows": len(train_labels), "generated_training_rows": generated_count,
        "real_indices": str(augmentation.get("real_indices")) if augmentation and augmentation.get("real_indices") else None,
        "synthetic_weight": float(augmentation.get("synthetic_weight", 1.0)) if augmentation else None,
        "class_balance_source": "real training rows only",
    }
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metrics.json", payload)
    return payload


def train_traditional(
    name: str, seed: int, project: dict, data_dir: Path, feature_dir: Path, output: Path,
    augmentation: dict | None, real_indices: np.ndarray | None,
) -> dict:
    set_seed(seed); config = project["baseline_models"][name]
    features = {
        split: np.load(feature_dir / f"{split}_features.npy", mmap_mode="r")
        for split in ("train", "validation", "test")
    }
    labels = {
        split: np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        for split in ("train", "validation", "test")
    }
    generated_count = int(augmentation["generated_rows"]) if augmentation else 0
    training_features = np.asarray(features["train"][real_indices]) if real_indices is not None else features["train"]
    training_labels = np.asarray(labels["train"][real_indices]) if real_indices is not None else labels["train"]
    real_count = len(training_labels); sample_weights = np.ones(real_count, dtype=np.float32)
    class_weights = {
        0: real_count / (2 * max(1, int((training_labels == 0).sum()))),
        1: real_count / (2 * max(1, int((training_labels == 1).sum()))),
    }
    if augmentation is not None:
        generated_features = np.load(Path(augmentation["generated_features"]), mmap_mode="r")
        training_features = np.concatenate((np.asarray(training_features), generated_features[:generated_count]))
        training_labels = np.concatenate((np.asarray(training_labels), np.ones(generated_count, dtype=np.uint8)))
        sample_weights = np.concatenate((sample_weights, np.full(generated_count, float(augmentation.get("synthetic_weight", 1.0)), dtype=np.float32)))
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"], max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"],
            class_weight=class_weights, random_state=seed, n_jobs=-1,
        )
        score = lambda values: model.predict_proba(values)[:, 1]
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=config["C"], class_weight=class_weights,
                max_iter=config["max_iter"], random_state=seed, dual="auto",
            )),
        ])
        score = model.decision_function
    started = time.perf_counter()
    if name == "random_forest": model.fit(training_features, training_labels, sample_weight=sample_weights)
    else: model.fit(training_features, training_labels, classifier__sample_weight=sample_weights)
    training_time = time.perf_counter() - started
    validation_scores = score(features["validation"])
    threshold = select_threshold(labels["validation"], validation_scores, maximum_fpr=0.01)
    started = time.perf_counter(); test_scores = score(features["test"])
    test_time = time.perf_counter() - started
    metrics = classification_metrics(labels["test"], test_scores, threshold["threshold"])
    run_dir = output / name / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "threshold": threshold["threshold"], "project_version": 2,
        "master_sha256": project["data"]["master_sha256"],
        "condition": augmentation["condition"] if augmentation else "real_only",
        "generator": augmentation["generator"] if augmentation else None,
    }, run_dir / "model.joblib", compress=3)
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=np.asarray(labels["test"]),
        scores=test_scores, predictions=(test_scores >= threshold["threshold"]).astype(np.uint8),
    )
    payload = {
        **metrics, "project_version": 2, "condition": augmentation["condition"] if augmentation else "real_only", "model": name,
        "seed": seed, "threshold": threshold["threshold"], "threshold_selection": threshold,
        "training_time_seconds": training_time, "test_inference_seconds": test_time,
        "inference_urls_per_second": len(labels["test"]) / test_time,
        "model_size_bytes": (run_dir / "model.joblib").stat().st_size,
        "device": "CPU", "mixed_precision": False,
        "master_sha256": project["data"]["master_sha256"],
        "generator": augmentation["generator"] if augmentation else None,
        "real_training_rows": real_count, "generated_training_rows": generated_count,
        "real_indices": str(augmentation.get("real_indices")) if augmentation and augmentation.get("real_indices") else None,
        "synthetic_weight": float(augmentation.get("synthetic_weight", 1.0)) if augmentation else None,
        "class_balance_source": "real training rows only",
    }
    write_json(run_dir / "metrics.json", payload)
    return payload


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    augmentation = load_json(args.augmentation_manifest) if args.augmentation_manifest else None
    real_indices_path = args.real_indices or (Path(augmentation["real_indices"]) if augmentation and augmentation.get("real_indices") else None)
    real_indices = np.load(real_indices_path) if real_indices_path else None
    output_dir = args.output_dir
    if augmentation is not None and args.output_dir == Path("artifacts/baselines"):
        output_dir = Path("artifacts/augmentation") / augmentation["generator"] / augmentation["condition"]
    if augmentation is not None and augmentation["master_sha256"] != project["data"]["master_sha256"]:
        raise RuntimeError("Augmentation manifest belongs to a different master dataset")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    unknown = set(models) - NEURAL - TRADITIONAL
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    for seed in seeds:
        for name in models:
            if name in NEURAL:
                result = train_neural(name, seed, project, args.data_dir, output_dir, args.num_workers, augmentation, real_indices)
            else:
                result = train_traditional(name, seed, project, args.data_dir, args.feature_dir, output_dir, augmentation, real_indices)
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
