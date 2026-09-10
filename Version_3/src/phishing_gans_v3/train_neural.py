from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, set_seed, write_json
from .datasets import EncodedURLDataset
from .metrics import classification_metrics, select_threshold
from .models import create_classifier


MODELS = {"cnn", "lstm", "rnn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version 3 CNN, LSTM, and RNN baselines on CUDA.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--condition", default="real_only")
    parser.add_argument("--models", default="cnn,lstm,rnn")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_loaders(data_dir: Path, batch_size: int, workers: int) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            EncodedURLDataset(data_dir, split), batch_size=batch_size,
            shuffle=split == "train", num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
        )
        for split in ("train", "validation", "test")
    }


@torch.inference_mode()
def score_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    started = time.perf_counter()
    for tokens, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(tokens)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy())
    torch.cuda.synchronize()
    return np.concatenate(scores), np.concatenate(labels).astype(np.uint8), time.perf_counter() - started


def train_one(name: str, seed: int, project: dict, data_dir: Path, output_dir: Path,
              workers: int, overwrite: bool, condition: str) -> dict:
    run_dir = output_dir / name / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        result = load_json(metrics_path)
        print(json.dumps({"stage": "skip_completed", "model": name, "seed": seed}), flush=True)
        return result

    device = require_cuda()
    set_seed(seed)
    vocabulary = load_json(data_dir / "vocabulary.json")
    model_config = project["baseline_models"][name]
    training_config = project["neural_training"]
    loaders = make_loaders(data_dir, int(model_config["batch_size"]), workers)
    model = create_classifier(name, len(vocabulary), model_config).to(device)
    train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    positive_weight = float((train_labels == 0).sum() / max(1, int((train_labels == 1).sum())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    total_training_seconds = 0.0

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        started = time.perf_counter()
        for tokens, targets in loaders["train"]:
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(tokens), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(targets)
            seen += len(targets)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        total_training_seconds += training_seconds
        validation_scores, validation_labels, validation_seconds = score_model(model, loaders["validation"], device)
        validation_pr_auc = float(average_precision_score(validation_labels, validation_scores))
        record = {
            "epoch": epoch, "train_loss": running_loss / max(1, seen),
            "validation_pr_auc": validation_pr_auc, "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
        }
        history.append(record)
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
        if validation_pr_auc > best_pr_auc:
            best_pr_auc = validation_pr_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(training_config["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_scores, validation_labels, validation_seconds = score_model(model, loaders["validation"], device)
    threshold = select_threshold(validation_labels, validation_scores, maximum_fpr=0.01)
    test_scores, test_labels, test_seconds = score_model(model, loaders["test"], device)
    metrics = classification_metrics(test_labels, test_scores, threshold["threshold"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "project_version": 3, "condition": condition, "model": name, "seed": seed,
        "model_config": model_config, "vocabulary": vocabulary, "state_dict": best_state,
        "threshold": threshold["threshold"], "best_epoch": best_epoch,
        "master_sha256": project["data"]["master_sha256"], "scheme_removed": True,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
        predictions=(test_scores >= threshold["threshold"]).astype(np.uint8),
    )
    payload = {
        **metrics, "project_version": 3, "condition": condition, "model": name, "seed": seed,
        "threshold": threshold["threshold"], "threshold_selection": threshold,
        "best_epoch": best_epoch, "validation_pr_auc": best_pr_auc,
        "training_time_seconds": total_training_seconds,
        "validation_inference_seconds": validation_seconds,
        "test_inference_seconds": test_seconds,
        "inference_urls_per_second": len(test_labels) / test_seconds,
        "model_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
        "device": torch.cuda.get_device_name(device), "mixed_precision": True,
        "real_training_rows": len(train_labels), "master_sha256": project["data"]["master_sha256"],
        "scheme_removed": True,
    }
    write_json(run_dir / "history.json", history)
    write_json(metrics_path, payload)
    del model, loaders, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    unknown = set(models) - MODELS
    if unknown:
        raise ValueError(f"Unknown neural models: {sorted(unknown)}")
    require_cuda()
    for seed in seeds:
        for name in models:
            result = train_one(name, seed, project, args.data_dir, args.output_dir, args.num_workers, args.overwrite, args.condition)
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
