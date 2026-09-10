from __future__ import annotations

import argparse
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import load_json, require_cuda, set_seed, write_json
from .metrics import classification_metrics, select_threshold
from .models import HybridCharacterCNN
from .url_data import LEXICAL_FEATURE_NAMES


class HybridURLDataset(Dataset):
    def __init__(
        self, data_dir: Path, feature_dir: Path, split: str,
        feature_mean: np.ndarray, feature_std: np.ndarray,
    ) -> None:
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.features = np.load(feature_dir / f"{split}_features.npy", mmap_mode="r")
        if not (len(self.tokens) == len(self.labels) == len(self.features)):
            raise RuntimeError(f"Misaligned hybrid inputs for {split}")
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.from_numpy(np.asarray(self.tokens[index], dtype=np.int64))
        lexical = (np.asarray(self.features[index], dtype=np.float32) - self.feature_mean) / self.feature_std
        label = torch.tensor(np.float32(self.labels[index]))
        return tokens, torch.from_numpy(lexical), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated real-only hybrid character/lexical CNN.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hybrid_cnn"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-gpu-temperature", type=int, default=84)
    return parser.parse_args()


def gpu_temperature() -> int:
    return int(subprocess.check_output(
        ("nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"),
        text=True,
    ).strip().splitlines()[0])


def thermal_guard(maximum: int, model: str, seed: int) -> None:
    if maximum <= 0:
        return
    temperature = gpu_temperature()
    if temperature < maximum:
        return
    resume_below = maximum - 6
    print(json.dumps({
        "model": model, "seed": seed, "thermal_pause_c": temperature,
        "resume_below_c": resume_below,
    }), flush=True)
    while temperature >= resume_below:
        time.sleep(15)
        temperature = gpu_temperature()
    print(json.dumps({
        "model": model, "seed": seed, "thermal_resume_c": temperature,
    }), flush=True)


def feature_statistics(path: Path, chunk_size: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path, mmap_mode="r")
    total = np.zeros(values.shape[1], dtype=np.float64)
    squared = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start:start + chunk_size], dtype=np.float64)
        total += chunk.sum(axis=0)
        squared += np.square(chunk).sum(axis=0)
    mean = total / len(values)
    variance = np.maximum(0.0, squared / len(values) - np.square(mean))
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_loaders(
    data_dir: Path, feature_dir: Path, mean: np.ndarray, std: np.ndarray,
    batch_size: int, workers: int,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            HybridURLDataset(data_dir, feature_dir, split, mean, std),
            batch_size=batch_size, shuffle=split == "train", num_workers=workers,
            pin_memory=True, persistent_workers=workers > 0,
        )
        for split in ("train", "validation", "test")
    }


@torch.inference_mode()
def score(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval(); probabilities: list[np.ndarray] = []; labels: list[np.ndarray] = []
    started = time.perf_counter()
    for tokens, lexical, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        lexical = lexical.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(tokens, lexical)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy())
    torch.cuda.synchronize()
    return np.concatenate(probabilities), np.concatenate(labels).astype(np.uint8), time.perf_counter() - started


def train_seed(
    seed: int, project: dict, data_dir: Path, feature_dir: Path, output_dir: Path,
    mean: np.ndarray, std: np.ndarray, workers: int,
    max_gpu_temperature: int,
) -> dict:
    device = require_cuda(); set_seed(seed)
    vocabulary = load_json(data_dir / "vocabulary.json")
    config = project["baseline_models"]["hybrid_cnn"]
    training = project["baseline_training"]
    model = HybridCharacterCNN(
        vocabulary_size=len(vocabulary), lexical_feature_count=len(LEXICAL_FEATURE_NAMES),
        embedding_dim=config["embedding_dim"],
        channels_per_kernel=config["channels_per_kernel"],
        kernel_sizes=tuple(config["kernel_sizes"]),
        lexical_hidden_dim=config["lexical_hidden_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"], dropout=config["dropout"],
    ).to(device)
    loaders = make_loaders(
        data_dir, feature_dir, mean, std, int(config["batch_size"]), workers
    )
    train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    positive_weight = float((train_labels == 0).sum() / max(1, (train_labels == 1).sum()))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict] = []; best_pr_auc = -np.inf; best_state = None
    best_epoch = 0; stale = 0; training_seconds = 0.0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); loss_sum = 0.0; seen = 0; started = time.perf_counter()
        for batch_index, (tokens, lexical, targets) in enumerate(loaders["train"], start=1):
            if batch_index == 1 or batch_index % 50 == 0:
                thermal_guard(max_gpu_temperature, "hybrid_cnn", seed)
            tokens = tokens.to(device, non_blocking=True)
            lexical = lexical.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(tokens, lexical), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.item()) * len(targets); seen += len(targets)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        training_seconds += elapsed
        validation_scores, validation_labels, _ = score(model, loaders["validation"], device)
        validation_pr_auc = float(average_precision_score(validation_labels, validation_scores))
        record = {
            "epoch": epoch, "train_loss": loss_sum / max(1, seen),
            "validation_pr_auc": validation_pr_auc, "training_seconds": elapsed,
        }
        history.append(record)
        print(json.dumps({"model": "hybrid_cnn", "seed": seed, **record}), flush=True)
        if validation_pr_auc > best_pr_auc:
            best_pr_auc = validation_pr_auc
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Hybrid training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_scores, validation_labels, validation_time = score(model, loaders["validation"], device)
    threshold = select_threshold(validation_labels, validation_scores, maximum_fpr=0.01)
    test_scores, test_labels, test_time = score(model, loaders["test"], device)
    metrics = classification_metrics(test_labels, test_scores, threshold["threshold"])
    run_dir = output_dir / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "project_version": 2, "condition": "real_only", "model": "hybrid_cnn", "seed": seed,
        "model_config": config, "vocabulary": vocabulary, "state_dict": best_state,
        "threshold": threshold["threshold"], "best_epoch": best_epoch,
        "feature_names": list(LEXICAL_FEATURE_NAMES),
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "master_sha256": project["data"]["master_sha256"],
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
        predictions=(test_scores >= threshold["threshold"]).astype(np.uint8),
    )
    payload = {
        **metrics, "project_version": 2, "condition": "real_only", "model": "hybrid_cnn",
        "seed": seed, "threshold": threshold["threshold"], "threshold_selection": threshold,
        "best_epoch": best_epoch, "validation_pr_auc": best_pr_auc,
        "training_time_seconds": training_seconds,
        "validation_inference_seconds": validation_time, "test_inference_seconds": test_time,
        "inference_urls_per_second": len(test_labels) / test_time,
        "model_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
        "device": torch.cuda.get_device_name(device), "mixed_precision": True,
        "master_sha256": project["data"]["master_sha256"],
        "real_training_rows": len(train_labels), "generated_training_rows": 0,
        "lexical_feature_count": len(LEXICAL_FEATURE_NAMES),
        "normalization_fit": "frozen real training split only",
    }
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metrics.json", payload)
    return payload


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    mean, std = feature_statistics(args.feature_dir / "train_features.npy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "feature_normalization.npz", mean=mean, std=std,
        feature_names=np.asarray(LEXICAL_FEATURE_NAMES),
    )
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        metrics_path = args.output_dir / f"seed_{seed}" / "metrics.json"
        if metrics_path.exists():
            print(json.dumps({"seed": seed, "status": "already_complete"}), flush=True)
            continue
        result = train_seed(
            seed, project, args.data_dir, args.feature_dir, args.output_dir,
            mean, std, args.num_workers, args.max_gpu_temperature,
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
