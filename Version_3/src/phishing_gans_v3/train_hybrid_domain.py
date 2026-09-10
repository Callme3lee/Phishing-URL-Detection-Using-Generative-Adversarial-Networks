from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .metrics import classification_metrics, select_threshold
from .models import HybridCharacterCNN


TARGET_GROUP_PROPORTIONS = {0: 0.35, 1: 0.15, 2: 0.30, 3: 0.15, 4: 0.05}
CORAL_PAIRS = ((0, 1), (2, 3), (2, 4), (3, 4))


class HybridDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, groups: bool = False) -> None:
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.lexical = np.load(data_dir / f"{split}_psl_lexical.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.groups = np.load(data_dir / f"{split}_groups.npy", mmap_mode="r") if groups else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        values = (
            torch.from_numpy(np.array(self.tokens[index], dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(self.lexical[index], dtype=np.float32, copy=True)),
            torch.tensor(np.float32(self.labels[index])),
        )
        return (*values, torch.tensor(np.int64(self.groups[index]))) if self.groups is not None else values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fixed hybrid CNN + PSL lexical + CORAL condition on CUDA.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--experiment-report", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/experiment_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--coral-weight", type=float, default=0.01)
    parser.add_argument(
        "--sampling-mode", choices=("weighted", "natural"), default="weighted",
        help=("weighted reproduces HYB-MS source-balanced exposure; natural visits each "
              "stored training row once per epoch with ordinary seeded shuffling"),
    )
    parser.add_argument(
        "--condition", default="hybrid_cnn_coral_v1",
        help="Condition name recorded in new checkpoints and metrics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def coral_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if len(left) < 2 or len(right) < 2:
        return left.new_zeros(())
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    left_covariance = left.T @ left / (len(left) - 1)
    right_covariance = right.T @ right / (len(right) - 1)
    dimension = left.shape[1]
    return (left_covariance - right_covariance).square().sum() / (4.0 * dimension * dimension)


def conditional_coral(representation: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    losses = []
    for left_group, right_group in CORAL_PAIRS:
        left = representation[groups == left_group]
        right = representation[groups == right_group]
        if len(left) >= 2 and len(right) >= 2:
            losses.append(coral_pair(left, right))
    return torch.stack(losses).mean() if losses else representation.new_zeros(())


@torch.inference_mode()
def score_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores, labels = [], []
    started = time.perf_counter()
    for batch in loader:
        tokens, lexical, targets = batch[:3]
        tokens = tokens.to(device, non_blocking=True)
        lexical = lexical.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(tokens, lexical)
        # Keep the model forward pass in mixed precision, but compute probabilities
        # in FP32. FP16 sigmoid saturates/quantizes high-confidence scores near 1.0
        # and can force an unnecessarily conservative operating threshold.
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(targets.numpy())
    torch.cuda.synchronize()
    return np.concatenate(scores), np.concatenate(labels).astype(np.uint8), time.perf_counter() - started


def create_model(vocabulary_size: int, model_config: dict) -> HybridCharacterCNN:
    return HybridCharacterCNN(
        vocabulary_size, int(model_config["lexical_feature_count"]),
        embedding_dim=int(model_config["embedding_dim"]),
        channels_per_kernel=int(model_config["channels_per_kernel"]),
        kernel_sizes=tuple(model_config["kernel_sizes"]),
        lexical_hidden=int(model_config["lexical_hidden"]),
        lexical_embedding=int(model_config["lexical_embedding"]),
        fusion_hidden=int(model_config["fusion_hidden"]), dropout=float(model_config["dropout"]),
    )


def train_one(seed: int, args: argparse.Namespace, project: dict) -> dict:
    run_dir = args.output_dir / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(json.dumps({"stage": "skip_completed", "seed": seed}), flush=True)
        return load_json(metrics_path)
    device = require_cuda()
    set_seed(seed)
    vocabulary = load_json(args.data_dir / "vocabulary.json")
    lexical_metadata = load_json(args.data_dir / "psl_lexical_metadata.json")
    model_config = {
        "embedding_dim": 48, "channels_per_kernel": 128, "kernel_sizes": [3, 4, 5],
        "lexical_hidden": 64, "lexical_embedding": 32, "fusion_hidden": 128,
        "dropout": 0.4, "lexical_feature_count": int(lexical_metadata["feature_count"]),
    }
    model = create_model(len(vocabulary), model_config).to(device)
    train_dataset = HybridDataset(args.data_dir, "train", groups=True)
    train_groups = np.load(args.data_dir / "train_groups.npy", mmap_mode="r")
    if args.sampling_mode == "weighted":
        unknown_groups = sorted(set(np.unique(train_groups).tolist()) - set(TARGET_GROUP_PROPORTIONS))
        if unknown_groups:
            raise RuntimeError(f"Weighted mode has no target proportions for groups {unknown_groups}")
        group_counts = {group: max(1, int((train_groups == group).sum())) for group in TARGET_GROUP_PROPORTIONS}
        sample_weights = np.asarray([
            TARGET_GROUP_PROPORTIONS[int(group)] / group_counts[int(group)] for group in train_groups
        ], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights), num_samples=len(train_dataset), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=args.num_workers > 0,
        )
    else:
        # This is the fairness control: the exact same stored row multiset used by the
        # ordinary MS baselines, with every row visited once per epoch and no source
        # oversampling.  The generator makes the shuffle reproducible for each seed.
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=args.num_workers,
            pin_memory=True, persistent_workers=args.num_workers > 0,
        )
    loaders = {
        split: DataLoader(
            HybridDataset(args.data_dir, split), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        ) for split in ("validation", "domain_validation", "test")
    }
    criterion = nn.BCEWithLogitsLoss()
    training_config = project["neural_training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_score = -np.inf
    best_state = None
    best_epoch = stale = 0
    total_training_seconds = 0.0
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        classification_total = coral_total = 0.0
        seen = 0
        observed_groups = Counter()
        started = time.perf_counter()
        for tokens, lexical, targets, groups in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            lexical = lexical.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            groups = groups.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, representation = model(tokens, lexical, return_representation=True)
                classification_loss = criterion(logits, targets)
            coral_loss = conditional_coral(representation, groups)
            loss = classification_loss + float(args.coral_weight) * coral_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            classification_total += float(classification_loss.item()) * len(targets)
            coral_total += float(coral_loss.item()) * len(targets)
            seen += len(targets)
            observed_groups.update(groups.detach().cpu().tolist())
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        total_training_seconds += training_seconds
        validation_scores, validation_labels, validation_seconds = score_model(model, loaders["validation"], device)
        domain_scores, domain_labels, domain_seconds = score_model(model, loaders["domain_validation"], device)
        validation_pr_auc = float(average_precision_score(validation_labels, validation_scores))
        domain_pr_auc = float(average_precision_score(domain_labels, domain_scores))
        domain_roc_auc = float(roc_auc_score(domain_labels, domain_scores))
        selection_score = 0.5 * validation_pr_auc + 0.5 * domain_pr_auc
        record = {
            "epoch": epoch, "classification_loss": classification_total / seen,
            "coral_loss": coral_total / seen, "coral_weight": float(args.coral_weight),
            "validation_pr_auc": validation_pr_auc, "domain_validation_pr_auc": domain_pr_auc,
            "domain_validation_roc_auc": domain_roc_auc, "selection_score": selection_score,
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "domain_validation_seconds": domain_seconds,
            "observed_group_proportions": {str(group): count / seen for group, count in observed_groups.items()},
        }
        history.append(record)
        print(json.dumps({"model": "hybrid_cnn_coral", "seed": seed, **record}), flush=True)
        if selection_score > best_score:
            best_score = selection_score
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
    validation_scores, validation_labels, _ = score_model(model, loaders["validation"], device)
    domain_scores, domain_labels, _ = score_model(model, loaders["domain_validation"], device)
    original_threshold = select_threshold(validation_labels, validation_scores, maximum_fpr=0.01)
    domain_threshold = select_threshold(domain_labels, domain_scores, maximum_fpr=0.01)
    threshold = max(float(original_threshold["threshold"]), float(domain_threshold["threshold"]))
    validation_metrics = classification_metrics(validation_labels, validation_scores, threshold)
    domain_validation_metrics = classification_metrics(domain_labels, domain_scores, threshold)
    test_scores, test_labels, test_seconds = score_model(model, loaders["test"], device)
    metrics = classification_metrics(test_labels, test_scores, threshold)
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler_values = np.load(args.data_dir / "psl_lexical_scaler.npz")
    checkpoint = {
        "project_version": 3, "condition": args.condition, "model": "hybrid_cnn_coral",
        "seed": seed, "model_config": model_config, "vocabulary": vocabulary, "state_dict": best_state,
        "threshold": threshold, "original_validation_threshold": original_threshold,
        "domain_validation_threshold": domain_threshold, "best_epoch": best_epoch,
        "coral_weight": float(args.coral_weight), "sampling_mode": args.sampling_mode,
        "target_group_proportions": TARGET_GROUP_PROPORTIONS if args.sampling_mode == "weighted" else None,
        "lexical_feature_names": lexical_metadata["feature_names"],
        "lexical_scaler_center": scaler_values["center"], "lexical_scaler_scale": scaler_values["scale"],
        "data_report_sha256": sha256_file(args.experiment_report),
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
        predictions=(test_scores >= threshold).astype(np.uint8),
    )
    payload = {
        **metrics, "project_version": 3, "condition": args.condition,
        "model": "hybrid_cnn_coral", "seed": seed, "threshold": threshold,
        "threshold_policy": "max of independently selected 1% FPR thresholds on original and domain validation",
        "original_validation_threshold": original_threshold, "domain_validation_threshold": domain_threshold,
        "original_validation_metrics_at_robust_threshold": validation_metrics,
        "domain_validation_metrics_at_robust_threshold": domain_validation_metrics,
        "best_epoch": best_epoch, "selection_score": best_score, "training_time_seconds": total_training_seconds,
        "test_inference_seconds": test_seconds, "inference_urls_per_second": len(test_labels) / test_seconds,
        "model_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
        "device": torch.cuda.get_device_name(device), "mixed_precision": True,
        "train_rows": len(train_dataset), "coral_weight": float(args.coral_weight),
        "sampling_mode": args.sampling_mode,
        "training_row_policy": ("weighted sampling with replacement" if args.sampling_mode == "weighted"
                                else "each stored row exactly once per epoch with seeded shuffling"),
        "target_group_proportions": TARGET_GROUP_PROPORTIONS if args.sampling_mode == "weighted" else None,
        "data_report_sha256": sha256_file(args.experiment_report),
    }
    write_json(run_dir / "history.json", history)
    write_json(metrics_path, payload)
    del model, train_loader, loaders, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    report = load_json(args.experiment_report)
    if "condition" not in report:
        raise RuntimeError("Experiment report does not identify its condition")
    require_cuda()
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        result = train_one(seed, args, project)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
