from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, seed_everything
from .data import (
    AugmentedEncodedURLDataset,
    EncodedURLDataset,
    FractionalEncodedURLDataset,
)
from .models import create_character_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPU character classifier baseline.")
    parser.add_argument("--model", choices=("cnn", "lstm"), default="cnn")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible checkpoint used to initialize the model before training.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation", choices=("none", "oversample", "synthetic"), default="none"
    )
    parser.add_argument("--augmentation-count", type=int, default=0)
    parser.add_argument("--synthetic-tokens", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--max-validation-fpr",
        type=float,
        default=None,
        help="Optional maximum validation false-positive rate used when choosing the threshold.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only final test metrics")
    return parser.parse_args()


def loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


@torch.inference_mode()
def predict(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    probabilities = []
    labels = []
    for tokens, targets in data_loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(tokens)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels).astype(np.uint8), np.concatenate(probabilities)


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(scores[:-1]))
    return float(thresholds[best])


def constrained_fpr_threshold(
    labels: np.ndarray, probabilities: np.ndarray, max_fpr: float
) -> float:
    if not 0 <= max_fpr < 1:
        raise ValueError("--max-validation-fpr must be in [0, 1).")
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    group_ends = np.flatnonzero(
        np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    )
    tp = true_positives[group_ends].astype(float)
    fp = false_positives[group_ends].astype(float)
    positives = max(int(np.count_nonzero(labels == 1)), 1)
    negatives = max(int(np.count_nonzero(labels == 0)), 1)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    valid = fp / negatives <= max_fpr + 1e-12
    if not np.any(valid):
        return float(np.nextafter(sorted_probabilities.max(), np.inf))
    valid_indices = np.flatnonzero(valid)
    best = max(
        valid_indices,
        key=lambda index: (f1[index], recall[index], precision[index]),
    )
    return float(sorted_probabilities[group_ends[best]])


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "false_positive_rate": fp / max(fp + tn, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        base_name = "classifier" if args.model == "cnn" else "classifier_lstm"
        suffix = "" if args.augmentation == "none" else f"_{args.augmentation}"
        args.output_dir = Path(f"artifacts/{base_name}{suffix}")
    if args.batch_size is None:
        args.batch_size = 1024 if args.model == "cnn" else 512
    device = require_cuda()
    seed_everything(args.seed)
    metadata = load_json(args.data_dir / "metadata.json")
    if args.augmentation == "none":
        train_data = (
            EncodedURLDataset(args.data_dir, "train")
            if args.train_fraction == 1.0
            else FractionalEncodedURLDataset(
                args.data_dir, fraction=args.train_fraction, seed=args.seed
            )
        )
        added_positives = 0
    else:
        train_data = AugmentedEncodedURLDataset(
            args.data_dir,
            augmentation=args.augmentation,
            augmentation_count=args.augmentation_count,
            seed=args.seed,
            synthetic_tokens=args.synthetic_tokens,
            train_fraction=args.train_fraction,
        )
        added_positives = args.augmentation_count
    val_data = EncodedURLDataset(args.data_dir, "val")
    test_data = EncodedURLDataset(args.data_dir, "test")

    model_args = {"vocab_size": len(metadata["vocab"])}
    model = create_character_model(args.model, **model_args).to(device)
    initialization = None
    if args.initial_checkpoint is not None:
        initial = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
        if initial.get("model_type") != args.model:
            raise ValueError(
                f"Initial checkpoint is {initial.get('model_type')}, requested {args.model}."
            )
        if initial.get("model_args") != model_args:
            raise ValueError("Initial checkpoint model arguments do not match this dataset.")
        model.load_state_dict(initial["model_state"])
        initialization = {
            "checkpoint": str(args.initial_checkpoint),
            "source_epoch": initial.get("epoch"),
            "source_augmentation": initial.get("augmentation"),
        }
    positive = int(metadata["counts"]["train"]["label_1"] * args.train_fraction) + added_positives
    negative = int(metadata["counts"]["train"]["label_0"] * args.train_fraction)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    train_loader = loader(train_data, args, shuffle=True)
    val_loader = loader(val_data, args, shuffle=False)
    test_loader = loader(test_data, args, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_path = args.output_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for tokens, targets in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(tokens), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(targets)
            seen += len(targets)

        val_labels, val_probabilities = predict(model, val_loader, device)
        threshold = (
            best_f1_threshold(val_labels, val_probabilities)
            if args.max_validation_fpr is None
            else constrained_fpr_threshold(
                val_labels, val_probabilities, args.max_validation_fpr
            )
        )
        val_metrics = metrics(val_labels, val_probabilities, threshold)
        record = {"epoch": epoch, "train_loss": total_loss / seen, "validation": val_metrics}
        history.append(record)
        if not args.quiet:
            print(json.dumps(record, indent=2))
        if val_metrics["f1"] > best_score:
            best_score = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_args": model_args,
                    "model_type": args.model,
                    "threshold": threshold,
                    "metadata": metadata,
                    "epoch": epoch,
                    "augmentation": {
                        "type": args.augmentation,
                        "count": added_positives,
                        "synthetic_tokens": str(args.synthetic_tokens) if args.synthetic_tokens else None,
                        "train_fraction": args.train_fraction,
                    },
                    "initialization": initialization,
                    "threshold_policy": {
                        "method": "maximum validation F1"
                        if args.max_validation_fpr is None
                        else "maximum validation F1 subject to false-positive-rate constraint",
                        "max_validation_fpr": args.max_validation_fpr,
                    },
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_labels, test_probabilities = predict(model, test_loader, device)
    test_metrics = metrics(test_labels, test_probabilities, checkpoint["threshold"])
    results = {
        "history": history,
        "test": test_metrics,
        "best_epoch": checkpoint["epoch"],
        "augmentation": checkpoint["augmentation"],
        "initialization": checkpoint.get("initialization"),
        "threshold_policy": checkpoint.get("threshold_policy"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("Final real-URL test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
