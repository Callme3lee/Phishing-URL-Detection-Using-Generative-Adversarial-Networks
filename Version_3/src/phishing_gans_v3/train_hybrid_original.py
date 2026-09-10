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
from torch.utils.data import DataLoader, Dataset

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .metrics import classification_metrics, select_threshold
from .models import HybridCharacterCNN


class OriginalHybridDataset(Dataset):
    def __init__(self, base: Path, lexical_dir: Path, split: str) -> None:
        self.tokens = np.load(base / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(base / f"{split}_labels.npy", mmap_mode="r")
        self.lexical = np.load(lexical_dir / f"{split}_psl_lexical.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (torch.from_numpy(np.array(self.tokens[index], dtype=np.int64, copy=True)),
                torch.from_numpy(np.array(self.lexical[index], dtype=np.float32, copy=True)),
                torch.tensor(np.float32(self.labels[index])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HYB-ORIG on exactly the original baseline rows.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--lexical-dir", type=Path, default=Path("data/experiments/hybrid_original_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/hybrid_original_v1"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


@torch.inference_mode()
def score(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); outputs = []; labels = []; started = time.perf_counter()
    for tokens, lexical, targets in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(tokens.to(device, non_blocking=True), lexical.to(device, non_blocking=True))
        outputs.append(torch.sigmoid(logits).float().cpu().numpy()); labels.append(targets.numpy())
    torch.cuda.synchronize()
    return np.concatenate(outputs), np.concatenate(labels).astype(np.uint8), time.perf_counter() - started


def train_one(seed: int, args: argparse.Namespace, project: dict, report: dict) -> dict:
    run_dir = args.output_dir / f"seed_{seed}"
    if (run_dir / "metrics.json").exists():
        return load_json(run_dir / "metrics.json")
    set_seed(seed); device = require_cuda(); vocabulary = load_json(args.data_dir / "vocabulary.json")
    config = {"embedding_dim": 48, "channels_per_kernel": 128, "kernel_sizes": [3, 4, 5],
              "lexical_hidden": 64, "lexical_embedding": 32, "fusion_hidden": 128,
              "dropout": 0.4, "lexical_feature_count": int(report["feature_count"])}
    model = HybridCharacterCNN(len(vocabulary), config["lexical_feature_count"],
                               config["embedding_dim"], config["channels_per_kernel"], tuple(config["kernel_sizes"]),
                               config["lexical_hidden"], config["lexical_embedding"], config["fusion_hidden"],
                               config["dropout"]).to(device)
    datasets = {split: OriginalHybridDataset(args.data_dir, args.lexical_dir, split)
                for split in ("train", "validation", "test")}
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train",
                                 num_workers=args.num_workers, pin_memory=True,
                                 persistent_workers=args.num_workers > 0)
               for split, ds in datasets.items()}
    train_labels = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    positive_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    training = project["neural_training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda"); history = []; best_score = -np.inf; best_state = None
    best_epoch = stale = 0; total_seconds = 0.0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); running = 0.0; seen = 0; started = time.perf_counter()
        for tokens, lexical, targets in loaders["train"]:
            tokens = tokens.to(device, non_blocking=True); lexical = lexical.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(tokens, lexical), targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); running += float(loss.item()) * len(targets); seen += len(targets)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started; total_seconds += elapsed
        validation_scores, validation_labels, validation_seconds = score(model, loaders["validation"], device)
        validation_pr_auc = float(average_precision_score(validation_labels, validation_scores))
        record = {"epoch": epoch, "train_loss": running / seen, "validation_pr_auc": validation_pr_auc,
                  "training_seconds": elapsed, "validation_seconds": validation_seconds}
        history.append(record); print(json.dumps({"model": "HYB-ORIG", "seed": seed, **record}), flush=True)
        if validation_pr_auc > best_score:
            best_score = validation_pr_auc; best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]): break
    if best_state is None: raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_scores, validation_labels, _ = score(model, loaders["validation"], device)
    threshold = select_threshold(validation_labels, validation_scores, maximum_fpr=.01)
    test_scores, test_labels, test_seconds = score(model, loaders["test"], device)
    metrics = classification_metrics(test_labels, test_scores, threshold["threshold"])
    scaler_values = np.load(args.lexical_dir / "psl_lexical_scaler.npz")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"project_version": 3, "condition": "hybrid_original_v1", "display_name": "HYB-ORIG",
                  "model": "hybrid_character_cnn", "seed": seed, "model_config": config,
                  "vocabulary": vocabulary, "state_dict": best_state, "threshold": threshold["threshold"],
                  "threshold_selection": threshold, "best_epoch": best_epoch,
                  "lexical_feature_names": report["feature_names"],
                  "lexical_scaler_center": scaler_values["center"], "lexical_scaler_scale": scaler_values["scale"],
                  "preparation_report_sha256": sha256_file(args.lexical_dir / "preparation_report.json")}
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
                        predictions=(test_scores >= threshold["threshold"]).astype(np.uint8))
    payload = {**metrics, "project_version": 3, "condition": "hybrid_original_v1", "display_name": "HYB-ORIG",
               "seed": seed, "threshold": threshold["threshold"], "threshold_selection": threshold,
               "best_epoch": best_epoch, "validation_pr_auc": best_score, "training_time_seconds": total_seconds,
               "test_inference_seconds": test_seconds, "device": torch.cuda.get_device_name(device),
               "mixed_precision": True, "real_training_rows": len(train_labels), "coral": False,
               "source_balanced_sampling": False, "outside_training_rows": 0, "gan_training_rows": 0}
    write_json(run_dir / "history.json", history); write_json(run_dir / "metrics.json", payload)
    del model, loaders, datasets, optimizer, scaler; gc.collect(); torch.cuda.empty_cache()
    return payload


def main() -> None:
    args = parse_args(); project = load_json(args.config); report = load_json(args.lexical_dir / "preparation_report.json")
    if report["train_rows"] != 358619: raise RuntimeError("HYB-ORIG contract requires exactly 358,619 training rows")
    require_cuda()
    for seed in [int(value) for value in args.seeds.split(",")]:
        print(json.dumps(train_one(seed, args, project, report), indent=2), flush=True)


if __name__ == "__main__":
    main()
