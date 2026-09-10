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
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .metrics import classification_metrics, select_threshold
from .train_hybrid_domain import (
    CORAL_PAIRS, TARGET_GROUP_PROPORTIONS, HybridDataset, conditional_coral, create_model, score_model,
)
from .train_traditional import train as train_traditional


CONDITIONS = ("real_only", "gan_10pct", "gan_20pct", "gan_30pct", "gan_40pct")
VARIANTS = ("cnn_psl", "hyb_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen feature-equal and HYB-MS GAN matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--variants", default="random_forest,linear_svm,cnn_psl,hyb_ms")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--coral-weight", type=float, default=0.01)
    return parser.parse_args()


def make_loaders(data_dir: Path, variant: str, seed: int, batch_size: int, workers: int,
                 generated_fraction: float) -> tuple[dict[str, DataLoader], dict[int, float]]:
    train = HybridDataset(data_dir, "train", groups=True)
    groups = np.load(data_dir / "train_groups.npy", mmap_mode="r")
    target = {group: proportion * (1.0 - generated_fraction) for group, proportion in TARGET_GROUP_PROPORTIONS.items()}
    if generated_fraction:
        target[5] = generated_fraction
    if variant == "hyb_ms":
        counts = {group: max(1, int((groups == group).sum())) for group in target}
        weights = np.asarray([target[int(group)] / counts[int(group)] for group in groups], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(train), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=workers,
                                  pin_memory=True, persistent_workers=workers > 0)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers,
                                  pin_memory=True, persistent_workers=workers > 0,
                                  generator=torch.Generator().manual_seed(seed))
    loaders = {"train": train_loader}
    for split in ("validation", "domain_validation", "test"):
        loaders[split] = DataLoader(HybridDataset(data_dir, split), batch_size=batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return loaders, target


def train_neural(variant: str, condition: str, seed: int, args: argparse.Namespace, project: dict) -> dict:
    run_dir = args.output_root / condition / variant / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(json.dumps({"stage": "skip_completed", "condition": condition, "variant": variant, "seed": seed}), flush=True)
        return load_json(metrics_path)
    data_dir = args.data_root / condition / "processed"
    report_path = args.data_root / condition / "experiment_report.json"
    report = load_json(report_path)
    device = require_cuda(); set_seed(seed)
    vocabulary = load_json(data_dir / "vocabulary.json")
    lexical_metadata = load_json(data_dir / "psl_lexical_metadata.json")
    model_config = {"embedding_dim": 48, "channels_per_kernel": 128, "kernel_sizes": [3, 4, 5],
                    "lexical_hidden": 64, "lexical_embedding": 32, "fusion_hidden": 128,
                    "dropout": 0.4, "lexical_feature_count": int(lexical_metadata["feature_count"])}
    model = create_model(len(vocabulary), model_config).to(device)
    loaders, target_proportions = make_loaders(data_dir, variant, seed, args.batch_size, args.num_workers,
                                               float(report["generated_fraction"]))
    labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    if variant == "cnn_psl":
        positive_weight = float((labels == 0).sum() / max(1, int((labels == 1).sum())))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    training = project["neural_training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")
    best_score = -np.inf; best_state = None; best_epoch = stale = 0; total_seconds = 0.0; history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); classification_total = coral_total = 0.0; seen = 0; observed = Counter(); started = time.perf_counter()
        for tokens, lexical, targets, groups in loaders["train"]:
            tokens = tokens.to(device, non_blocking=True); lexical = lexical.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True); groups = groups.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, representation = model(tokens, lexical, return_representation=True)
                classification_loss = criterion(logits, targets)
            coral_loss = conditional_coral(representation, groups) if variant == "hyb_ms" else representation.new_zeros(())
            loss = classification_loss + (float(args.coral_weight) * coral_loss if variant == "hyb_ms" else 0.0)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update()
            classification_total += float(classification_loss.item()) * len(targets)
            coral_total += float(coral_loss.item()) * len(targets); seen += len(targets)
            observed.update(groups.detach().cpu().tolist())
        torch.cuda.synchronize(); seconds = time.perf_counter() - started; total_seconds += seconds
        val_scores, val_labels, val_seconds = score_model(model, loaders["validation"], device)
        domain_scores, domain_labels, domain_seconds = score_model(model, loaders["domain_validation"], device)
        val_pr = float(average_precision_score(val_labels, val_scores)); domain_pr = float(average_precision_score(domain_labels, domain_scores))
        selection = val_pr if variant == "cnn_psl" else 0.5 * (val_pr + domain_pr)
        record = {"epoch": epoch, "classification_loss": classification_total / seen,
                  "coral_loss": coral_total / seen, "validation_pr_auc": val_pr,
                  "domain_validation_pr_auc": domain_pr,
                  "domain_validation_roc_auc": float(roc_auc_score(domain_labels, domain_scores)),
                  "selection_score": selection, "training_seconds": seconds,
                  "validation_seconds": val_seconds, "domain_validation_seconds": domain_seconds,
                  "observed_group_proportions": {str(k): v / seen for k, v in observed.items()}}
        history.append(record); print(json.dumps({"condition": condition, "variant": variant, "seed": seed, **record}), flush=True)
        if selection > best_score:
            best_score = selection; best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]): break
    if best_state is None: raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_scores, val_labels, _ = score_model(model, loaders["validation"], device)
    domain_scores, domain_labels, _ = score_model(model, loaders["domain_validation"], device)
    original_threshold = select_threshold(val_labels, val_scores, maximum_fpr=0.01)
    domain_threshold = select_threshold(domain_labels, domain_scores, maximum_fpr=0.01)
    robust_threshold = max(float(original_threshold["threshold"]), float(domain_threshold["threshold"]))
    test_scores, test_labels, test_seconds = score_model(model, loaders["test"], device)
    fair_metrics = classification_metrics(test_labels, test_scores, float(original_threshold["threshold"]))
    robust_metrics = classification_metrics(test_labels, test_scores, robust_threshold)
    primary = fair_metrics if variant == "cnn_psl" else robust_metrics
    scaler_values = np.load(data_dir / "psl_lexical_scaler.npz")
    checkpoint = {"project_version": 3, "condition": condition, "model": variant, "seed": seed,
                  "model_config": model_config, "vocabulary": vocabulary, "state_dict": best_state,
                  "threshold": float(original_threshold["threshold"]) if variant == "cnn_psl" else robust_threshold,
                  "original_validation_threshold": original_threshold, "domain_validation_threshold": domain_threshold,
                  "robust_threshold": robust_threshold, "best_epoch": best_epoch,
                  "coral_weight": float(args.coral_weight) if variant == "hyb_ms" else 0.0,
                  "target_group_proportions": target_proportions, "coral_pairs": CORAL_PAIRS,
                  "synthetic_coral_excluded": True, "lexical_feature_names": lexical_metadata["feature_names"],
                  "lexical_scaler_center": scaler_values["center"], "lexical_scaler_scale": scaler_values["scale"],
                  "data_report_sha256": sha256_file(report_path)}
    run_dir.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, run_dir / "checkpoint.pt")
    np.savez_compressed(run_dir / "test_predictions.npz", labels=test_labels, scores=test_scores,
                        fair_predictions=(test_scores >= float(original_threshold["threshold"])).astype(np.uint8),
                        robust_predictions=(test_scores >= robust_threshold).astype(np.uint8))
    payload = {**primary, "project_version": 3, "condition": condition, "model": variant, "seed": seed,
               "threshold": checkpoint["threshold"], "threshold_policy": "original validation 1%-FPR" if variant == "cnn_psl" else "max original/domain validation 1%-FPR",
               "fair_original_threshold_metrics": fair_metrics, "robust_threshold_metrics": robust_metrics,
               "original_validation_threshold": original_threshold, "domain_validation_threshold": domain_threshold,
               "best_epoch": best_epoch, "selection_score": best_score, "training_time_seconds": total_seconds,
               "test_inference_seconds": test_seconds, "device": torch.cuda.get_device_name(device),
               "mixed_precision": True, "real_training_rows": int(report["real_training_rows"]),
               "synthetic_training_rows": int(report["synthetic_training_rows"]),
               "training_rows": int(report["training_rows"]), "generated_fraction": float(report["generated_fraction"]),
               "feature_count": int(lexical_metadata["feature_count"]), "coral_weight": checkpoint["coral_weight"],
               "target_group_proportions": target_proportions, "synthetic_coral_excluded": True}
    write_json(run_dir / "history.json", history); write_json(metrics_path, payload)
    del model, loaders, optimizer, scaler; gc.collect(); torch.cuda.empty_cache(); return payload


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    conditions = [x for x in args.conditions.split(",") if x]
    variants = [x for x in args.variants.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if set(conditions) - set(CONDITIONS): raise ValueError("Unknown condition")
    if set(variants) - {"random_forest", "linear_svm", *VARIANTS}: raise ValueError("Unknown variant")
    if any(v in VARIANTS for v in variants): require_cuda()
    for condition in conditions:
        data_dir = args.data_root / condition / "processed"
        report = load_json(args.data_root / condition / "experiment_report.json")
        for seed in seeds:
            for variant in variants:
                if variant in {"random_forest", "linear_svm"}:
                    result = train_traditional(variant, seed, project, data_dir, data_dir,
                                               args.output_root / condition, f"fair_psl_{condition}")
                    result.update({"real_training_rows": int(report["real_training_rows"]),
                                   "synthetic_training_rows": int(report["synthetic_training_rows"]),
                                   "training_rows": int(report["training_rows"]),
                                   "generated_fraction": float(report["generated_fraction"]),
                                   "feature_count": 35, "feature_view": "PSL-aware lexical"})
                    write_json(args.output_root / condition / variant / f"seed_{seed}" / "metrics.json", result)
                else:
                    result = train_neural(variant, condition, seed, args, project)
                print(json.dumps({"condition": condition, "variant": variant, "seed": seed,
                                  "f1": result["f1"], "roc_auc": result["roc_auc"]}), flush=True)


if __name__ == "__main__":
    main()
