from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .autoregressive_models import AutoregressiveURLGenerator, to_base_tokens
from .common import load_json, require_cuda, seed_everything
from .data import AutoregressiveMaliciousDataset
from .train_gan import decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the autoregressive URL generator.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generator_pretrained"))
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional pretrained generator checkpoint to adapt instead of starting from random weights.",
    )
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--validation-samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=1_000)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


@torch.inference_mode()
def validation_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model.teacher_logits(inputs)
            loss = criterion(logits.flatten(0, 1), targets.flatten())
        count = targets.ne(0).sum().item()
        total += loss.item() * count
        tokens += count
    return total / max(tokens, 1)


@torch.inference_mode()
def save_samples(model, args, metadata, epoch, device) -> None:
    model.eval()
    noise = torch.randn(args.sample_count, args.latent_dim, device=device)
    generated = model.generate(
        noise,
        bos_id=len(metadata["vocab"]),
        sequence_length=metadata["max_length"],
        temperature=args.sample_temperature,
        differentiable=False,
    )
    base_tokens = to_base_tokens(
        generated,
        base_vocab_size=len(metadata["vocab"]),
        bos_id=len(metadata["vocab"]),
        eos_id=len(metadata["vocab"]) + 1,
    ).cpu().numpy()
    id_to_token = {
        value: key for key, value in metadata["vocab"].items() if len(key) == 1
    }
    urls = decode(base_tokens, id_to_token)
    np.save(args.output_dir / f"samples_epoch_{epoch:03d}.npy", base_tokens)
    (args.output_dir / f"samples_epoch_{epoch:03d}.txt").write_text(
        "\n".join(urls), encoding="utf-8"
    )
    print("samples:", flush=True)
    for url in urls[:5]:
        print(f"  {url}", flush=True)


def main() -> None:
    args = parse_args()
    device = require_cuda()
    seed_everything(args.seed)
    metadata = load_json(args.data_dir / "metadata.json")
    base_vocab_size = len(metadata["vocab"])
    extended_vocab_size = base_vocab_size + 2
    train_data = AutoregressiveMaliciousDataset(
        args.data_dir,
        split="train",
        max_samples=args.max_samples,
        seed=args.seed,
        base_vocab_size=base_vocab_size,
    )
    validation_data = AutoregressiveMaliciousDataset(
        args.data_dir,
        split="val",
        max_samples=args.validation_samples,
        seed=args.seed + 1,
        base_vocab_size=base_vocab_size,
    )
    train_loader = make_loader(train_data, args, True)
    validation_loader = make_loader(validation_data, args, False)
    model_args = {
        "vocab_size": extended_vocab_size,
        "latent_dim": args.latent_dim,
    }
    model = AutoregressiveURLGenerator(**model_args).to(device)
    adapted_from = None
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        if initial["model_args"] != model_args:
            raise ValueError(
                f"Initial model arguments {initial['model_args']} do not match requested {model_args}."
            )
        if initial["metadata"]["vocab"] != metadata["vocab"]:
            raise ValueError("Initial checkpoint and adaptation data use different vocabularies.")
        model.load_state_dict(initial["model_state"])
        adapted_from = str(args.initial_checkpoint)
        print(f"Adapting generator from {adapted_from}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model.teacher_logits(inputs)
                loss = criterion(logits.flatten(0, 1), targets.flatten())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = targets.ne(0).sum().item()
            total_loss += loss.item() * count
            total_tokens += count

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = validation_loss(model, validation_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_cross_entropy": train_loss,
            "validation_cross_entropy": val_loss,
            "validation_perplexity": math.exp(min(val_loss, 20)),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        save_samples(model, args, metadata, epoch, device)
        checkpoint = {
            "model_state": model.state_dict(),
            "model_args": model_args,
            "metadata": metadata,
            "bos_id": base_vocab_size,
            "eos_id": base_vocab_size + 1,
            "epoch": epoch,
            "history": history,
            "training_args": vars(args),
            "adapted_from": adapted_from,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print(f"Saved pretrained generator to {args.output_dir}")


if __name__ == "__main__":
    main()
