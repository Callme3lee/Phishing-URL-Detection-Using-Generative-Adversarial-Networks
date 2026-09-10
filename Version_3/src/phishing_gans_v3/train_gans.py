from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import load_json, require_cuda, set_seed, write_json
from .gan_models import (AutoregressiveGenerator, GumbelLSTMGenerator, RelGANDiscriminator,
                         RelGANGenerator, SequenceDiscriminator)
from .gan_utils import basic_quality, decode_tokens, rollout_rewards, sample_hard, teacher_inputs
from .url_data import BOS, EOS, PAD


class PhishingTokens(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.tokens = np.load(data_dir / "train_tokens.npy", mmap_mode="r")
        self.indices = np.load(data_dir / "train_malicious_indices.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.tokens[int(self.indices[index])], dtype=np.int64, copy=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen Version 3 SeqGAN or RelGAN on CUDA.")
    parser.add_argument("--model", choices=("seqgan", "relgan", "urlgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/generators"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader); return next(iterator), iterator


def create_models(name: str, vocabulary_size: int, project: dict, device: torch.device):
    shared = project["generators"]["shared"]; config = project["generators"][name]
    if name == "seqgan":
        generator = AutoregressiveGenerator(vocabulary_size, shared["embedding_dim"], shared["hidden_dim"])
        discriminator = SequenceDiscriminator(vocabulary_size, shared["embedding_dim"],
                                               config["discriminator_channels"],
                                               tuple(config["discriminator_kernel_sizes"]))
    elif name == "relgan":
        generator = RelGANGenerator(vocabulary_size, shared["embedding_dim"], config["memory_slots"],
                                    config["memory_size"], config["attention_heads"], config["latent_dim"])
        discriminator = RelGANDiscriminator(vocabulary_size, shared["embedding_dim"],
                                            config["discriminator_representations"],
                                            config["discriminator_channels"],
                                            tuple(config["discriminator_kernel_sizes"]))
    else:
        generator = GumbelLSTMGenerator(
            vocabulary_size, shared["embedding_dim"], shared["hidden_dim"],
            layers=int(config["generator_layers"]), dropout=float(config["generator_dropout"]),
        )
        discriminator = SequenceDiscriminator(
            vocabulary_size, shared["embedding_dim"], config["discriminator_channels"],
            tuple(config["discriminator_kernel_sizes"]), dropout=float(config["discriminator_dropout"]),
        )
    return generator.to(device), discriminator.to(device)


def mle_loss(name: str, generator, real: torch.Tensor, bos_id: int) -> torch.Tensor:
    inputs = teacher_inputs(real, bos_id)
    logits = generator(inputs)[0] if name in {"seqgan", "urlgan"} else generator(inputs)
    return F.cross_entropy(logits.flatten(0, 1), real.flatten(), ignore_index=0)


@torch.inference_mode()
def quality_sample(name: str, generator, count: int, shared: dict, vocabulary: dict,
                   device: torch.device, protected: set[str]) -> tuple[dict, list[str]]:
    generator.eval(); inverse = {int(value): token for token, value in vocabulary.items()}; urls = []
    while len(urls) < count:
        batch = min(int(shared["batch_size"]), count - len(urls))
        if name == "seqgan":
            ids, _ = sample_hard(generator, batch, int(shared["maximum_length"]), vocabulary[BOS],
                                 vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device)
        elif name == "urlgan":
            ids = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                       vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device)
        else:
            ids = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                       float(shared["temperature"]), device)
        urls.extend(decode_tokens(row, inverse) for row in ids.cpu().numpy())
    return basic_quality(urls, protected), urls


def set_trainable(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def main() -> None:
    args = parse_args(); project = load_json(args.config); shared = project["generators"]["shared"]
    config = project["generators"][args.model]; device = require_cuda(); seed = int(project["seed"]); set_seed(seed)
    mle_epochs = int(config.get("mle_epochs", shared["mle_epochs"]))
    adversarial_epochs = int(config.get("adversarial_epochs", shared["adversarial_epochs"]))
    vocabulary = load_json(args.data_dir / "vocabulary.json")
    dataset = PhishingTokens(args.data_dir)
    if len(dataset) != int(shared["training_rows"]):
        raise RuntimeError(f"GAN contract expected {shared['training_rows']} phishing rows, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=int(shared["batch_size"]), shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    generator, discriminator = create_models(args.model, len(vocabulary), project, device)
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["generator_learning_rate"]), weight_decay=1e-4)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(config["discriminator_learning_rate"]), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda"); output = args.output_root / args.model; output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"; history_path = output / "history.json"; history = []
    stage, start_epoch, best_score, best_g, best_d = "mle", 1, -float("inf"), None, None
    if args.resume:
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["master_sha256"] != project["data"]["master_sha256"]:
            raise RuntimeError("Checkpoint belongs to another dataset")
        generator.load_state_dict(saved["generator_state"]); discriminator.load_state_dict(saved["discriminator_state"])
        g_optimizer.load_state_dict(saved["generator_optimizer"]); d_optimizer.load_state_dict(saved["discriminator_optimizer"])
        stage, start_epoch = saved["stage"], int(saved["epoch"]) + 1
        best_score, best_g, best_d = saved["best_score"], saved["best_generator"], saved["best_discriminator"]
        history = load_json(history_path)
        if stage == "mle" and start_epoch > mle_epochs:
            stage, start_epoch = "adversarial", 1

    protected = set()
    with (args.data_dir / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) == 1:
                protected.add(row["url"])
    iterator = iter(loader)

    def save(stage_name: str, epoch: int) -> None:
        torch.save({
            "project_version": 3, "model": args.model, "stage": stage_name, "epoch": epoch,
            "generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
            "generator_optimizer": g_optimizer.state_dict(), "discriminator_optimizer": d_optimizer.state_dict(),
            "best_score": best_score, "best_generator": best_g, "best_discriminator": best_d,
            "vocabulary": vocabulary, "generator_config": {"shared": shared, "specific": config},
            "master_sha256": project["data"]["master_sha256"], "training_rows": len(dataset),
        }, latest)

    if stage == "mle":
        for epoch in range(start_epoch, mle_epochs + 1):
            generator.train(); losses = []; started = time.perf_counter()
            mle_steps = int(config.get("mle_steps_per_epoch", len(loader)))
            for step, real in enumerate(loader, start=1):
                if step > mle_steps:
                    break
                real = real.to(device, non_blocking=True); g_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16): loss = mle_loss(args.model, generator, real, vocabulary[BOS])
                scaler.scale(loss).backward(); scaler.unscale_(g_optimizer); nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                scaler.step(g_optimizer); scaler.update(); losses.append(float(loss.item()))
            quality, urls = quality_sample(args.model, generator, int(shared["quality_sample_count"]), shared,
                                           vocabulary, device, protected)
            record = {"stage": "mle", "epoch": epoch, "loss": float(np.mean(losses)),
                      "eligible_training_rows": len(dataset), "batches_consumed": len(losses),
                      "examples_consumed": len(losses) * int(shared["batch_size"]),
                      "seconds": time.perf_counter() - started, "quality": quality}
            history.append(record); write_json(history_path, history)
            write_json(output / f"mle_epoch_{epoch:02d}.json", record)
            (output / f"mle_epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            save("mle", epoch); print(json.dumps(record), flush=True)
        stage, start_epoch = "adversarial", 1

    for epoch in range(start_epoch, adversarial_epochs + 1):
        generator.train(); discriminator.train(); g_losses = []; d_losses = []; started = time.perf_counter()
        for _ in range(int(config["adversarial_steps_per_epoch"])):
            real, iterator = next_batch(iterator, loader); real = real.to(device, non_blocking=True); batch = len(real)
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if args.model == "seqgan":
                    fake, _ = sample_hard(generator, batch, int(shared["maximum_length"]), vocabulary[BOS],
                                          vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device)
                elif args.model == "urlgan":
                    fake = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                                vocabulary[EOS], vocabulary[PAD],
                                                float(shared["temperature"]), device)
                else:
                    fake = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                                float(shared["temperature"]), device)
            # sample_ids intentionally runs in inference mode. Clone after leaving
            # that context so the discriminator can retain its embedding output
            # for its own backward pass.
            fake = fake.clone()
            with torch.autocast("cuda", dtype=torch.float16):
                real_scores, _ = discriminator(real); fake_scores, _ = discriminator(fake)
                d_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) + \
                         F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            scaler.scale(d_loss).backward(); scaler.unscale_(d_optimizer); nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(d_optimizer); scaler.update(); d_losses.append(float(d_loss.item()))

            set_trainable(discriminator, False); g_optimizer.zero_grad(set_to_none=True)
            language = mle_loss(args.model, generator, real, vocabulary[BOS])
            if args.model == "seqgan":
                sampled, logs = sample_hard(generator, batch, int(shared["maximum_length"]), vocabulary[BOS],
                                            vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device, True)
                rewards = rollout_rewards(generator, discriminator, sampled, vocabulary[BOS], vocabulary[EOS],
                                          vocabulary[PAD], float(shared["temperature"]), int(config["rollout_count"]),
                                          int(config["rollout_stride"])); baseline = rewards.mean(0, keepdim=True)
                mask = sampled.ne(vocabulary[PAD]).float()
                policy = -(((rewards - baseline).detach() * logs * mask).sum() / mask.sum().clamp(min=1))
                g_loss = policy + float(config["mle_regularization_weight"]) * language
            else:
                if args.model == "urlgan":
                    soft, _ = generator.sample(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                               vocabulary[EOS], vocabulary[PAD],
                                               float(shared["temperature"]), device, hard=False)
                else:
                    soft, _ = generator.sample(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                               float(shared["temperature"]), device, hard=False)
                fake_scores, fake_features = discriminator.forward_soft(soft)
                with torch.no_grad(): _, real_features = discriminator(real)
                adversarial = F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
                feature = (fake_features.mean(0) - real_features.mean(0)).square().mean()
                marginal = soft.mean((0, 1)); diversity = (marginal * torch.log(marginal.clamp(min=1e-8))).sum()
                g_loss = adversarial + float(config["mle_regularization_weight"]) * language + \
                         float(config["feature_matching_weight"]) * feature + float(config["diversity_weight"]) * diversity
            g_loss.backward(); nn.utils.clip_grad_norm_(generator.parameters(), 1.0); g_optimizer.step()
            set_trainable(discriminator, True); g_losses.append(float(g_loss.item()))

        quality, urls = quality_sample(args.model, generator, int(shared["quality_sample_count"]), shared,
                                       vocabulary, device, protected)
        selection = quality["validity"] * quality["uniqueness"] * (1 - quality["memorization_rate"]) * \
                    (1 - quality["repeated_template_rate"])
        if selection > best_score:
            best_score, best_g, best_d = selection, deepcopy(generator.state_dict()), deepcopy(discriminator.state_dict())
        record = {"stage": "adversarial", "epoch": epoch, "generator_loss": float(np.mean(g_losses)),
                  "discriminator_loss": float(np.mean(d_losses)), "selection_score": selection,
                  "seconds": time.perf_counter() - started, "quality": quality}
        history.append(record); write_json(history_path, history); write_json(output / f"adversarial_epoch_{epoch:02d}.json", record)
        (output / f"adversarial_epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        save("adversarial", epoch); print(json.dumps(record), flush=True)

    torch.save({"project_version": 3, "model": args.model, "generator_state": best_g,
                "discriminator_state": best_d, "vocabulary": vocabulary,
                "generator_config": {"shared": shared, "specific": config},
                "selection_score": best_score, "master_sha256": project["data"]["master_sha256"],
                "training_rows": len(dataset)}, output / "best.pt")
    del generator, discriminator, loader
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
