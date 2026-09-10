"""CUDA-only, resumable RelGAN training on the frozen malicious train view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, set_seed, write_json
from .datasets import GeneratorDataset
from .generation import basic_generation_metrics, decode_tokens, teacher_forcing_inputs
from .relgan import RelGANDiscriminator, RelGANGenerator
from .url_data import BOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RelGAN on the Version 2 malicious training view.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generators/relgan"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def protected_hashes(path: Path) -> set[bytes]:
    connection = sqlite3.connect(path)
    hashes = {hashlib.blake2b(url.encode(), digest_size=16).digest()
              for (url,) in connection.execute("SELECT normalized_url FROM urls")}
    connection.close()
    return hashes


def make_models(vocabulary_size: int, config: dict, device: torch.device) -> tuple[nn.Module, nn.Module]:
    generator = RelGANGenerator(
        vocabulary_size, config["embedding_dim"], config["memory_slots"], config["memory_size"],
        config["attention_heads"], config["latent_dim"],
    ).to(device)
    discriminator = RelGANDiscriminator(
        vocabulary_size, config["embedding_dim"], config["discriminator_representations"],
        config["discriminator_channels_per_kernel"], tuple(config["discriminator_kernel_sizes"]),
        config["dropout"],
    ).to(device)
    return generator, discriminator


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def mle_loss(generator: RelGANGenerator, targets: torch.Tensor, bos_id: int) -> torch.Tensor:
    logits = generator(teacher_forcing_inputs(targets, bos_id))
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)


@torch.inference_mode()
def quality_audit(
    generator: RelGANGenerator, count: int, batch_size: int, maximum_length: int,
    bos_id: int, temperature: float, device: torch.device, inverse: dict[int, str],
    hashes: set[bytes],
) -> tuple[dict, list[str]]:
    generator.eval(); urls: list[str] = []
    while len(urls) < count:
        current = min(batch_size, count - len(urls))
        _, ids = generator.sample(current, maximum_length, bos_id, temperature, hard=True, device=device)
        urls.extend(decode_tokens(row, inverse) for row in ids.cpu().numpy())
    return basic_generation_metrics(urls, hashes), urls


def set_trainable(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def checkpoint_payload(
    project: dict, config: dict, vocabulary: dict, generator: nn.Module, discriminator: nn.Module,
    generator_optimizer, discriminator_optimizer, stage: str, epoch: int, best_score: float,
    best_generator: dict | None, best_discriminator: dict | None,
) -> dict:
    return {
        "project_version": 2, "model": "relgan", "stage": stage, "epoch": epoch,
        "generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "best_generator_state": best_generator, "best_discriminator_state": best_discriminator,
        "best_selection_score": best_score, "relgan_config": config, "vocabulary": vocabulary,
        "master_sha256": project["data"]["master_sha256"],
    }


def main() -> None:
    args = parse_args(); project = load_json(args.config); config = project["generators"]["relgan"]
    shared = project["generators"]["shared"]; device = require_cuda(); set_seed(int(project["seed"]))
    vocabulary = load_json(args.data_dir / "vocabulary.json"); inverse = {int(v): k for k, v in vocabulary.items()}
    bos_id = vocabulary[BOS]; maximum_length = int(project["data"]["maximum_encoded_length"])
    generator, discriminator = make_models(len(vocabulary), config, device)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=config["learning_rate_generator"], weight_decay=1e-4)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config["learning_rate_discriminator"], weight_decay=1e-4)
    dataset = GeneratorDataset(args.data_dir, "train", int(shared["training_sample_cap"]), int(project["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True)
    args.output_dir.mkdir(parents=True, exist_ok=True); latest = args.output_dir / "latest.pt"
    history_path = args.output_dir / "history.json"; history = load_json(history_path) if args.resume and history_path.exists() else []
    stage = "pretraining"; start_epoch = 1; best_score = -float("inf"); best_generator = None; best_discriminator = None
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"Cannot resume: {latest} does not exist")
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["master_sha256"] != project["data"]["master_sha256"]:
            raise RuntimeError("RelGAN checkpoint belongs to another master dataset")
        generator.load_state_dict(saved["generator_state"]); discriminator.load_state_dict(saved["discriminator_state"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        best_score = saved["best_selection_score"]; best_generator = saved["best_generator_state"]
        best_discriminator = saved["best_discriminator_state"]
        stage = saved["stage"]; start_epoch = int(saved["epoch"]) + 1
        if stage == "pretraining" and start_epoch > int(config["pretraining_epochs"]):
            stage = "adversarial"; start_epoch = 1

    hashes = protected_hashes(args.data_dir / "master.sqlite"); iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda")
    if stage == "pretraining":
        for epoch in range(start_epoch, int(config["pretraining_epochs"]) + 1):
            generator.train(); losses = []; started = time.perf_counter()
            for _ in range(int(config["pretraining_steps_per_epoch"])):
                real, iterator = next_batch(iterator, loader); real = real.to(device, non_blocking=True)
                generator_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16): loss = mle_loss(generator, real, bos_id)
                scaler.scale(loss).backward(); scaler.unscale_(generator_optimizer)
                nn.utils.clip_grad_norm_(generator.parameters(), 1.0); scaler.step(generator_optimizer); scaler.update()
                losses.append(float(loss.item()))
            quality, urls = quality_audit(generator, int(config["quality_sample_count"]), int(config["batch_size"]),
                                          maximum_length, bos_id, config["initial_temperature"], device, inverse, hashes)
            record = {"stage": "pretraining", "epoch": epoch, "mle_loss_mean": float(np.mean(losses)),
                      "seconds": time.perf_counter() - started, "quality": quality}
            history.append(record); write_json(history_path, history)
            write_json(args.output_dir / f"pretrain_epoch_{epoch:02d}_metrics.json", record)
            (args.output_dir / f"pretrain_epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            torch.save(checkpoint_payload(project, config, vocabulary, generator, discriminator,
                       generator_optimizer, discriminator_optimizer, "pretraining", epoch, best_score,
                       best_generator, best_discriminator), latest)
            print(json.dumps(record), flush=True)
        stage = "adversarial"; start_epoch = 1

    for epoch in range(start_epoch, int(config["adversarial_epochs"]) + 1):
        generator.train(); discriminator.train(); d_losses = []; g_losses = []; mle_losses = []
        adversarial_losses = []; started = time.perf_counter()
        for _ in range(int(config["adversarial_steps_per_epoch"])):
            real, iterator = next_batch(iterator, loader); real = real.to(device, non_blocking=True); batch_size = len(real)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): _, fake_ids = generator.sample(batch_size, maximum_length, bos_id,
                                                                 config["initial_temperature"], hard=True, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                real_scores, _ = discriminator(real); fake_scores, _ = discriminator(fake_ids)
                d_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) + \
                         F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            scaler.scale(d_loss).backward(); scaler.unscale_(discriminator_optimizer)
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0); scaler.step(discriminator_optimizer); scaler.update()
            d_losses.append(float(d_loss.item()))

            set_trainable(discriminator, False); generator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                language = mle_loss(generator, real, bos_id)
                soft, _ = generator.sample(batch_size, maximum_length, bos_id, config["initial_temperature"],
                                           hard=False, device=device)
                fake_scores, fake_features = discriminator.forward_soft(soft)
                with torch.no_grad(): _, real_features = discriminator(real)
                adversarial = F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
                feature = (fake_features.mean(0) - real_features.mean(0)).square().mean()
                marginal = soft.mean((0, 1)); diversity = (marginal * torch.log(marginal.clamp_min(1e-8))).sum()
                g_loss = adversarial + config["mle_regularization_weight"] * language + \
                         config["feature_matching_weight"] * feature + config["diversity_weight"] * diversity
            scaler.scale(g_loss).backward(); scaler.unscale_(generator_optimizer)
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0); scaler.step(generator_optimizer); scaler.update()
            set_trainable(discriminator, True); g_losses.append(float(g_loss.item()))
            mle_losses.append(float(language.item())); adversarial_losses.append(float(adversarial.item()))

        quality, urls = quality_audit(generator, int(config["quality_sample_count"]), int(config["batch_size"]),
                                      maximum_length, bos_id, config["initial_temperature"], device, inverse, hashes)
        selection = quality["validity"] * quality["uniqueness"] * (1 - quality["memorization_rate"]) * \
                    (1 - quality["repeated_template_rate"])
        if selection > best_score:
            best_score = selection; best_generator = deepcopy(generator.state_dict()); best_discriminator = deepcopy(discriminator.state_dict())
        record = {"stage": "adversarial", "epoch": epoch, "generator_loss_mean": float(np.mean(g_losses)),
                  "discriminator_loss_mean": float(np.mean(d_losses)), "mle_loss_mean": float(np.mean(mle_losses)),
                  "adversarial_loss_mean": float(np.mean(adversarial_losses)), "selection_score": selection,
                  "seconds": time.perf_counter() - started, "quality": quality}
        history.append(record); write_json(history_path, history)
        write_json(args.output_dir / f"epoch_{epoch:02d}_metrics.json", record)
        (args.output_dir / f"epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        torch.save(checkpoint_payload(project, config, vocabulary, generator, discriminator,
                   generator_optimizer, discriminator_optimizer, "adversarial", epoch, best_score,
                   best_generator, best_discriminator), latest)
        print(json.dumps(record), flush=True)

    if best_generator is None:
        raise RuntimeError("RelGAN adversarial training produced no checkpoint")
    torch.save({"project_version": 2, "model": "relgan", "generator_state": best_generator,
                "discriminator_state": best_discriminator, "relgan_config": config, "vocabulary": vocabulary,
                "selection_score": best_score, "master_sha256": project["data"]["master_sha256"]},
               args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
