from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .autoregressive_models import (
    AutoregressiveURLGenerator,
    SpectralURLCritic,
    to_base_tokens,
)
from .common import load_json, require_cuda, seed_everything
from .data import AutoregressiveMaliciousDataset
from .train_gan import decode, wasserstein_gradient_penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarially fine-tune pretrained URL LSTM.")
    parser.add_argument("--gan", choices=("lsgan", "wgan-gp"), required=True)
    parser.add_argument("--pretrained", type=Path, default=Path("artifacts/generator_pretrained/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-5)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--gradient-penalty", type=float, default=10.0)
    parser.add_argument("--language-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--temperature-start", type=float, default=1.5)
    parser.add_argument("--temperature-end", type=float, default=0.8)
    parser.add_argument("--sample-count", type=int, default=1_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def diversity_loss(fake: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    half = len(fake) // 2
    if half == 0:
        return fake.new_zeros(())
    output_distance = (fake[:half] - fake[half : 2 * half]).abs().mean()
    noise_distance = (noise[:half] - noise[half : 2 * half]).abs().mean()
    return -output_distance / (noise_distance + 1e-6)


@torch.inference_mode()
def save_samples(generator, args, checkpoint, epoch, device) -> None:
    generator.eval()
    metadata = checkpoint["metadata"]
    noise = torch.randn(args.sample_count, generator.latent_dim, device=device)
    generated = generator.generate(
        noise,
        bos_id=checkpoint["bos_id"],
        sequence_length=metadata["max_length"],
        temperature=0.8,
        differentiable=False,
    )
    base_tokens = to_base_tokens(
        generated,
        base_vocab_size=len(metadata["vocab"]),
        bos_id=checkpoint["bos_id"],
        eos_id=checkpoint["eos_id"],
    ).cpu().numpy()
    inverse = {value: key for key, value in metadata["vocab"].items() if len(key) == 1}
    urls = decode(base_tokens, inverse)
    np.save(args.output_dir / f"samples_epoch_{epoch:03d}.npy", base_tokens)
    (args.output_dir / f"samples_epoch_{epoch:03d}.txt").write_text(
        "\n".join(urls), encoding="utf-8"
    )
    print("samples:", flush=True)
    for url in urls[:5]:
        print(f"  {url}", flush=True)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/autoregressive_{args.gan.replace('-', '_')}")
    device = require_cuda()
    seed_everything(args.seed)
    pretrained = torch.load(args.pretrained, map_location=device, weights_only=False)
    metadata = pretrained["metadata"]
    base_vocab_size = len(metadata["vocab"])
    extended_vocab_size = base_vocab_size + 2
    dataset = AutoregressiveMaliciousDataset(
        args.data_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        base_vocab_size=base_vocab_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    generator = AutoregressiveURLGenerator(**pretrained["model_args"]).to(device)
    generator.load_state_dict(pretrained["model_state"])
    critic = SpectralURLCritic(extended_vocab_size).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.generator_learning_rate, betas=(0.5, 0.999)
    )
    critic_betas = (0.5, 0.999) if args.gan == "lsgan" else (0.0, 0.9)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_learning_rate, betas=critic_betas
    )
    language_criterion = nn.CrossEntropyLoss(ignore_index=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        critic.train()
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        temperature = args.temperature_start * (
            args.temperature_end / args.temperature_start
        ) ** progress
        sums = {name: 0.0 for name in ("generator", "critic", "language", "feature", "diversity", "gp")}
        critic_batches = generator_batches = 0

        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            real_tokens = torch.cat((inputs[:, :1], targets), dim=1)
            real = F.one_hot(real_tokens, num_classes=extended_vocab_size).float()

            set_trainable(critic, True)
            critic_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(len(inputs), generator.latent_dim, device=device)
            with torch.no_grad():
                fake = generator.generate(
                    noise,
                    pretrained["bos_id"],
                    metadata["max_length"],
                    temperature=temperature,
                    differentiable=True,
                    hard=False,
                )
            real_scores = critic(real)
            fake_scores = critic(fake)
            if args.gan == "lsgan":
                gp = torch.zeros((), device=device)
                critic_loss = 0.5 * (
                    (real_scores - 0.9).square().mean() + fake_scores.square().mean()
                )
            else:
                gp = wasserstein_gradient_penalty(critic, real, fake)
                critic_loss = fake_scores.mean() - real_scores.mean() + args.gradient_penalty * gp
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_optimizer.step()
            sums["critic"] += critic_loss.item()
            sums["gp"] += gp.item()
            critic_batches += 1

            update_generator = args.gan == "lsgan" or batch_index % args.critic_steps == 0
            if not update_generator:
                continue
            set_trainable(critic, False)
            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(len(inputs), generator.latent_dim, device=device)
            fake = generator.generate(
                noise,
                pretrained["bos_id"],
                metadata["max_length"],
                temperature=temperature,
                differentiable=True,
                hard=False,
            )
            fake_scores = critic(fake)
            adversarial = (
                0.5 * (fake_scores - 0.9).square().mean()
                if args.gan == "lsgan"
                else -fake_scores.mean()
            )
            language_logits = generator.teacher_logits(inputs)
            language = language_criterion(
                language_logits.flatten(0, 1), targets.flatten()
            )
            with torch.no_grad():
                real_features = critic.extract_features(real).mean(dim=0)
            fake_features = critic.extract_features(fake).mean(dim=0)
            feature = F.l1_loss(fake_features, real_features)
            diversity = diversity_loss(fake, noise)
            generator_loss = (
                adversarial
                + args.language_weight * language
                + args.feature_weight * feature
                + args.diversity_weight * diversity
            )
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            generator_optimizer.step()
            sums["generator"] += generator_loss.item()
            sums["language"] += language.item()
            sums["feature"] += feature.item()
            sums["diversity"] += diversity.item()
            generator_batches += 1

        record = {
            "epoch": epoch,
            "temperature": temperature,
            "generator_loss": sums["generator"] / max(generator_batches, 1),
            "critic_loss": sums["critic"] / max(critic_batches, 1),
            "language_loss": sums["language"] / max(generator_batches, 1),
            "feature_loss": sums["feature"] / max(generator_batches, 1),
            "diversity_loss": sums["diversity"] / max(generator_batches, 1),
            "gradient_penalty": sums["gp"] / max(critic_batches, 1),
            "generator_updates": generator_batches,
            "critic_updates": critic_batches,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        save_samples(generator, args, pretrained, epoch, device)
        saved_checkpoint = {
                "gan_type": args.gan,
                "generator_state": generator.state_dict(),
                "critic_state": critic.state_dict(),
                "model_args": pretrained["model_args"],
                "metadata": metadata,
                "bos_id": pretrained["bos_id"],
                "eos_id": pretrained["eos_id"],
                "epoch": epoch,
                "history": history,
                "training_args": vars(args),
            }
        torch.save(saved_checkpoint, args.output_dir / "latest.pt")
        torch.save(
            saved_checkpoint, args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print(f"Saved stabilized {args.gan} run to {args.output_dir}")


if __name__ == "__main__":
    main()
