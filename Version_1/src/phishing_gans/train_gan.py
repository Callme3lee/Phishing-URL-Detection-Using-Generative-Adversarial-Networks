from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, seed_everything
from .data import MaliciousURLDataset
from .gan_models import URLDiscriminator, URLGenerator, initialize_gan_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a character-level URL GAN on GPU.")
    parser.add_argument("--gan", choices=("lsgan", "wgan-gp"), default="lsgan")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gradient-penalty", type=float, default=10.0)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--sample-count", type=int, default=1_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decode(tokens: np.ndarray, id_to_token: dict[int, str]) -> list[str]:
    urls = []
    for row in tokens:
        characters = []
        for value in row:
            value = int(value)
            if value == 0:
                break
            if value == 1:
                characters.append("�")
            else:
                characters.append(id_to_token.get(value, "�"))
        urls.append("".join(characters))
    return urls


@torch.inference_mode()
def generate_samples(
    generator: URLGenerator,
    fixed_noise: torch.Tensor,
    temperature: float,
) -> np.ndarray:
    generator.eval()
    logits = generator(fixed_noise)
    # Deterministic argmax makes epoch-to-epoch comparison reproducible.
    return logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)


def wasserstein_gradient_penalty(
    discriminator: URLDiscriminator,
    real_sequences: torch.Tensor,
    fake_sequences: torch.Tensor,
) -> torch.Tensor:
    batch_size = real_sequences.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_sequences.device)
    interpolated = (
        alpha * real_sequences + (1 - alpha) * fake_sequences
    ).requires_grad_(True)
    scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = gradients.flatten(1).norm(2, dim=1)
    return (gradient_norm - 1).square().mean()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.epochs <= 0:
        raise ValueError("--max-samples and --epochs must be positive")
    if args.output_dir is None:
        args.output_dir = Path(
            "artifacts/lsgan_pilot" if args.gan == "lsgan" else "artifacts/wgan_gp_pilot"
        )
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.gan == "lsgan" else 1e-4
    device = require_cuda()
    seed_everything(args.seed)
    metadata = load_json(args.data_dir / "metadata.json")
    vocab = metadata["vocab"]
    vocab_size = len(vocab)
    sequence_length = int(metadata["max_length"])
    dataset = MaliciousURLDataset(
        args.data_dir, max_samples=args.max_samples, seed=args.seed
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    generator_args = {
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "latent_dim": args.latent_dim,
    }
    discriminator_args = {"vocab_size": vocab_size}
    generator = URLGenerator(**generator_args).to(device)
    discriminator = URLDiscriminator(**discriminator_args).to(device)
    generator.apply(initialize_gan_weights)
    discriminator.apply(initialize_gan_weights)
    betas = (0.5, 0.999) if args.gan == "lsgan" else (0.0, 0.9)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=betas
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=betas
    )
    fixed_generator = torch.Generator().manual_seed(args.seed + 1)
    fixed_noise = torch.randn(
        args.sample_count, args.latent_dim, generator=fixed_generator
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_to_token = {value: key for key, value in vocab.items() if len(key) == 1}
    history = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        temperature = args.temperature_start * (
            args.temperature_end / args.temperature_start
        ) ** progress
        generator_loss_sum = 0.0
        discriminator_loss_sum = 0.0
        gradient_penalty_sum = 0.0
        batches = 0
        generator_batches = 0
        for batch_index, real_tokens in enumerate(data_loader, start=1):
            real_tokens = real_tokens.to(device, non_blocking=True)
            real_sequences = F.one_hot(real_tokens, num_classes=vocab_size).float()

            discriminator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(len(real_tokens), args.latent_dim, device=device)
            with torch.no_grad():
                fake_sequences = generator.sample_soft(noise, temperature, hard=True)
            real_scores = discriminator(real_sequences)
            fake_scores = discriminator(fake_sequences)
            if args.gan == "lsgan":
                gradient_penalty = torch.zeros((), device=device)
                discriminator_loss = 0.5 * (
                    (real_scores - 0.9).square().mean() + fake_scores.square().mean()
                )
            else:
                gradient_penalty = wasserstein_gradient_penalty(
                    discriminator, real_sequences, fake_sequences
                )
                discriminator_loss = (
                    fake_scores.mean()
                    - real_scores.mean()
                    + args.gradient_penalty * gradient_penalty
                )
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_optimizer.step()

            update_generator = args.gan == "lsgan" or batch_index % args.critic_steps == 0
            if update_generator:
                generator_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(len(real_tokens), args.latent_dim, device=device)
                fake_sequences = generator.sample_soft(noise, temperature, hard=True)
                generator_scores = discriminator(fake_sequences)
                if args.gan == "lsgan":
                    generator_loss = 0.5 * (generator_scores - 0.9).square().mean()
                else:
                    generator_loss = -generator_scores.mean()
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
                generator_optimizer.step()
                generator_loss_sum += generator_loss.item()
                generator_batches += 1

            discriminator_loss_sum += discriminator_loss.item()
            gradient_penalty_sum += gradient_penalty.item()
            batches += 1

        sample_tokens = generate_samples(generator, fixed_noise, temperature)
        sample_urls = decode(sample_tokens, id_to_token)
        np.save(args.output_dir / f"samples_epoch_{epoch:03d}.npy", sample_tokens)
        (args.output_dir / f"samples_epoch_{epoch:03d}.txt").write_text(
            "\n".join(sample_urls), encoding="utf-8"
        )
        record = {
            "epoch": epoch,
            "temperature": temperature,
            "generator_loss": generator_loss_sum / max(generator_batches, 1),
            "discriminator_loss": discriminator_loss_sum / max(batches, 1),
            "gradient_penalty": gradient_penalty_sum / max(batches, 1),
            "batches": batches,
            "generator_updates": generator_batches,
        }
        history.append(record)
        checkpoint = {
            "gan_type": args.gan,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "generator_args": generator_args,
            "discriminator_args": discriminator_args,
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "metadata": metadata,
            "epoch": epoch,
            "history": history,
            "training_args": vars(args),
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
        print("samples:", flush=True)
        for sample in sample_urls[:5]:
            print(f"  {sample}", flush=True)

    print(f"Saved {args.gan} checkpoint and samples to {args.output_dir}")


if __name__ == "__main__":
    main()
