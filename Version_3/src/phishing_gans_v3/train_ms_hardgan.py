from __future__ import annotations

import argparse
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .gan_utils import basic_quality, decode_tokens, teacher_inputs
from .models import create_classifier
from .ms_hardgan_models import ConditionalURLGenerator, MultiHeadURLDiscriminator
from .url_data import BOS, EOS, PAD


class MSHardGANDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.tokens = np.load(root / "train_tokens.npy", mmap_mode="r")
        self.patterns = np.load(root / "train_patterns.npy", mmap_mode="r")
        self.sources = np.load(root / "train_sources.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(np.asarray(self.tokens[index], dtype=np.int64).copy()),
            int(self.patterns[index]), int(self.sources[index]),
        )


class FrozenChallengeEnsemble:
    """Rank-normalized challenge reward from frozen CNN, LSTM, and RNN families."""

    def __init__(self, project: dict, data_dir: Path, checkpoint_dir: Path,
                 model_names: list[str], seed: int, device: torch.device) -> None:
        validation = np.load(data_dir / "validation_tokens.npy", mmap_mode="r")
        validation_tensor = torch.from_numpy(np.asarray(validation, dtype=np.int64))
        self.entries: list[tuple[str, nn.Module, torch.Tensor, float]] = []
        for name in model_names:
            saved = torch.load(
                checkpoint_dir / name / f"seed_{seed}" / "checkpoint.pt",
                map_location=device, weights_only=False,
            )
            model = create_classifier(name, len(saved["vocabulary"]), saved["model_config"]).to(device)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            model.requires_grad_(False)
            reference = []
            with torch.inference_mode():
                for start in range(0, len(validation_tensor), 1024):
                    batch = validation_tensor[start:start + 1024].to(device)
                    with torch.autocast("cuda", dtype=torch.float16):
                        reference.append(torch.sigmoid(model(batch)).float())
            sorted_reference = torch.sort(torch.cat(reference))[0]
            self.entries.append((name, model, sorted_reference, float(saved["threshold"])))

    @torch.inference_mode()
    def reward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rewards, details = [], {}
        for name, model, reference, threshold in self.entries:
            with torch.autocast("cuda", dtype=torch.float16):
                scores = torch.sigmoid(model(tokens)).float()
            ranks = torch.searchsorted(reference, scores).float() / max(1, len(reference) - 1)
            missed = scores.lt(threshold).float()
            value = 0.5 * (1.0 - ranks) + 0.5 * missed
            rewards.append(value)
            details[f"{name}_mean_reward"] = float(value.mean().item())
            details[f"{name}_miss_rate"] = float(missed.mean().item())
        matrix = torch.stack(rewards)
        consensus = matrix.mean(0)
        details["ensemble_mean_reward"] = float(consensus.mean().item())
        return consensus, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defensive MS-HardGAN on CUDA.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments/ms_hardgan_v1"))
    parser.add_argument("--classifier-data-dir", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/experiments/multisource_baselines_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generators/ms_hardgan_v1"))
    parser.add_argument("--generator-config", default="ms_hardgan")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_trainable(module: nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


@torch.inference_mode()
def quality_sample(generator, count: int, config: dict, shared: dict, vocabulary: dict,
                   source_probabilities: torch.Tensor, device: torch.device,
                   protected: set[str]) -> tuple[dict, list[str]]:
    generator.eval()
    inverse = {int(value): token for token, value in vocabulary.items()}
    output = []
    while len(output) < count:
        batch = min(int(shared["batch_size"]), count - len(output))
        patterns = torch.randint(int(config["pattern_count"]), (batch,), device=device)
        sources = torch.multinomial(source_probabilities, batch, replacement=True)
        ids = generator.sample_ids(
            patterns, sources, int(shared["maximum_length"]), vocabulary[BOS], vocabulary[EOS],
            vocabulary[PAD], float(shared["temperature"]),
        )
        output.extend(decode_tokens(row, inverse) for row in ids.cpu().numpy())
    return basic_quality(output, protected), output


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    config = project["generators"][args.generator_config]
    shared = project["generators"]["shared"]
    preparation = load_json(args.data_dir / "preparation_report.json")
    device = require_cuda()
    seed = int(project["seed"])
    set_seed(seed)
    dataset = MSHardGANDataset(args.data_dir)
    if len(dataset) != int(config["training_rows"]) or len(dataset) != int(preparation["training_rows"]):
        raise RuntimeError("MS-HardGAN training row contract failed")
    vocabulary = load_json(args.data_dir / "vocabulary.json")

    counts = np.bincount(np.asarray(dataset.sources), minlength=int(config["source_count"]))
    largest = counts.max()
    source_weights = np.minimum(
        (largest / np.maximum(counts, 1)) ** float(config["source_sampling_exponent"]),
        float(config["maximum_source_weight"]),
    )
    pattern_counts = np.bincount(np.asarray(dataset.patterns), minlength=int(config["pattern_count"]))
    pattern_weights = np.minimum(
        (pattern_counts.max() / np.maximum(pattern_counts, 1)) ** float(config.get("pattern_sampling_exponent", 0.0)),
        float(config.get("maximum_pattern_weight", 1.0)),
    )
    sample_weights = torch.as_tensor(
        source_weights[np.asarray(dataset.sources)] * pattern_weights[np.asarray(dataset.patterns)],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    balanced_loader = DataLoader(
        dataset, batch_size=int(shared["batch_size"]), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    full_loader = DataLoader(
        dataset, batch_size=int(shared["batch_size"]), shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    source_probabilities = torch.as_tensor(counts / counts.sum(), dtype=torch.float32, device=device)
    generator = ConditionalURLGenerator(
        len(vocabulary), int(config["pattern_count"]), int(config["source_count"]),
        int(shared["embedding_dim"]), int(config["condition_dim"]), int(shared["hidden_dim"]),
        int(config["generator_layers"]), int(config["latent_dim"]), float(config["generator_dropout"]),
    ).to(device)
    discriminator = MultiHeadURLDiscriminator(
        len(vocabulary), int(config["pattern_count"]), int(config["source_count"]),
        int(shared["embedding_dim"]), int(config["discriminator_channels"]),
        tuple(config["discriminator_kernel_sizes"]), float(config["discriminator_dropout"]),
    ).to(device)
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["generator_learning_rate"]), weight_decay=1e-4)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(config["discriminator_learning_rate"]), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir / "latest.pt"
    history_path = args.output_dir / "history.json"
    history: list[dict] = []
    stage, start_epoch, best_score, best_g, best_d = "mle", 1, -float("inf"), None, None
    if args.resume:
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["preparation_sha256"] != sha256_file(args.data_dir / "preparation_report.json"):
            raise RuntimeError("Checkpoint belongs to another MS-HardGAN corpus")
        generator.load_state_dict(saved["generator_state"])
        discriminator.load_state_dict(saved["discriminator_state"])
        g_optimizer.load_state_dict(saved["generator_optimizer"])
        d_optimizer.load_state_dict(saved["discriminator_optimizer"])
        stage, start_epoch = saved["stage"], int(saved["epoch"]) + 1
        best_score, best_g, best_d = saved["best_score"], saved["best_generator"], saved["best_discriminator"]
        history = load_json(history_path)
        if stage == "mle" and start_epoch > int(config["mle_epochs"]):
            stage, start_epoch = "adversarial", 1

    protected = set((args.data_dir / "train_urls.txt").read_text(encoding="utf-8").splitlines())

    def save(stage_name: str, epoch: int) -> None:
        payload = {
            "project_version": 3, "condition": args.generator_config, "stage": stage_name, "epoch": epoch,
            "generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
            "generator_optimizer": g_optimizer.state_dict(), "discriminator_optimizer": d_optimizer.state_dict(),
            "best_score": best_score, "best_generator": best_g, "best_discriminator": best_d,
            "vocabulary": vocabulary, "config": config, "shared_config": shared,
            "training_rows": len(dataset), "source_counts": counts.tolist(),
            "preparation_sha256": sha256_file(args.data_dir / "preparation_report.json"),
            "master_sha256": project["data"]["master_sha256"],
        }
        torch.save(payload, latest)

    if stage == "mle":
        for epoch in range(start_epoch, int(config["mle_epochs"]) + 1):
            generator.train()
            losses = []
            started = time.perf_counter()
            for real, patterns, sources in full_loader:
                real = real.to(device, non_blocking=True)
                patterns = patterns.to(device, non_blocking=True)
                sources = sources.to(device, non_blocking=True)
                g_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = generator(teacher_inputs(real, vocabulary[BOS]), patterns, sources)
                    loss = F.cross_entropy(logits.flatten(0, 1), real.flatten(), ignore_index=vocabulary[PAD])
                scaler.scale(loss).backward()
                scaler.unscale_(g_optimizer)
                nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                scaler.step(g_optimizer)
                scaler.update()
                losses.append(float(loss.item()))
            quality, urls = quality_sample(
                generator, int(shared["quality_sample_count"]), config, shared, vocabulary,
                source_probabilities, device, protected,
            )
            record = {
                "stage": "mle", "epoch": epoch, "loss": float(np.mean(losses)),
                "seconds": time.perf_counter() - started, "batches": len(losses), "quality": quality,
            }
            history.append(record)
            write_json(history_path, history)
            write_json(args.output_dir / f"mle_epoch_{epoch:02d}.json", record)
            (args.output_dir / f"mle_epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            save("mle", epoch)
            print(json.dumps(record), flush=True)
        stage, start_epoch = "adversarial", 1

    challenge = FrozenChallengeEnsemble(
        project, args.classifier_data_dir, args.checkpoint_dir,
        list(config["challenge_models"]), int(config["challenge_seed"]), device,
    )
    iterator = iter(balanced_loader)
    for epoch in range(start_epoch, int(config["adversarial_epochs"]) + 1):
        generator.train(); discriminator.train()
        g_losses, d_losses, challenge_values = [], [], []
        challenge_details: dict[str, float] = {}
        started = time.perf_counter()
        for step in range(1, int(config["adversarial_steps_per_epoch"]) + 1):
            try:
                real, patterns, sources = next(iterator)
            except StopIteration:
                iterator = iter(balanced_loader); real, patterns, sources = next(iterator)
            real = real.to(device, non_blocking=True)
            patterns = patterns.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)

            with torch.no_grad():
                fake_soft, fake_ids = generator.sample_soft(
                    patterns, sources, int(shared["maximum_length"]), vocabulary[BOS], vocabulary[EOS],
                    vocabulary[PAD], float(shared["temperature"]), hard=True,
                )
            d_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                real_auth, real_pattern, real_source, _ = discriminator(real)
                fake_auth, fake_pattern, fake_source, _ = discriminator(fake_ids.clone())
                authenticity = F.binary_cross_entropy_with_logits(real_auth, torch.ones_like(real_auth)) + \
                    F.binary_cross_entropy_with_logits(fake_auth, torch.zeros_like(fake_auth))
                categories = (
                    F.cross_entropy(real_pattern, patterns) + F.cross_entropy(fake_pattern, patterns) +
                    F.cross_entropy(real_source, sources) + F.cross_entropy(fake_source, sources)
                ) / 4
                pattern_loss = 0.5 * (F.cross_entropy(real_pattern, patterns) + F.cross_entropy(fake_pattern, patterns))
                source_loss = 0.5 * (F.cross_entropy(real_source, sources) + F.cross_entropy(fake_source, sources))
                d_loss = authenticity + float(config.get("pattern_weight", config.get("category_weight", 0.25))) * pattern_loss + \
                    float(config.get("source_weight", config.get("category_weight", 0.25))) * source_loss
            scaler.scale(d_loss).backward()
            scaler.unscale_(d_optimizer)
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(d_optimizer); scaler.update()
            d_losses.append(float(d_loss.item()))

            set_trainable(discriminator, False)
            g_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                language_logits = generator(teacher_inputs(real, vocabulary[BOS]), patterns, sources)
                language = F.cross_entropy(language_logits.flatten(0, 1), real.flatten(), ignore_index=vocabulary[PAD])
                soft, _ = generator.sample_soft(
                    patterns, sources, int(shared["maximum_length"]), vocabulary[BOS], vocabulary[EOS],
                    vocabulary[PAD], float(shared["temperature"]), hard=False,
                )
                fake_auth, fake_pattern, fake_source, fake_features = discriminator.forward_soft(soft)
                with torch.no_grad():
                    _, _, _, real_features = discriminator(real)
                realism = F.binary_cross_entropy_with_logits(fake_auth, torch.ones_like(fake_auth))
                pattern_category = F.cross_entropy(fake_pattern, patterns)
                source_category = F.cross_entropy(fake_source, sources)
                feature = (fake_features.mean(0) - real_features.mean(0)).square().mean()
                diversity = -fake_features.float().std(dim=0).mean()
                g_loss = (
                    realism + float(config.get("pattern_weight", config.get("category_weight", 0.25))) * pattern_category +
                    float(config.get("source_weight", config.get("category_weight", 0.25))) * source_category +
                    float(config["mle_regularization_weight"]) * language +
                    float(config["feature_matching_weight"]) * feature +
                    float(config["diversity_weight"]) * diversity
                )

            if epoch >= int(config["challenge_start_epoch"]) and step % int(config["challenge_interval"]) == 0:
                policy_ids, log_probabilities = generator.sample_policy(
                    patterns, sources, int(shared["maximum_length"]), vocabulary[BOS], vocabulary[EOS],
                    vocabulary[PAD], float(shared["temperature"]),
                )
                rewards, challenge_details = challenge.reward(policy_ids)
                advantage = rewards - rewards.mean()
                lengths = policy_ids.ne(vocabulary[PAD]).sum(1).clamp_min(1)
                policy_loss = -(advantage.detach() * (log_probabilities.sum(1) / lengths)).mean()
                g_loss = g_loss + float(config["challenge_weight"]) * policy_loss
                challenge_values.append(float(rewards.mean().item()))

            scaler.scale(g_loss).backward()
            scaler.unscale_(g_optimizer)
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(g_optimizer); scaler.update()
            set_trainable(discriminator, True)
            g_losses.append(float(g_loss.item()))

        quality, urls = quality_sample(
            generator, int(shared["quality_sample_count"]), config, shared, vocabulary,
            source_probabilities, device, protected,
        )
        score = quality["validity"] + quality["uniqueness"] - quality["repeated_template_rate"] - quality["memorization_rate"]
        if score > best_score:
            best_score = score
            best_g = deepcopy(generator.state_dict())
            best_d = deepcopy(discriminator.state_dict())
        record = {
            "stage": "adversarial", "epoch": epoch,
            "generator_loss": float(np.mean(g_losses)), "discriminator_loss": float(np.mean(d_losses)),
            "challenge_reward": float(np.mean(challenge_values)) if challenge_values else None,
            "challenge_details_last_batch": challenge_details,
            "seconds": time.perf_counter() - started, "quality": quality, "selection_score": score,
        }
        history.append(record)
        write_json(history_path, history)
        write_json(args.output_dir / f"adversarial_epoch_{epoch:02d}.json", record)
        (args.output_dir / f"adversarial_epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        save("adversarial", epoch)
        print(json.dumps(record), flush=True)

    final = torch.load(latest, map_location="cpu", weights_only=False)
    final["generator_state"] = best_g
    final["discriminator_state"] = best_d
    final["best_score"] = best_score
    torch.save(final, args.output_dir / "best.pt")
    del challenge, generator, discriminator, balanced_loader, full_loader
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
