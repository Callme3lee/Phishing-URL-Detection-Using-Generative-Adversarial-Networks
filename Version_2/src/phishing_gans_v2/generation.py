from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from .models import AutoregressiveGenerator
from .url_data import BOS, EOS, PAD, UNK, normalize_url


def teacher_forcing_inputs(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    bos = torch.full((len(tokens), 1), bos_id, device=tokens.device, dtype=tokens.dtype)
    return torch.cat((bos, tokens[:, :-1]), dim=1)


def sample_hard(
    generator: AutoregressiveGenerator,
    batch_size: int,
    maximum_length: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    temperature: float,
    device: torch.device,
    require_log_probabilities: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    current = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
    hidden = None; outputs = []; log_probabilities = []
    active = torch.ones(batch_size, device=device, dtype=torch.bool)
    for _ in range(maximum_length):
        logits, hidden = generator.step(current, hidden)
        distribution = torch.distributions.Categorical(logits=logits / temperature)
        sampled = distribution.sample()
        token = torch.where(active, sampled, torch.full_like(sampled, pad_id))
        outputs.append(token)
        if require_log_probabilities:
            log_probabilities.append(
                torch.where(active, distribution.log_prob(sampled), torch.zeros_like(sampled, dtype=logits.dtype))
            )
        active = active & token.ne(eos_id)
        current = token[:, None]
    tokens = torch.stack(outputs, dim=1)
    logs = torch.stack(log_probabilities, dim=1) if require_log_probabilities else None
    return tokens, logs


def sample_gumbel_soft(
    generator: AutoregressiveGenerator,
    batch_size: int,
    maximum_length: int,
    bos_id: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    current = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
    hidden = None; outputs = []
    for index in range(maximum_length):
        if index == 0:
            logits, hidden = generator.step(current, hidden)
        else:
            embedding = outputs[-1] @ generator.embedding.weight
            logits, hidden = generator.step(embedding[:, None, :], hidden, embedded=True)
        distribution = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        outputs.append(distribution)
    return torch.stack(outputs, dim=1)


@torch.inference_mode()
def complete_from_prefix(
    generator: AutoregressiveGenerator,
    prefix: torch.Tensor,
    maximum_length: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    temperature: float,
) -> torch.Tensor:
    """Complete fixed prefix tokens using the current stochastic generator."""
    batch_size, prefix_length = prefix.shape
    bos = torch.full(
        (batch_size, 1), bos_id, device=prefix.device, dtype=torch.long
    )
    inputs = torch.cat((bos, prefix), dim=1)
    logits, hidden = generator(inputs)
    output = [prefix]
    active = ~prefix.eq(eos_id).any(dim=1)
    next_logits = logits[:, -1]
    for _ in range(prefix_length, maximum_length):
        sampled = torch.distributions.Categorical(
            logits=next_logits / temperature
        ).sample()
        token = torch.where(active, sampled, torch.full_like(sampled, pad_id))
        output.append(token[:, None])
        active = active & token.ne(eos_id)
        next_logits, hidden = generator.step(token[:, None], hidden)
    return torch.cat(output, dim=1)[:, :maximum_length]


@torch.inference_mode()
def strided_rollout_rewards(
    generator: AutoregressiveGenerator,
    discriminator: torch.nn.Module,
    sampled_tokens: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    temperature: float,
    rollout_count: int,
    rollout_stride: int,
) -> torch.Tensor:
    """Approximate SeqGAN Monte-Carlo rewards at fixed prefix intervals.

    Full per-token rollouts at length 128 are prohibitively expensive on the
    target laptop GPU. Rewards are estimated every ``rollout_stride`` tokens
    and shared by actions in that segment. The final segment uses the completed
    sampled sequence directly.
    """
    batch_size, maximum_length = sampled_tokens.shape
    rewards = torch.zeros(
        batch_size, maximum_length, device=sampled_tokens.device
    )
    cutoffs = list(range(rollout_stride, maximum_length, rollout_stride))
    cutoffs.append(maximum_length)
    segment_start = 0
    for cutoff in cutoffs:
        if cutoff == maximum_length:
            score = torch.sigmoid(discriminator(sampled_tokens)[0])
        else:
            prefix = sampled_tokens[:, :cutoff]
            repeated = prefix.repeat_interleave(rollout_count, dim=0)
            completed = complete_from_prefix(
                generator, repeated, maximum_length, bos_id, eos_id, pad_id, temperature
            )
            score = torch.sigmoid(discriminator(completed)[0]).reshape(
                batch_size, rollout_count
            ).mean(dim=1)
        rewards[:, segment_start:cutoff] = score[:, None]
        segment_start = cutoff
    return rewards


def decode_tokens(tokens: np.ndarray, inverse_vocabulary: dict[int, str]) -> str:
    characters = []
    for value in tokens:
        token = inverse_vocabulary.get(int(value), UNK)
        if token == EOS:
            break
        if token in {PAD, BOS, UNK}:
            continue
        characters.append(token)
    return "".join(characters)


def basic_generation_metrics(urls: list[str], protected_hashes: set[bytes] | None = None) -> dict:
    total = len(urls); counts = Counter(urls); unique = len(counts)
    valid = 0; lengths = []; characters = Counter(); bigrams = Counter(); templates = Counter()
    memorized = 0
    for url in urls:
        lengths.append(len(url)); characters.update(url); bigrams.update(zip(url, url[1:]))
        templates[url[:24]] += 1
        try:
            normalize_url(url)
            valid += 1
        except ValueError:
            pass
        if protected_hashes is not None:
            digest = hashlib.blake2b(url.encode("utf-8"), digest_size=16).digest()
            memorized += int(digest in protected_hashes)
    repeated_templates = sum(count for count in templates.values() if count > 5)
    return {
        "samples": total,
        "unique_samples": unique,
        "uniqueness": unique / max(1, total),
        "valid_url_like": valid,
        "validity": valid / max(1, total),
        "memorized_exact": memorized,
        "memorization_rate": memorized / max(1, total),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
        "unique_characters": len(characters),
        "unique_bigrams": len(bigrams),
        "bigram_diversity": len(bigrams) / max(1, sum(bigrams.values())),
        "repeated_template_rate": repeated_templates / max(1, total),
    }
