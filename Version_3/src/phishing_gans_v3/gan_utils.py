from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import torch

from .gan_models import AutoregressiveGenerator
from .url_data import BOS, EOS, PAD, UNK, canonicalize_url


def teacher_inputs(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    bos = torch.full((len(tokens), 1), bos_id, dtype=tokens.dtype, device=tokens.device)
    return torch.cat((bos, tokens[:, :-1]), dim=1)


def decode_tokens(tokens: np.ndarray, inverse: dict[int, str]) -> str:
    output = []
    for value in tokens:
        token = inverse.get(int(value), UNK)
        if token == EOS:
            break
        if token not in {PAD, BOS, UNK}:
            output.append(token)
    return "".join(output)


def sample_hard(generator: AutoregressiveGenerator, batch: int, length: int,
                bos_id: int, eos_id: int, pad_id: int, temperature: float,
                device: torch.device, with_logs: bool = False):
    current = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    hidden = None; outputs = []; logs = []; active = torch.ones(batch, dtype=torch.bool, device=device)
    for _ in range(length):
        logits, hidden = generator.step(current, hidden)
        distribution = torch.distributions.Categorical(logits=logits / temperature)
        sampled = distribution.sample(); token = torch.where(active, sampled, torch.full_like(sampled, pad_id))
        outputs.append(token)
        if with_logs:
            logs.append(torch.where(active, distribution.log_prob(sampled), torch.zeros_like(logits[:, 0])))
        active = active & token.ne(eos_id); current = token[:, None]
    return torch.stack(outputs, dim=1), torch.stack(logs, dim=1) if with_logs else None


@torch.inference_mode()
def complete_prefix(generator, prefix, length, bos_id, eos_id, pad_id, temperature):
    batch, prefix_length = prefix.shape
    bos = torch.full((batch, 1), bos_id, dtype=torch.long, device=prefix.device)
    logits, hidden = generator(torch.cat((bos, prefix), dim=1))
    outputs = [prefix]; active = ~prefix.eq(eos_id).any(dim=1); next_logits = logits[:, -1]
    for _ in range(prefix_length, length):
        sampled = torch.distributions.Categorical(logits=next_logits / temperature).sample()
        token = torch.where(active, sampled, torch.full_like(sampled, pad_id))
        outputs.append(token[:, None]); active = active & token.ne(eos_id)
        next_logits, hidden = generator.step(token[:, None], hidden)
    return torch.cat(outputs, dim=1)[:, :length]


@torch.inference_mode()
def rollout_rewards(generator, discriminator, tokens, bos_id, eos_id, pad_id,
                    temperature, count, stride):
    batch, length = tokens.shape; rewards = torch.zeros(batch, length, device=tokens.device)
    cutoffs = list(range(stride, length, stride)) + [length]; start = 0
    for cutoff in cutoffs:
        if cutoff == length:
            values = torch.sigmoid(discriminator(tokens)[0])
        else:
            prefix = tokens[:, :cutoff].repeat_interleave(count, dim=0)
            completed = complete_prefix(generator, prefix, length, bos_id, eos_id, pad_id, temperature)
            values = torch.sigmoid(discriminator(completed)[0]).reshape(batch, count).mean(1)
        rewards[:, start:cutoff] = values[:, None]; start = cutoff
    return rewards


def basic_quality(urls: list[str], protected: set[str] | None = None) -> dict:
    counts = Counter(urls); valid = 0; lengths = []; bigrams = Counter(); templates = Counter(); memorized = 0
    for url in urls:
        lengths.append(len(url)); bigrams.update(zip(url, url[1:])); templates[url[:24]] += 1
        try:
            canonicalize_url(url); valid += 1
        except ValueError:
            pass
        memorized += int(protected is not None and url in protected)
    repeated = sum(value for value in templates.values() if value > 5)
    return {
        "samples": len(urls), "unique_samples": len(counts), "uniqueness": len(counts) / max(1, len(urls)),
        "valid_url_like": valid, "validity": valid / max(1, len(urls)),
        "memorized_exact": memorized, "memorization_rate": memorized / max(1, len(urls)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
        "unique_bigrams": len(bigrams), "bigram_diversity": len(bigrams) / max(1, sum(bigrams.values())),
        "repeated_template_rate": repeated / max(1, len(urls)),
    }
