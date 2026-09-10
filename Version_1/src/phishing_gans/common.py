from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver with `nvidia-smi` and confirm that /dev/nvidia* "
            "devices are available, then rerun this command."
        )
    return torch.device("cuda")


def hostname_group(url: str) -> str:
    """Return a stable host group so one hostname cannot cross data splits."""
    candidate = url.strip()
    parsed = urlsplit(candidate if "://" in candidate else f"http://{candidate}")
    host = (parsed.hostname or "").lower().strip(".")
    if host:
        return host
    # Malformed URLs still receive a stable group without being discarded.
    return candidate.lower()[:80]


def split_name(group: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    payload = f"{seed}:{group}".encode("utf-8", errors="replace")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def build_vocab(characters: set[str]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for char in sorted(characters):
        vocab[char] = len(vocab)
    if len(vocab) > 256:
        raise ValueError(
            f"Vocabulary has {len(vocab)} entries; uint8 storage supports at most 256."
        )
    return vocab


def encode_url(url: str, vocab: dict[str, int], max_length: int) -> np.ndarray:
    encoded = np.zeros(max_length, dtype=np.uint8)
    unk = vocab[UNK_TOKEN]
    values = [vocab.get(char, unk) for char in url[:max_length]]
    encoded[: len(values)] = values
    return encoded


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)
