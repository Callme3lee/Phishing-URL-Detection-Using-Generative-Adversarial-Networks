from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedURLDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, indices: np.ndarray | None = None) -> None:
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        actual = int(self.indices[index]) if self.indices is not None else index
        tokens = np.asarray(self.tokens[actual], dtype=np.int64)
        label = np.float32(self.labels[actual])
        return torch.from_numpy(tokens), torch.tensor(label)


class GeneratorDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, cap: int | None, seed: int) -> None:
        all_indices = np.load(data_dir / f"{split}_malicious_indices.npy", mmap_mode="r")
        if cap is not None and len(all_indices) > cap:
            generator = np.random.default_rng(seed)
            self.indices = np.sort(generator.choice(np.asarray(all_indices), size=cap, replace=False))
        else:
            self.indices = np.asarray(all_indices)
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.tokens[int(self.indices[index])], dtype=np.int64))


class AugmentedEncodedURLDataset(Dataset):
    """Unchanged real train rows followed by a prefix of one generated pool."""

    def __init__(
        self, data_dir: Path, generated_tokens: Path, generated_count: int,
        real_indices: np.ndarray | None = None, synthetic_weight: float = 1.0,
    ) -> None:
        self.real_tokens = np.load(data_dir / "train_tokens.npy", mmap_mode="r")
        self.real_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
        self.generated_tokens = np.load(generated_tokens, mmap_mode="r")
        self.generated_count = int(generated_count)
        self.real_indices = real_indices
        self.synthetic_weight = float(synthetic_weight)
        if self.generated_count > len(self.generated_tokens):
            raise ValueError("Augmentation condition exceeds its generated pool")

    def __len__(self) -> int:
        real_count = len(self.real_indices) if self.real_indices is not None else len(self.real_labels)
        return real_count + self.generated_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_count = len(self.real_indices) if self.real_indices is not None else len(self.real_labels)
        if index < real_count:
            actual = int(self.real_indices[index]) if self.real_indices is not None else index
            tokens = self.real_tokens[actual]; label = self.real_labels[actual]; weight = 1.0
        else:
            tokens = self.generated_tokens[index - real_count]; label = 1; weight = self.synthetic_weight
        return (torch.from_numpy(np.asarray(tokens, dtype=np.int64)), torch.tensor(np.float32(label)),
                torch.tensor(np.float32(weight)))
