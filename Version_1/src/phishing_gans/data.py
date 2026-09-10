from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedURLDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        if len(self.tokens) != len(self.labels):
            raise ValueError(f"Token and label counts differ for split {split}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # copy() avoids passing a read-only memory-mapped view to PyTorch.
        tokens = torch.from_numpy(self.tokens[index].copy()).long()
        label = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return tokens, label


class MaliciousURLDataset(Dataset):
    """View of only malicious training URLs, optionally reproducibly subsampled."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        indices = np.flatnonzero(labels == 1)
        if max_samples is not None and max_samples < len(indices):
            generator = np.random.default_rng(seed)
            indices = generator.choice(indices, size=max_samples, replace=False)
            indices.sort()
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.tokens[self.indices[index]].copy()
        return torch.from_numpy(row).long()


class AutoregressiveMaliciousDataset(MaliciousURLDataset):
    """Malicious URLs represented as BOS + characters + EOS + padding."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
        base_vocab_size: int = 115,
    ):
        super().__init__(data_dir, split=split, max_samples=max_samples, seed=seed)
        self.sequence_length = self.tokens.shape[1]
        self.bos_id = base_vocab_size
        self.eos_id = base_vocab_size + 1

    def full_sequence(self, index: int) -> torch.Tensor:
        source = self.tokens[self.indices[index]]
        characters = source[source != 0][: self.sequence_length - 2]
        sequence = np.zeros(self.sequence_length, dtype=np.int64)
        sequence[0] = self.bos_id
        sequence[1 : 1 + len(characters)] = characters
        sequence[1 + len(characters)] = self.eos_id
        return torch.from_numpy(sequence)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.full_sequence(index)
        return sequence[:-1], sequence[1:]


class AugmentedEncodedURLDataset(EncodedURLDataset):
    """Training dataset with equal-size real oversampling or synthetic positives."""

    def __init__(
        self,
        data_dir: Path,
        augmentation: str,
        augmentation_count: int,
        seed: int = 42,
        synthetic_tokens: Path | None = None,
        train_fraction: float = 1.0,
    ):
        super().__init__(data_dir, "train")
        if augmentation not in {"oversample", "synthetic"}:
            raise ValueError(f"Unsupported augmentation: {augmentation}")
        if augmentation_count <= 0:
            raise ValueError("augmentation_count must be positive")
        generator = np.random.default_rng(seed)
        if not 0 < train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        if train_fraction < 1:
            selected_parts = []
            for label in (0, 1):
                candidates = np.flatnonzero(self.labels == label)
                count = max(1, int(len(candidates) * train_fraction))
                selected_parts.append(generator.choice(candidates, size=count, replace=False))
            self.base_indices = np.sort(np.concatenate(selected_parts))
        else:
            self.base_indices = np.arange(len(self.labels))
        self.augmentation = augmentation
        if augmentation == "oversample":
            malicious_indices = self.base_indices[self.labels[self.base_indices] == 1]
            self.extra_indices = generator.choice(
                malicious_indices, size=augmentation_count, replace=True
            )
            self.synthetic = None
        else:
            if synthetic_tokens is None:
                raise ValueError("synthetic_tokens is required for synthetic augmentation")
            available = np.load(synthetic_tokens, mmap_mode="r")
            if augmentation_count > len(available):
                raise ValueError(
                    f"Requested {augmentation_count} synthetic rows, only {len(available)} available"
                )
            selected = generator.choice(
                len(available), size=augmentation_count, replace=False
            )
            self.synthetic = available
            self.extra_indices = selected

    def __len__(self) -> int:
        return len(self.base_indices) + len(self.extra_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_length = len(self.base_indices)
        if index < base_length:
            return super().__getitem__(int(self.base_indices[index]))
        extra_index = self.extra_indices[index - base_length]
        if self.augmentation == "oversample":
            row = self.tokens[extra_index].copy()
        else:
            row = self.synthetic[extra_index].copy()
        return torch.from_numpy(row).long(), torch.tensor(1.0, dtype=torch.float32)


class FractionalEncodedURLDataset(EncodedURLDataset):
    """Reproducible class-stratified fraction of the real training split."""

    def __init__(self, data_dir: Path, fraction: float, seed: int = 42):
        super().__init__(data_dir, "train")
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        generator = np.random.default_rng(seed)
        selected = []
        for label in (0, 1):
            candidates = np.flatnonzero(self.labels == label)
            count = max(1, int(len(candidates) * fraction))
            selected.append(generator.choice(candidates, size=count, replace=False))
        self.indices = np.sort(np.concatenate(selected))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(int(self.indices[index]))
