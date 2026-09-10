from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedURLDataset(Dataset):
    def __init__(self, data_dir: Path, split: str) -> None:
        self.tokens = np.load(data_dir / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.from_numpy(np.asarray(self.tokens[index], dtype=np.int64))
        label = torch.tensor(np.float32(self.labels[index]))
        return tokens, label
