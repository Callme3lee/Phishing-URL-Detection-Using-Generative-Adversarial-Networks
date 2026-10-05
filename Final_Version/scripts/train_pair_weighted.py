#!/usr/bin/env python3
"""Run the unchanged V3 HYB-MS trainer with pair-mass-preserving sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Version_3/src"))

import phishing_gans_v3.train_fair_hybrid_gan as trainer  # noqa: E402
from phishing_gans_v3.train_hybrid_domain import HybridDataset, TARGET_GROUP_PROPORTIONS  # noqa: E402


def pair_preserving_loaders(data_dir: Path, variant: str, seed: int, batch_size: int, workers: int,
                            generated_fraction: float):
    train = HybridDataset(data_dir, "train", groups=True)
    groups = np.load(data_dir / "train_groups.npy", mmap_mode="r")
    factors = np.load(data_dir / "train_sample_factors.npy", mmap_mode="r")
    if len(factors) != len(groups):
        raise RuntimeError("Sample factors and groups are not aligned")
    target = {group: proportion * (1.0 - generated_fraction)
              for group, proportion in TARGET_GROUP_PROPORTIONS.items()}
    if generated_fraction:
        target[5] = generated_fraction
    effective_counts = {group: max(1e-12, float(factors[groups == group].sum())) for group in target}
    weights = np.asarray([
        float(factors[index]) * target[int(group)] / effective_counts[int(group)]
        for index, group in enumerate(groups)
    ], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    loaders = {"train": train_loader}
    for split in ("validation", "domain_validation", "test"):
        loaders[split] = DataLoader(HybridDataset(data_dir, split), batch_size=batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return loaders, target


def main() -> None:
    trainer.make_loaders = pair_preserving_loaders
    trainer.main()


if __name__ == "__main__":
    main()
