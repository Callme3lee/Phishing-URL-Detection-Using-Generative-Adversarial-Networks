from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class URLGenerator(nn.Module):
    """Generate fixed-length character logits from Gaussian noise."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = 128,
        latent_dim: int = 128,
        base_channels: int = 256,
    ):
        super().__init__()
        if sequence_length % 8 != 0:
            raise ValueError("sequence_length must be divisible by 8")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.initial_length = sequence_length // 8
        self.base_channels = base_channels
        self.project = nn.Linear(latent_dim, base_channels * self.initial_length)
        self.blocks = nn.Sequential(
            _UpsampleBlock(base_channels, base_channels),
            _UpsampleBlock(base_channels, base_channels // 2),
            _UpsampleBlock(base_channels // 2, base_channels // 2),
        )
        self.output = nn.Conv1d(base_channels // 2, vocab_size, kernel_size=1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        features = self.project(noise).view(
            noise.size(0), self.base_channels, self.initial_length
        )
        return self.output(self.blocks(features)).transpose(1, 2)

    def sample_soft(self, noise: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
        logits = self(noise)
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


class _UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.convolution = nn.Conv1d(
            input_channels, output_channels, kernel_size=5, padding=2
        )
        self.normalization = nn.BatchNorm1d(output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, scale_factor=2, mode="nearest")
        return F.leaky_relu(self.normalization(self.convolution(values)), 0.2)


class URLDiscriminator(nn.Module):
    """Score real or generated one-hot URL sequences."""

    def __init__(self, vocab_size: int, base_channels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(vocab_size, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(base_channels * 4, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        features = self.features(sequences.transpose(1, 2)).squeeze(2)
        return self.output(features).squeeze(1)


def initialize_gan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
