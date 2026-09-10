from __future__ import annotations

import torch
from torch import nn


class CharacterCNN(nn.Module):
    """Compact URL classifier designed for a laptop GPU."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        channels: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=size, padding=size // 2)
            for size in (3, 5, 7)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(features)).amax(dim=2) for conv in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


class CharacterLSTM(nn.Module):
    """Single-layer character LSTM kept deliberately small and reproducible."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1)
        sequence, _ = self.lstm(self.embedding(tokens))
        last = sequence[
            torch.arange(tokens.size(0), device=tokens.device), lengths - 1
        ]
        return self.classifier(self.dropout(last)).squeeze(1)


def create_character_model(model_type: str, **model_args) -> nn.Module:
    if model_type == "cnn":
        return CharacterCNN(**model_args)
    if model_type == "lstm":
        return CharacterLSTM(**model_args)
    raise ValueError(f"Unknown character model: {model_type}")
