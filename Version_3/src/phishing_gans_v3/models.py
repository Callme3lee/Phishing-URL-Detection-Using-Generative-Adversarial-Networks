from __future__ import annotations

import torch
from torch import nn


class CharacterCNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, channels_per_kernel: int,
                 kernel_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels_per_kernel, kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels_per_kernel * len(kernel_sizes), 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(layer(embedded)).amax(dim=2) for layer in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


class HybridCharacterCNN(nn.Module):
    """Character CNN fused with normalized PSL-aware lexical features."""

    def __init__(self, vocabulary_size: int, lexical_feature_count: int, embedding_dim: int = 48,
                 channels_per_kernel: int = 128, kernel_sizes: tuple[int, ...] = (3, 4, 5),
                 lexical_hidden: int = 64, lexical_embedding: int = 32,
                 fusion_hidden: int = 128, dropout: float = 0.4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels_per_kernel, kernel_size) for kernel_size in kernel_sizes
        )
        self.lexical_branch = nn.Sequential(
            nn.Linear(lexical_feature_count, lexical_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(lexical_hidden, lexical_embedding), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels_per_kernel * len(kernel_sizes) + lexical_embedding, fusion_hidden),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, 1)

    def forward(self, tokens: torch.Tensor, lexical: torch.Tensor,
                return_representation: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = torch.cat([torch.relu(layer(embedded)).amax(dim=2) for layer in self.convolutions], dim=1)
        lexical_embedding = self.lexical_branch(lexical)
        representation = self.fusion(torch.cat((pooled, lexical_embedding), dim=1))
        logits = self.classifier(representation).squeeze(1)
        return (logits, representation) if return_representation else logits


class CharacterRecurrentClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, cell: str, embedding_dim: int, hidden_dim: int,
                 layers: int, dropout: float, nonlinearity: str = "tanh") -> None:
        super().__init__()
        self.cell = cell
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        recurrent_dropout = dropout if layers > 1 else 0.0
        if cell == "lstm":
            self.recurrent: nn.Module = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
                dropout=recurrent_dropout,
            )
        elif cell == "rnn":
            self.recurrent = nn.RNN(
                embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
                dropout=recurrent_dropout, nonlinearity=nonlinearity,
            )
        else:
            raise ValueError(f"Unknown recurrent cell: {cell}")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output = self.recurrent(packed)
        hidden = output[1][0] if self.cell == "lstm" else output[1]
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


def create_classifier(name: str, vocabulary_size: int, config: dict) -> nn.Module:
    if name == "cnn":
        return CharacterCNN(
            vocabulary_size, int(config["embedding_dim"]), int(config["channels_per_kernel"]),
            tuple(config["kernel_sizes"]), float(config["dropout"]),
        )
    if name in {"lstm", "rnn"}:
        return CharacterRecurrentClassifier(
            vocabulary_size, name, int(config["embedding_dim"]), int(config["hidden_dim"]),
            int(config["layers"]), float(config["dropout"]), config.get("nonlinearity", "tanh"),
        )
    raise ValueError(f"Unknown neural classifier: {name}")
