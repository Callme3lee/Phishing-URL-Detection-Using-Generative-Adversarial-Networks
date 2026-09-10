from __future__ import annotations

import torch
from torch import nn


class CharacterCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 48,
        channels_per_kernel: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels_per_kernel, kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels_per_kernel * len(kernel_sizes), 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(layer(embedded)).amax(dim=2) for layer in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


class CharacterLSTM(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 48,
        hidden_dim: int = 96,
        layers: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


class HybridCharacterCNN(nn.Module):
    """Character CNN fused with inexpensive lexical/structural URL features."""

    def __init__(
        self,
        vocabulary_size: int,
        lexical_feature_count: int,
        embedding_dim: int = 48,
        channels_per_kernel: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        lexical_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels_per_kernel, kernel_size)
            for kernel_size in kernel_sizes
        )
        self.lexical_encoder = nn.Sequential(
            nn.Linear(lexical_feature_count, lexical_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        combined_size = channels_per_kernel * len(kernel_sizes) + lexical_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, lexical: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        character_features = torch.cat(
            [torch.relu(layer(embedded)).amax(dim=2) for layer in self.convolutions], dim=1
        )
        lexical_features = self.lexical_encoder(lexical)
        return self.fusion(torch.cat((character_features, lexical_features), dim=1)).squeeze(1)


def create_classifier(name: str, vocabulary_size: int, config: dict) -> nn.Module:
    if name == "cnn":
        return CharacterCNN(
            vocabulary_size=vocabulary_size,
            embedding_dim=config["embedding_dim"],
            channels_per_kernel=config["channels_per_kernel"],
            kernel_sizes=tuple(config["kernel_sizes"]),
            dropout=config["dropout"],
        )
    if name == "lstm":
        return CharacterLSTM(
            vocabulary_size=vocabulary_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            layers=config["layers"],
            dropout=config["dropout"],
        )
    raise ValueError(f"Unknown character classifier: {name}")


class AutoregressiveGenerator(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def forward(
        self, tokens: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        sequence, hidden = self.lstm(self.embedding(tokens), hidden)
        return self.output(sequence), hidden

    def step(
        self, token_or_embedding: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        embedded: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        value = token_or_embedding if embedded else self.embedding(token_or_embedding)
        sequence, hidden = self.lstm(value, hidden)
        return self.output(sequence[:, -1]), hidden


class SequenceDiscriminator(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        channels_per_kernel: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels_per_kernel, size) for size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels_per_kernel * len(kernel_sizes), 1)

    def features_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        values = embeddings.transpose(1, 2)
        pooled = [torch.relu(layer(values)).amax(dim=2) for layer in self.convolutions]
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features_from_embeddings(self.embedding(tokens))
        return self.output(features).squeeze(1), features

    def forward_soft(self, distributions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = distributions @ self.embedding.weight
        features = self.features_from_embeddings(embeddings)
        return self.output(features).squeeze(1), features
