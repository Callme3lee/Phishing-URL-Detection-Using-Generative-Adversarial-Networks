"""RelGAN components staged for future experiments.

This module is deliberately not imported by the Version 2 training runners.
It provides a compact, URL-character implementation that can be activated only
after a separate experimental protocol is approved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RelationalMemoryCore(nn.Module):
    """A small recurrent relational-memory cell for character generation."""

    def __init__(self, input_dim: int, memory_slots: int, memory_size: int, attention_heads: int) -> None:
        super().__init__()
        if memory_size % attention_heads:
            raise ValueError("memory_size must be divisible by attention_heads")
        self.memory_slots = memory_slots
        self.memory_size = memory_size
        self.input_projection = nn.Linear(input_dim, memory_size)
        self.attention = nn.MultiheadAttention(memory_size, attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(memory_size)
        self.mlp = nn.Sequential(
            nn.Linear(memory_size, memory_size * 2), nn.GELU(), nn.Linear(memory_size * 2, memory_size)
        )
        self.mlp_norm = nn.LayerNorm(memory_size)
        self.gates = nn.Linear(memory_size * 2, memory_size * 2)

    def forward(self, embedded_token: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if embedded_token.ndim != 2:
            raise ValueError("embedded_token must have shape [batch, input_dim]")
        projected = self.input_projection(embedded_token).unsqueeze(1)
        attended, _ = self.attention(memory, torch.cat((memory, projected), dim=1),
                                     torch.cat((memory, projected), dim=1), need_weights=False)
        candidate = self.attention_norm(memory + attended)
        candidate = self.mlp_norm(candidate + self.mlp(candidate))
        token_context = projected.expand(-1, self.memory_slots, -1)
        input_gate, forget_gate = self.gates(torch.cat((token_context, memory), dim=-1)).chunk(2, dim=-1)
        return torch.sigmoid(forget_gate) * memory + torch.sigmoid(input_gate) * torch.tanh(candidate)


class RelGANGenerator(nn.Module):
    """Relational-memory generator with differentiable Gumbel-Softmax output."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        memory_slots: int = 4,
        memory_size: int = 128,
        attention_heads: int = 4,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.memory_slots = memory_slots
        self.memory_size = memory_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.initial_memory = nn.Parameter(torch.empty(memory_slots, memory_size))
        nn.init.xavier_uniform_(self.initial_memory)
        self.latent_projection = nn.Linear(latent_dim, memory_slots * memory_size)
        self.core = RelationalMemoryCore(embedding_dim, memory_slots, memory_size, attention_heads)
        self.output = nn.Linear(memory_slots * memory_size, vocabulary_size)

    def make_memory(
        self, batch_size: int, device: torch.device, latent: torch.Tensor | None = None
    ) -> torch.Tensor:
        memory = self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)
        if latent is not None:
            if latent.shape != (batch_size, self.latent_dim):
                raise ValueError(f"latent must have shape [{batch_size}, {self.latent_dim}]")
            memory = memory + self.latent_projection(latent).view(batch_size, self.memory_slots, self.memory_size)
        return memory.to(device)

    def step(self, embedded_token: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.core(embedded_token, memory)
        logits = self.output(memory.flatten(start_dim=1))
        return logits, memory

    def forward(self, tokens: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.make_memory(tokens.size(0), tokens.device, latent)
        outputs = []
        for embedded_token in self.embedding(tokens).unbind(dim=1):
            logits, memory = self.step(embedded_token, memory)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def sample(
        self,
        batch_size: int,
        maximum_length: int,
        bos_id: int,
        temperature: float = 0.8,
        latent: torch.Tensor | None = None,
        hard: bool = False,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = device or self.initial_memory.device
        memory = self.make_memory(batch_size, device, latent)
        embedded = self.embedding(torch.full((batch_size,), bos_id, dtype=torch.long, device=device))
        distributions, token_ids = [], []
        for _ in range(maximum_length):
            logits, memory = self.step(embedded, memory)
            distribution = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            distributions.append(distribution)
            token_ids.append(distribution.argmax(dim=-1))
            embedded = distribution @ self.embedding.weight
        return torch.stack(distributions, dim=1), torch.stack(token_ids, dim=1)

    def sample_ids(
        self,
        batch_size: int,
        maximum_length: int,
        bos_id: int,
        temperature: float = 0.8,
        latent: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Memory-efficient hard sampling for large, non-differentiable audits."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = device or self.initial_memory.device
        memory = self.make_memory(batch_size, device, latent)
        embedded = self.embedding(torch.full((batch_size,), bos_id, dtype=torch.long, device=device))
        token_ids = []
        for _ in range(maximum_length):
            logits, memory = self.step(embedded, memory)
            distribution = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            ids = distribution.argmax(dim=-1); token_ids.append(ids)
            embedded = distribution @ self.embedding.weight
        return torch.stack(token_ids, dim=1)


class RelGANDiscriminator(nn.Module):
    """CNN discriminator over multiple independently learned token embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        representations: int = 3,
        channels_per_kernel: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0) for _ in range(representations)
        )
        self.convolutions = nn.ModuleList(
            nn.ModuleList(nn.Conv1d(embedding_dim, channels_per_kernel, size) for size in kernel_sizes)
            for _ in range(representations)
        )
        feature_size = representations * len(kernel_sizes) * channels_per_kernel
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(feature_size, 1)

    def _features(self, embedded_views: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for embedded, convolution_bank in zip(embedded_views, self.convolutions):
            channels = embedded.transpose(1, 2)
            pooled.extend(torch.relu(layer(channels)).amax(dim=2) for layer in convolution_bank)
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features([embedding(tokens) for embedding in self.embeddings])
        return self.output(features).squeeze(1), features

    def forward_soft(self, distributions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if distributions.size(-1) != self.vocabulary_size:
            raise ValueError("last distribution dimension must equal vocabulary_size")
        features = self._features([distributions @ embedding.weight for embedding in self.embeddings])
        return self.output(features).squeeze(1), features


class RelGAN(nn.Module):
    """Container used only to construct the staged generator/discriminator pair."""

    def __init__(self, generator: RelGANGenerator, discriminator: RelGANDiscriminator) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
