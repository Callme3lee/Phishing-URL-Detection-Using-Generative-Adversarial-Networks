from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AutoregressiveGenerator(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, inputs: torch.Tensor, hidden=None) -> tuple[torch.Tensor, tuple]:
        embedded = self.embedding(inputs) if inputs.dtype in (torch.int32, torch.int64) else inputs
        output, hidden = self.lstm(embedded, hidden)
        return self.output(output), hidden

    def step(self, inputs: torch.Tensor, hidden=None, embedded: bool = False):
        values = inputs if embedded else self.embedding(inputs)
        output, hidden = self.lstm(values, hidden)
        return self.output(output[:, -1]), hidden


class GumbelLSTMGenerator(AutoregressiveGenerator):
    """Two-layer LSTM URL generator with differentiable Gumbel-Softmax sampling."""

    def __init__(self, vocabulary_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__(vocabulary_size, embedding_dim, hidden_dim, layers, dropout)

    def sample(self, batch: int, length: int, bos_id: int, eos_id: int, pad_id: int,
               temperature: float, device: torch.device,
               hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(torch.full((batch, 1), bos_id, dtype=torch.long, device=device))
        hidden = None
        distributions, ids = [], []
        active = torch.ones(batch, dtype=torch.bool, device=device)
        padding = F.one_hot(torch.full((batch,), pad_id, device=device), self.output.out_features).float()
        for _ in range(length):
            logits, hidden = self.step(embedded, hidden, embedded=True)
            proposed = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            distribution = torch.where(active[:, None], proposed, padding)
            token_ids = distribution.argmax(dim=-1)
            distributions.append(distribution)
            ids.append(token_ids)
            active = active & token_ids.ne(eos_id)
            embedded = (distribution @ self.embedding.weight).unsqueeze(1)
        return torch.stack(distributions, dim=1), torch.stack(ids, dim=1)

    @torch.inference_mode()
    def sample_ids(self, batch: int, length: int, bos_id: int, eos_id: int, pad_id: int,
                   temperature: float, device: torch.device) -> torch.Tensor:
        return self.sample(batch, length, bos_id, eos_id, pad_id, temperature, device, hard=True)[1]


class SequenceDiscriminator(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 64,
                 channels: int = 128, kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(nn.Conv1d(embedding_dim, channels, size) for size in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels * len(kernel_sizes), 1)

    def _forward_embedded(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = embedded.transpose(1, 2)
        features = self.dropout(torch.cat([
            torch.relu(layer(channels)).amax(dim=2) for layer in self.convolutions
        ], dim=1))
        return self.output(features).squeeze(1), features

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_embedded(self.embedding(tokens))

    def forward_soft(self, distributions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_embedded(distributions @ self.embedding.weight)


class RelationalMemoryCore(nn.Module):
    def __init__(self, input_dim: int, memory_slots: int, memory_size: int,
                 attention_heads: int) -> None:
        super().__init__()
        if memory_size % attention_heads:
            raise ValueError("memory_size must be divisible by attention_heads")
        self.memory_slots, self.memory_size = memory_slots, memory_size
        self.input_projection = nn.Linear(input_dim, memory_size)
        self.attention = nn.MultiheadAttention(memory_size, attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(memory_size)
        self.mlp = nn.Sequential(nn.Linear(memory_size, 2 * memory_size), nn.GELU(),
                                 nn.Linear(2 * memory_size, memory_size))
        self.mlp_norm = nn.LayerNorm(memory_size)
        self.gates = nn.Linear(2 * memory_size, 2 * memory_size)

    def forward(self, embedded: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(embedded).unsqueeze(1)
        context = torch.cat((memory, projected), dim=1)
        attended, _ = self.attention(memory, context, context, need_weights=False)
        candidate = self.attention_norm(memory + attended)
        candidate = self.mlp_norm(candidate + self.mlp(candidate))
        token_context = projected.expand(-1, self.memory_slots, -1)
        input_gate, forget_gate = self.gates(torch.cat((token_context, memory), dim=-1)).chunk(2, dim=-1)
        return torch.sigmoid(forget_gate) * memory + torch.sigmoid(input_gate) * torch.tanh(candidate)


class RelGANGenerator(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 64,
                 memory_slots: int = 4, memory_size: int = 128,
                 attention_heads: int = 4, latent_dim: int = 64) -> None:
        super().__init__()
        self.memory_slots, self.memory_size, self.latent_dim = memory_slots, memory_size, latent_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.initial_memory = nn.Parameter(torch.empty(memory_slots, memory_size))
        nn.init.xavier_uniform_(self.initial_memory)
        self.latent_projection = nn.Linear(latent_dim, memory_slots * memory_size)
        self.core = RelationalMemoryCore(embedding_dim, memory_slots, memory_size, attention_heads)
        self.output = nn.Linear(memory_slots * memory_size, vocabulary_size)

    def make_memory(self, batch: int, device: torch.device, latent=None) -> torch.Tensor:
        memory = self.initial_memory.unsqueeze(0).expand(batch, -1, -1)
        if latent is not None:
            memory = memory + self.latent_projection(latent).view(batch, self.memory_slots, self.memory_size)
        return memory.to(device)

    def step(self, embedded: torch.Tensor, memory: torch.Tensor):
        memory = self.core(embedded, memory)
        return self.output(memory.flatten(start_dim=1)), memory

    def forward(self, tokens: torch.Tensor, latent=None) -> torch.Tensor:
        memory = self.make_memory(len(tokens), tokens.device, latent)
        outputs = []
        for embedded in self.embedding(tokens).unbind(dim=1):
            logits, memory = self.step(embedded, memory); outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def sample(self, batch: int, length: int, bos_id: int, temperature: float,
               device: torch.device, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.make_memory(batch, device, torch.randn(batch, self.latent_dim, device=device))
        embedded = self.embedding(torch.full((batch,), bos_id, dtype=torch.long, device=device))
        distributions, ids = [], []
        for _ in range(length):
            logits, memory = self.step(embedded, memory)
            distribution = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            distributions.append(distribution); ids.append(distribution.argmax(dim=-1))
            embedded = distribution @ self.embedding.weight
        return torch.stack(distributions, dim=1), torch.stack(ids, dim=1)

    @torch.inference_mode()
    def sample_ids(self, batch: int, length: int, bos_id: int, temperature: float,
                   device: torch.device) -> torch.Tensor:
        return self.sample(batch, length, bos_id, temperature, device, hard=True)[1]


class RelGANDiscriminator(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 64,
                 representations: int = 3, channels: int = 128,
                 kernel_sizes: tuple[int, ...] = (2, 3, 4, 5), dropout: float = 0.3) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0) for _ in range(representations)
        ])
        self.convolutions = nn.ModuleList([
            nn.ModuleList([nn.Conv1d(embedding_dim, channels, size) for size in kernel_sizes])
            for _ in range(representations)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(representations * len(kernel_sizes) * channels, 1)

    def _features(self, views: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for embedded, bank in zip(views, self.convolutions):
            channels = embedded.transpose(1, 2)
            pooled.extend(torch.relu(layer(channels)).amax(dim=2) for layer in bank)
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, tokens: torch.Tensor):
        features = self._features([embedding(tokens) for embedding in self.embeddings])
        return self.output(features).squeeze(1), features

    def forward_soft(self, distributions: torch.Tensor):
        features = self._features([distributions @ embedding.weight for embedding in self.embeddings])
        return self.output(features).squeeze(1), features
