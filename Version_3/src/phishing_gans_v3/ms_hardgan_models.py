from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConditionalURLGenerator(nn.Module):
    """Two-layer character LSTM conditioned on phishing pattern and data source."""

    def __init__(self, vocabulary_size: int, pattern_count: int, source_count: int,
                 embedding_dim: int = 64, condition_dim: int = 24,
                 hidden_dim: int = 128, layers: int = 2, latent_dim: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.pattern_embedding = nn.Embedding(pattern_count, condition_dim)
        self.source_embedding = nn.Embedding(source_count, condition_dim)
        self.recurrent = nn.LSTM(
            embedding_dim + 2 * condition_dim, hidden_dim, layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        state_input = latent_dim + 2 * condition_dim
        self.initial_hidden = nn.Linear(state_input, layers * hidden_dim)
        self.initial_cell = nn.Linear(state_input, layers * hidden_dim)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def _condition(self, patterns: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.pattern_embedding(patterns), self.source_embedding(sources)), dim=1)

    def _state(self, condition: torch.Tensor, latent: torch.Tensor | None = None):
        if latent is None:
            latent = torch.zeros(len(condition), self.latent_dim, device=condition.device)
        values = torch.cat((condition, latent), dim=1)
        hidden = self.initial_hidden(values).view(len(values), self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        cell = self.initial_cell(values).view(len(values), self.layers, self.hidden_dim).transpose(0, 1).contiguous()
        return hidden, cell

    def forward(self, inputs: torch.Tensor, patterns: torch.Tensor,
                sources: torch.Tensor) -> torch.Tensor:
        condition = self._condition(patterns, sources)
        repeated = condition[:, None, :].expand(-1, inputs.shape[1], -1)
        embedded = torch.cat((self.embedding(inputs), repeated), dim=2)
        output, _ = self.recurrent(embedded, self._state(condition))
        return self.output(output)

    def _step(self, embedded: torch.Tensor, condition: torch.Tensor, state):
        values = torch.cat((embedded, condition), dim=1).unsqueeze(1)
        output, state = self.recurrent(values, state)
        return self.output(output[:, -1]), state

    def sample_soft(self, patterns: torch.Tensor, sources: torch.Tensor, length: int,
                    bos_id: int, eos_id: int, pad_id: int, temperature: float,
                    hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self._condition(patterns, sources)
        latent = torch.randn(len(patterns), self.latent_dim, device=patterns.device)
        state = self._state(condition, latent)
        embedded = self.embedding(torch.full_like(patterns, bos_id))
        active = torch.ones(len(patterns), dtype=torch.bool, device=patterns.device)
        padding = F.one_hot(torch.full_like(patterns, pad_id), self.vocabulary_size).float()
        distributions, ids = [], []
        for _ in range(length):
            logits, state = self._step(embedded, condition, state)
            proposed = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=1)
            distribution = torch.where(active[:, None], proposed, padding)
            token = distribution.argmax(dim=1)
            distributions.append(distribution)
            ids.append(token)
            active = active & token.ne(eos_id)
            embedded = distribution @ self.embedding.weight
        return torch.stack(distributions, dim=1), torch.stack(ids, dim=1)

    def sample_policy(self, patterns: torch.Tensor, sources: torch.Tensor, length: int,
                      bos_id: int, eos_id: int, pad_id: int,
                      temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard categorical samples and differentiable log-probabilities for REINFORCE."""
        condition = self._condition(patterns, sources)
        latent = torch.randn(len(patterns), self.latent_dim, device=patterns.device)
        state = self._state(condition, latent)
        embedded = self.embedding(torch.full_like(patterns, bos_id))
        active = torch.ones(len(patterns), dtype=torch.bool, device=patterns.device)
        ids, log_probabilities = [], []
        for _ in range(length):
            logits, state = self._step(embedded, condition, state)
            probabilities = torch.softmax(logits / temperature, dim=1)
            proposed = torch.multinomial(probabilities, 1).squeeze(1)
            token = torch.where(active, proposed, torch.full_like(proposed, pad_id))
            selected = torch.log(probabilities.gather(1, proposed[:, None]).squeeze(1).clamp_min(1e-8))
            selected = torch.where(active, selected, torch.zeros_like(selected))
            ids.append(token)
            log_probabilities.append(selected)
            active = active & token.ne(eos_id)
            embedded = self.embedding(token)
        return torch.stack(ids, dim=1), torch.stack(log_probabilities, dim=1)

    @torch.inference_mode()
    def sample_ids(self, patterns: torch.Tensor, sources: torch.Tensor, length: int,
                   bos_id: int, eos_id: int, pad_id: int,
                   temperature: float) -> torch.Tensor:
        return self.sample_soft(
            patterns, sources, length, bos_id, eos_id, pad_id, temperature, hard=True
        )[1]


class MultiHeadURLDiscriminator(nn.Module):
    """CNN discriminator for authenticity, structural family, and source category."""

    def __init__(self, vocabulary_size: int, pattern_count: int, source_count: int,
                 embedding_dim: int = 64, channels: int = 128,
                 kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel) for kernel in kernel_sizes
        )
        width = channels * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.authenticity = nn.Linear(width, 1)
        self.pattern = nn.Linear(width, pattern_count)
        self.source = nn.Linear(width, source_count)

    def _features(self, embedded: torch.Tensor) -> torch.Tensor:
        values = embedded.transpose(1, 2)
        pooled = [torch.relu(layer(values)).amax(dim=2) for layer in self.convolutions]
        return self.dropout(torch.cat(pooled, dim=1))

    def _heads(self, features: torch.Tensor):
        return (
            self.authenticity(features).squeeze(1),
            self.pattern(features), self.source(features), features,
        )

    def forward(self, tokens: torch.Tensor):
        return self._heads(self._features(self.embedding(tokens)))

    def forward_soft(self, distributions: torch.Tensor):
        return self._heads(self._features(distributions @ self.embedding.weight))
