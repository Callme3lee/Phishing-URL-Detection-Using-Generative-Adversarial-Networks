from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AutoregressiveURLGenerator(nn.Module):
    """Character LSTM usable for teacher forcing and differentiable generation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.noise_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.noise_to_cell = nn.Linear(latent_dim, hidden_size)
        nn.init.normal_(self.noise_to_hidden.weight, std=0.005)
        nn.init.normal_(self.noise_to_cell.weight, std=0.005)
        nn.init.zeros_(self.noise_to_hidden.bias)
        nn.init.zeros_(self.noise_to_cell.bias)

    def teacher_logits(self, input_tokens: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self.embedding(input_tokens))
        return self.output(sequence)

    def generate(
        self,
        noise: torch.Tensor,
        bos_id: int,
        sequence_length: int,
        temperature: float = 1.0,
        differentiable: bool = False,
        hard: bool = False,
    ) -> torch.Tensor:
        batch_size = noise.size(0)
        hidden = torch.tanh(self.noise_to_hidden(noise)).unsqueeze(0)
        cell = torch.tanh(self.noise_to_cell(noise)).unsqueeze(0)
        current = F.one_hot(
            torch.full((batch_size,), bos_id, device=noise.device),
            num_classes=self.vocab_size,
        ).float()
        generated = [current]
        for _ in range(sequence_length - 1):
            embedded = current @ self.embedding.weight
            output, (hidden, cell) = self.lstm(
                embedded.unsqueeze(1), (hidden, cell)
            )
            logits = self.output(output[:, 0])
            if differentiable:
                current = F.gumbel_softmax(
                    logits, tau=temperature, hard=hard, dim=-1
                )
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                current = F.one_hot(token, num_classes=self.vocab_size).float()
            generated.append(current)
        return torch.stack(generated, dim=1)


class SpectralURLCritic(nn.Module):
    """Spectrally normalized CNN critic for complete URL sequences."""

    def __init__(self, vocab_size: int, base_channels: int = 128):
        super().__init__()
        spectral = nn.utils.spectral_norm
        self.features = nn.Sequential(
            spectral(nn.Conv1d(vocab_size, base_channels, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            spectral(nn.Conv1d(base_channels, base_channels * 2, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            spectral(nn.Conv1d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = spectral(nn.Linear(base_channels * 4, 1))

    def extract_features(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.features(sequences.transpose(1, 2)).squeeze(2)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(sequences)
        return self.output(features).squeeze(1)


def to_base_tokens(
    generated: torch.Tensor,
    base_vocab_size: int,
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """Convert extended one-hot sequences into original classifier token arrays."""
    extended = generated.argmax(dim=-1)
    output = torch.zeros_like(extended, dtype=torch.uint8)
    for row_index, row in enumerate(extended):
        position = 0
        for value in row.tolist():
            if value == bos_id:
                continue
            if value in (0, eos_id):
                break
            if position >= output.size(1):
                break
            output[row_index, position] = value if value < base_vocab_size else 1
            position += 1
    return output
