from __future__ import annotations

import unittest

import torch

from phishing_gans_v2.relgan import RelGANDiscriminator, RelGANGenerator


class RelGANShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary_size = 37
        self.generator = RelGANGenerator(
            self.vocabulary_size, embedding_dim=16, memory_slots=2, memory_size=24,
            attention_heads=4, latent_dim=8,
        )
        self.discriminator = RelGANDiscriminator(
            self.vocabulary_size, embedding_dim=12, representations=2,
            channels_per_kernel=8, kernel_sizes=(2, 3), dropout=0.0,
        )

    def test_teacher_forcing_shapes(self) -> None:
        tokens = torch.randint(0, self.vocabulary_size, (3, 12))
        logits = self.generator(tokens)
        self.assertEqual(logits.shape, (3, 12, self.vocabulary_size))

    def test_soft_samples_reach_discriminator(self) -> None:
        distributions, token_ids = self.generator.sample(3, 12, bos_id=1, temperature=0.8)
        scores, features = self.discriminator.forward_soft(distributions)
        self.assertEqual(token_ids.shape, (3, 12))
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(features.ndim, 2)
        scores.mean().backward()
        self.assertIsNotNone(self.generator.output.weight.grad)

    def test_discrete_discriminator_path(self) -> None:
        scores, _ = self.discriminator(torch.randint(0, self.vocabulary_size, (3, 12)))
        self.assertEqual(scores.shape, (3,))

    def test_memory_efficient_ids(self) -> None:
        ids = self.generator.sample_ids(3, 12, bos_id=1, temperature=0.8)
        self.assertEqual(ids.shape, (3, 12))
        self.assertEqual(ids.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
