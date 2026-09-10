import unittest

import torch

from phishing_gans_v3.gan_models import (AutoregressiveGenerator, RelGANDiscriminator,
                                         RelGANGenerator, SequenceDiscriminator)
from phishing_gans_v3.gan_utils import teacher_inputs
from phishing_gans_v3.ms_hardgan_models import (ConditionalURLGenerator,
                                                MultiHeadURLDiscriminator)


class GANTests(unittest.TestCase):
    def test_seqgan_generator_and_discriminator_gradients(self) -> None:
        generator = AutoregressiveGenerator(32, 8, 12)
        discriminator = SequenceDiscriminator(32, 8, 4, (2, 3))
        tokens = torch.randint(1, 32, (4, 16))
        logits, _ = generator(teacher_inputs(tokens, 2))
        score, features = discriminator(tokens)
        self.assertEqual(tuple(logits.shape), (4, 16, 32))
        self.assertEqual(tuple(score.shape), (4,))
        self.assertEqual(tuple(features.shape), (4, 8))
        (logits.mean() + score.mean()).backward()
        self.assertIsNotNone(generator.embedding.weight.grad)
        self.assertIsNotNone(discriminator.embedding.weight.grad)

    def test_relgan_soft_path_gradients(self) -> None:
        generator = RelGANGenerator(32, 8, 2, 16, 4, 8)
        discriminator = RelGANDiscriminator(32, 8, 2, 4, (2, 3))
        soft, ids = generator.sample(4, 16, 2, 0.8, torch.device("cpu"))
        score, features = discriminator.forward_soft(soft)
        self.assertEqual(tuple(ids.shape), (4, 16))
        self.assertEqual(tuple(score.shape), (4,))
        self.assertEqual(tuple(features.shape), (4, 16))
        score.mean().backward()
        self.assertIsNotNone(generator.initial_memory.grad)
        self.assertIsNotNone(discriminator.embeddings[0].weight.grad)

    def test_relgan_inference_ids_can_train_discriminator(self) -> None:
        generator = RelGANGenerator(32, 8, 2, 16, 4, 8)
        discriminator = RelGANDiscriminator(32, 8, 2, 4, (2, 3))
        fake = generator.sample_ids(4, 16, 2, 0.8, torch.device("cpu")).clone()
        score, _ = discriminator(fake)
        score.mean().backward()
        self.assertIsNotNone(discriminator.embeddings[0].weight.grad)

    def test_ms_hardgan_conditioned_soft_path_and_heads(self) -> None:
        generator = ConditionalURLGenerator(32, 5, 3, 8, 4, 12, 2, 6, 0.1)
        discriminator = MultiHeadURLDiscriminator(32, 5, 3, 8, 4, (2, 3), 0.1)
        patterns = torch.tensor([0, 1, 2, 3])
        sources = torch.tensor([0, 1, 2, 0])
        soft, ids = generator.sample_soft(patterns, sources, 16, 2, 3, 0, 0.8)
        authenticity, pattern_logits, source_logits, features = discriminator.forward_soft(soft)
        self.assertEqual(tuple(ids.shape), (4, 16))
        self.assertEqual(tuple(authenticity.shape), (4,))
        self.assertEqual(tuple(pattern_logits.shape), (4, 5))
        self.assertEqual(tuple(source_logits.shape), (4, 3))
        self.assertEqual(tuple(features.shape), (4, 8))
        (authenticity.mean() + pattern_logits.mean() + source_logits.mean()).backward()
        self.assertIsNotNone(generator.pattern_embedding.weight.grad)
        self.assertIsNotNone(discriminator.embedding.weight.grad)

    def test_ms_hardgan_policy_sampling_has_generator_gradient(self) -> None:
        generator = ConditionalURLGenerator(24, 4, 3, 8, 4, 12, 2, 6, 0.1)
        patterns = torch.tensor([0, 1, 2, 3])
        sources = torch.tensor([0, 1, 2, 0])
        ids, logs = generator.sample_policy(patterns, sources, 12, 2, 3, 0, 0.9)
        self.assertEqual(tuple(ids.shape), (4, 12))
        self.assertEqual(tuple(logs.shape), (4, 12))
        logs.mean().backward()
        self.assertIsNotNone(generator.source_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
