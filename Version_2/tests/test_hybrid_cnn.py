import unittest

import torch

from phishing_gans_v2.models import HybridCharacterCNN


class HybridCNNTests(unittest.TestCase):
    def test_forward_shape_and_both_branches_receive_gradients(self) -> None:
        model = HybridCharacterCNN(vocabulary_size=50, lexical_feature_count=28)
        tokens = torch.randint(0, 50, (4, 128))
        lexical = torch.randn(4, 28)
        output = model(tokens, lexical)
        self.assertEqual(tuple(output.shape), (4,))
        output.sum().backward()
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertIsNotNone(model.lexical_encoder[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
