import unittest

import torch

from phishing_gans_v3.models import HybridCharacterCNN, create_classifier


class ModelTests(unittest.TestCase):
    def test_hybrid_model_forward_and_backward(self) -> None:
        model = HybridCharacterCNN(64, 35)
        tokens = torch.randint(0, 64, (4, 128))
        lexical = torch.randn(4, 35)
        logits, representation = model(tokens, lexical, return_representation=True)
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(representation.shape), (4, 128))
        logits.mean().backward()
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_all_neural_models_forward_and_backward(self) -> None:
        configs = {
            "cnn": {"embedding_dim": 8, "channels_per_kernel": 4, "kernel_sizes": [3, 4, 5], "dropout": 0.1},
            "lstm": {"embedding_dim": 8, "hidden_dim": 10, "layers": 1, "dropout": 0.1},
            "rnn": {"embedding_dim": 8, "hidden_dim": 10, "layers": 1, "dropout": 0.1, "nonlinearity": "tanh"},
        }
        tokens = torch.randint(1, 20, (4, 32))
        tokens[:, -3:] = 0
        for name, config in configs.items():
            with self.subTest(model=name):
                model = create_classifier(name, 20, config)
                output = model(tokens)
                self.assertEqual(tuple(output.shape), (4,))
                output.sum().backward()
                self.assertIsNotNone(model.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
