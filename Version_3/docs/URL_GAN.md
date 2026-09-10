# URL-GAN: Two-Layer Gumbel URL Generator

## Names used in the presentation

| Short name | Meaning |
|---|---|
| URL-GAN | The complete new phishing-URL GAN |
| 2L-LSTM-G | Two-layer LSTM generator |
| Char-CNN-D | Character-CNN real/fake discriminator |
| GS | Gumbel-Softmax relaxation |
| SG | Original compact SeqGAN reference |
| HYB | Calibrated Hybrid CNN + lexical + CORAL classifier ensemble |

These names describe roles without claiming that URL-GAN is an exact reproduction of the published SeqGAN or RelGAN architectures.

## What was held fixed

- 82,219 phishing URLs from the original frozen training split;
- no legitimate, validation, test, calibration, or external URLs;
- 168 character tokens and a maximum sequence length of 128;
- batch size 64, seed 42, GPU-only training, and 500,000 final generation attempts;
- the existing strict URL, TLD, uniqueness, real-overlap, encoded-overlap, and template filters.

## Architecture

`2L-LSTM-G` has a 64-dimensional character embedding, two LSTM layers with 128 hidden units, 0.2 inter-layer dropout, and 263,848 trainable parameters. It emits a differentiable character distribution using Gumbel-Softmax at temperature 0.8. Generation is stopped logically at the end token and all remaining positions are masked as padding.

`Char-CNN-D` has a 64-dimensional character embedding and 128 convolution filters for each kernel size 2, 3, 4, and 5. Global max pooling, dropout 0.3, and a binary real/fake output give 126,465 trainable parameters. It distinguishes real training phishing URLs from generated strings; it is not the phishing-versus-legitimate HYB classifier.

## Training

The generator first received five full maximum-likelihood language-model epochs. Adversarial training then alternated one discriminator update and one generator update for 150 steps per epoch across four epochs.

The generator objective combined:

1. adversarial BCE through Gumbel-Softmax;
2. MLE regularization with weight 0.5;
3. discriminator feature matching with weight 0.5; and
4. marginal character-diversity regularization with weight 0.02.

Adversarial epoch 3 was selected by the pre-existing validity × uniqueness × non-memorization × non-template-repetition rule.

## Result

The selected checkpoint produced 98.15% raw URL-like validity, 99.81% raw uniqueness, zero exact real-URL memorization, and 239,368 strictly accepted URLs from 500,000 attempts. It is stable and supports all registered augmentation sizes.

It does not outperform the original SG reference: SG accepted 349,687 rows, had closer lexical distributions, covered more real phishing clusters, and generated more samples challenging to the majority of frozen classifiers. URL-GAN is therefore retained as a useful architecture comparison; SG remains the preferred augmentation generator.

Complete comparison: `artifacts/results/urlgan_experiment/REPORT.md`.
