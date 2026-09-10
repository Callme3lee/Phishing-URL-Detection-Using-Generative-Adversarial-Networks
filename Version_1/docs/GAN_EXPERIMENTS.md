# GAN Experiments

## LSGAN pilot

The character LSGAN uses a 1.12M-parameter convolutional generator, a
0.89M-parameter discriminator, fixed 128-character outputs, and straight-through
Gumbel-Softmax samples. It was trained on a reproducible subset of 100,000
malicious training URLs for five epochs.

Run it with:

```bash
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans.train_gan \
  --gan lsgan \
  --max-samples 100000 \
  --epochs 5 \
  --sample-count 1000 \
  --output-dir artifacts/lsgan_pilot
```

Evaluate any saved epoch with:

```bash
python3 -m phishing_gans.evaluate_generated \
  --samples artifacts/lsgan_pilot/samples_epoch_005.npy
```

### Finding

This pilot exhibited mode collapse. Bigram diversity remained below 0.2%, and
inspection showed repeated characters rather than coherent URL grammar. The
samples must not be used for classifier augmentation. This negative result is
preserved because it motivates the WGAN-GP comparison and prevents misleading
claims based only on classifier scores or uniqueness.

## WGAN-GP pilot

WGAN-GP reuses the exact generator, critic, data subset, encoding and evaluation.
It changes the objective to Wasserstein loss, adds a gradient penalty of 10, uses
five critic updates per generator update, Adam betas `(0.0, 0.9)`, and a learning
rate of `1e-4`.

```bash
python3 -m phishing_gans.train_gan \
  --gan wgan-gp \
  --max-samples 100000 \
  --epochs 5 \
  --sample-count 1000 \
  --output-dir artifacts/wgan_gp_pilot
```

### Finding

WGAN-GP stabilized the critic and reduced the average gradient penalty from
0.212 at epoch 1 to 0.023 at epoch 5. It nevertheless collapsed to short runs
of a few characters. At epoch 5, structural validity was 0.5%, unique rate was
28.7%, bigram diversity was 0.11%, and exact training memorization was 0%.

The existing CNN labeled 95.6% of these outputs malicious, but this is not a
quality success: the visible outputs are abnormal repeated strings, so the CNN
is reacting to obvious out-of-distribution patterns. These samples must not be
used for augmentation.

## Decision after the pilots

Neither pilot should be scaled to the full 1.19M malicious training URLs. The
controlled results show that the shared fixed-length convolutional generator
and straight-through hard Gumbel representation are the bottleneck. More data
will not supply the missing sequential URL grammar.

The next model revision should preserve the LSGAN and WGAN-GP results as
baselines, then introduce an autoregressive character generator or explicit
URL-length/component conditioning. Any revised generator must pass structural
validity and diversity gates before classifier augmentation.

## Successful autoregressive redesign

The collapse was fixed by replacing simultaneous convolutional generation with
a 154,933-parameter autoregressive LSTM generator. Each URL now uses explicit
`BOS` and `EOS` tokens and is generated one character at a time.

The generator was first pretrained on 100,000 malicious URLs for five epochs
using teacher-forced character cross-entropy. Validation perplexity decreased
from 10.92 to 7.19. Its epoch-5 samples achieved 99.5% structural validity,
100% uniqueness, and 0% exact training memorization.

```bash
python3 -m phishing_gans.pretrain_generator \
  --max-samples 100000 \
  --validation-samples 20000 \
  --epochs 5 \
  --sample-count 1000 \
  --output-dir artifacts/generator_pretrained
```

Adversarial fine-tuning retains a weighted language-model loss and adds soft
Gumbel sampling, spectral normalization, feature matching, diversity
regularization, and a lower critic learning rate.

```bash
python3 -m phishing_gans.finetune_autoregressive_gan \
  --gan lsgan \
  --max-samples 100000 \
  --epochs 3 \
  --batch-size 64 \
  --output-dir artifacts/autoregressive_lsgan

python3 -m phishing_gans.finetune_autoregressive_gan \
  --gan wgan-gp \
  --max-samples 100000 \
  --epochs 3 \
  --batch-size 64 \
  --output-dir artifacts/autoregressive_wgan_gp
```

### Final pilot comparison

| Model, epoch 3 | Structural validity | Unique | Memorization | Bigram diversity | CNN malicious-style detection |
|---|---:|---:|---:|---:|---:|
| LSTM-LSGAN | 99.8% | 100% | 0.6% | 4.35% | 94.5% |
| LSTM-WGAN-GP | 98.6% | 100% | 0.1% | 3.66% | 95.5% |

Both redesigned models pass the provisional quality gate. WGAN-GP has lower
memorization and higher CNN malicious-style detection; LSGAN has slightly
higher structural validity and n-gram diversity. Generated URLs still require
quality filtering and a larger evaluation sample before classifier augmentation.
