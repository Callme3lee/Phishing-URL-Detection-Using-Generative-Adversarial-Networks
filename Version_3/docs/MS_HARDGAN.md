# MS-HardGAN Defensive Hard-Example Experiment

## Research claim

MS-HardGAN does **not** claim to predict zero-day phishing attacks. A generator cannot learn a completely absent future concept. It generates novel, realistic, non-memorized combinations of patterns represented in the frozen multi-source (MS) phishing corpus and searches for samples that expose shared blind spots in frozen classifiers. These are **zero-day-like hard examples**, not forecasts.

## Frozen training corpus

Only phishing rows from the exact stored HYB-MS training input are eligible:

| Source | Rows |
|---|---:|
| URLs_New phishing | 82,219 |
| PhishTank development phishing | 2,000 |
| OpenPhish development phishing | 61 |
| **Total** | **84,280** |

Validation, test, calibration, external, generated, and Version 2 rows are excluded. The freeze is recorded in `data/experiments/ms_hardgan_v1/preparation_report.json`. Eight structural families are learned reproducibly with RobustScaler and MiniBatchKMeans over the 35 scheme-free PSL-aware lexical features.

All rows are consumed once per maximum-likelihood epoch. Adversarial training uses moderate source weighting with exponent 0.35 and a maximum weight of 8.0. The cap prevents the 61 OpenPhish rows from dominating through repetition.

## Architecture

### Conditional generator

- two-layer character LSTM;
- 64-dimensional character embeddings;
- 24-dimensional structural-family and source embeddings;
- 64-dimensional random latent vector;
- 128 hidden units;
- Gumbel-Softmax relaxation for differentiable categorical generation;
- hard categorical policy samples only for the classifier-challenge reward.

### Multi-head discriminator

A character CNN uses kernel widths 2, 3, 4, and 5 with 128 channels per width. Its shared representation feeds three heads:

1. real phishing versus generated;
2. requested structural family;
3. requested MS source category.

The category heads make conditional adherence measurable. They do not prove maliciousness or future relevance.

### Frozen classifier challenge ensemble

Seed-42 CNN-MS, LSTM-MS, and RNN-MS checkpoints are loaded read-only. Each generated phishing score is converted to its percentile in that classifier's frozen validation-score distribution. The reward averages rank-normalized difficulty and the threshold-miss indicator across the three families. Raw model scores are never directly combined.

The challenge weight is 0.05, starts at adversarial epoch 2, and is evaluated every fourth generator step. Realism, category consistency, maximum-likelihood regularization, feature matching, and diversity remain active so nonsense cannot win merely by obtaining a low classifier score.

## Frozen schedule

- GPU required; CPU fallback disabled.
- Seed 42 and batch size 64.
- Five full maximum-likelihood epochs.
- Four adversarial epochs with 250 steps each.
- Best checkpoint selected by validity + uniqueness - template repetition - exact memorization.
- Fixed raw generation budget of 500,000.
- Exact URL, hostname, encoding, duplicate, invalid, non-TLD-like, and repeated-template rejection.
- Twenty percent of accepted rows frozen by URL hash as a synthetic challenge holdout; these cannot be used for augmentation selection or training.

## Required evaluation

Report validity, strict acceptance, uniqueness, overlap, template concentration, character n-gram distances, lexical distances, structural-family coverage, requested-category agreement, and multi-family challenge rates. Downstream experiments retain real-only and 10%, 20%, 30%, and 40% augmentation, seeds 42/43/44, and unchanged validation, test, external data, classifiers, thresholds, and metrics.

Evidence of value requires improved later-snapshot recall/FNR or ranking without a material increase in legitimate FPR. Hard-example rate or uniqueness alone is insufficient.

## Research basis

- [SeqGAN](https://arxiv.org/abs/1609.05473): policy-gradient learning for discrete sequences.
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144): differentiable categorical relaxation.
- [RelGAN](https://openreview.net/forum?id=rJedV3R5tm): informative CNN discriminator representations and Gumbel text generation.
- [Conditional phishing URL GAN](https://arxiv.org/abs/2108.01852): real/fake and malicious/benign attacker-defender objectives.
- [ADC-GAN](https://proceedings.mlr.press/v162/hou22a.html): why auxiliary classification needs diversity control.
- [PhishSim](https://arxiv.org/abs/2207.10801): previous variations cannot reveal a completely distinctive structure that never occurred.
- [Evasion-robust phishing URL detection](https://arxiv.org/abs/2209.01454): URL text should ultimately be complemented by network/context evidence.
- [PhishTime](https://www.usenix.org/conference/usenixsecurity20/presentation/oest-phishtime): longitudinal controlled measurement of evolving phishing infrastructure.

## Commands

```bash
PYTHONPATH=src python3 -m phishing_gans_v3.prepare_ms_hardgan
PYTHONPATH=src python3 -m phishing_gans_v3.train_ms_hardgan
PYTHONPATH=src python3 -m phishing_gans_v3.generate_ms_hardgan
PYTHONPATH=src python3 -m phishing_gans_v3.materialize_ms_hardgan_experiments
PYTHONPATH=src python3 -m phishing_gans_v3.train_ms_hardgan_augmentations
PYTHONPATH=src python3 -m phishing_gans_v3.evaluate_ms_hardgan_challenge
PYTHONPATH=src python3 -m phishing_gans_v3.summarize_ms_hardgan_augmentation
```

## Completed results (MS-HardGAN-C)

The initial `ms_hardgan_v1` condition was retained as a transparent ablation. It achieved 28.95% requested-pattern agreement. This exceeded the 12.5% chance rate but was not considered adequate categorical control. It was not silently overwritten.

The corrected `MS-HardGAN-C` (`ms_hardgan_v2`) adds pattern-balanced adversarial sampling where available, increases pattern supervision from 0.25 to 1.0, reduces source supervision to 0.1, and delays the classifier challenge signal until adversarial epoch 3. On the final 630,000-sample generation run:

| Measure | Result |
|---|---:|
| Raw uniqueness | 99.93% |
| Raw URL-like validity | 94.47% |
| Strict accepted rows | 309,208 (49.08%) |
| Accepted uniqueness / validity | 100% / 100% |
| Accepted real URL or hostname overlap | 0 |
| Requested-pattern agreement | 72.22% |
| Requested-source agreement | 96.85% |
| Real structural-cluster coverage | 90.00% |
| Source classifier ROC-AUC | 0.7992 ± 0.0055 |
| Majority-classifier challenge rate | 9.95% |
| Challenging to at least one classifier | 52.83% |
| Challenging to every classifier | 2.48% |

The source-classifier AUC shows that real and generated distributions remain distinguishable; therefore the corpus is useful as a controlled stress-test candidate, not a claim that the real phishing distribution has been perfectly reproduced.

The URL-hash split freezes 61,728 accepted rows as the synthetic challenge holdout and leaves 247,480 rows eligible for training. Deterministic nested pools contain 40,410, 90,923, 155,868, and 242,461 generated rows for exact 10%, 20%, 30%, and 40% shares of the final training set. Pattern family 5 has only 786 eligible rows after strict filtering; later pools retain all 786 rather than duplicating them or weakening filters. This is disclosed as a remaining conditional-coverage limitation.

Artifacts:

- `artifacts/generators/ms_hardgan_v2/best.pt`
- `data/generated/ms_hardgan_v2_final/audit.json`
- `artifacts/results/ms_hardgan_v2/quality/quality.json`
- `artifacts/results/ms_hardgan_v2/quality/quality_summary.pdf`
- `data/augmentations/ms_hardgan_v2/manifest.json`

## Controlled downstream results

All five classifier families were retrained at 10%, 20%, 30%, and 40% generated shares for seeds 42/43/44: 60 augmented runs. Every condition retains the same 363,692 real MS rows and the unchanged real validation/test data. The frozen 10,000-row external snapshot and the 61,728-row synthetic challenge holdout were not used for training, pool selection, or threshold selection.

On the unchanged internal real test set, real-only remains strongest overall. Mean CNN F1 falls from 0.9737 real-only to 0.9728, 0.9665, 0.9644, and 0.9655 as the generated share increases. RF F1 falls monotonically from 0.8644 to 0.8458 at 40%; SVM falls from 0.6889 to 0.6636. The 40% LSTM is close to, but does not exceed, real-only F1 (0.9714 versus 0.9721).

The frozen external snapshot does not support a real-world improvement claim. LSTM retains the best ranking, but ROC-AUC decreases from 0.9309 real-only to 0.9090 at 40%. Its FPR improves from 93.15% to 89.10%, which remains unusable. CNN and RNN also retain approximately 91–99% external FPR. RF and SVM ranking degrades substantially. High external phishing recall must not be presented without these legitimate false-positive rates.

Augmentation does improve recognition of held-out outputs from this generator. Mean challenge recall changes as follows:

| Model | Real only | 10% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|
| CNN | 94.36% | 98.20% | 98.56% | 98.93% | 98.95% |
| LSTM | 90.57% | 97.38% | 98.22% | 98.25% | 98.82% |
| RNN | 89.24% | 93.95% | 97.50% | 98.20% | 98.36% |
| RF | 75.78% | 84.53% | 85.99% | 86.65% | 86.86% |
| SVM | 52.79% | 59.00% | 57.52% | 57.24% | 62.80% |

This is evidence that classifiers learned coverage of the MS-HardGAN-C distribution. It is **not** evidence that the GAN predicts zero-day attacks or that overall real-world performance improved. Because the challenge set contains only phishing rows, only recall/FNR are defined; accuracy, precision, F1, ROC-AUC, and FPR are intentionally omitted.

Complete downstream report and graphs:

- `artifacts/results/ms_hardgan_augmentation_v2/comparison/REPORT.md`
- `artifacts/results/ms_hardgan_augmentation_v2/comparison/internal_all_metrics.pdf`
- `artifacts/results/ms_hardgan_augmentation_v2/comparison/external_all_metrics.pdf`
- `artifacts/results/ms_hardgan_augmentation_v2/comparison/synthetic_challenge_recall.pdf`
