# RelGAN Model — Full Corpus and Augmentation Screen Complete

## Status boundary

RelGAN architecture and its CUDA trainer exist in `src/phishing_gans_v2/relgan.py`
and `train_relgan.py`. Training and a checkpoint-level quality audit are
complete. A full two-million-sample corpus was generated and strictly filtered,
nested 10–40% manifests were built, and the primary 10% condition was screened
with all four classifiers at seed 42. It did not satisfy the promotion rule, so
seeds 43/44 and higher-dose classifier runs were deliberately not performed.

## Why this is the next candidate

RelGAN was designed for discrete text generation and combines three ideas that
fit character-level URLs: a relational-memory generator, differentiable
Gumbel-Softmax samples, and a discriminator that views the sequence through
multiple learned embeddings. The original paper also exposes temperature as a
quality/diversity control. These properties make it a more appropriate future
candidate than directly applying an image GAN to URL strings.

Primary references:

- [RelGAN: Relational Generative Adversarial Networks for Text Generation](https://openreview.net/forum?id=rJedV3R5tm)
- [Relational recurrent neural networks](https://arxiv.org/abs/1806.01822)
- [Authors' broader TextGAN PyTorch framework](https://github.com/williamSYSU/TextGAN-PyTorch)

## Staged implementation

- `RelationalMemoryCore`: recurrent memory slots updated by multi-head
  self-attention, an MLP, and learned input/forget gates.
- `RelGANGenerator`: character embeddings, optional latent initialization,
  relational memory, autoregressive logits, and differentiable Gumbel-Softmax
  sampling.
- `RelGANDiscriminator`: multiple independent embeddings followed by parallel
  URL-character CNN filters and one real/fake logit.
- `RelGAN`: a lightweight container for the pair.

The implementation is intentionally compact and project-native rather than a
claim of byte-for-byte reproduction of the authors' research repository.

## Completed training protocol

- Data source: the same frozen malicious training split used by the other GANs.
- Seeded training view: 200,000 rows.
- Actually consumed: 105,600 distinct examples (76,800 during MLE warm-up and
  28,800 during adversarial training).
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU.
- Schedule: three warm-up epochs × 400 batches and three adversarial epochs ×
  150 batches, batch size 64.
- Selected checkpoint: adversarial epoch 3.

| Metric | Selected RelGAN checkpoint |
|---|---:|
| Audit samples | 2,000 |
| Uniqueness | 1.0000 |
| URL-like validity | 0.9955 |
| Exact memorization | 0.0005 |
| Repeated-template rate | 0.0000 |
| Selection score | 0.995002 |

Gumbel-LSGAN and SeqGAN each score 0.9995 in their comparable selected-checkpoint
audits. RelGAN therefore trains without uniqueness collapse, but currently
trails them at the small checkpoint audit because of validity.

## Full-corpus verification

| Metric | RelGAN raw corpus | Strictly accepted corpus |
|---|---:|---:|
| Samples | 2,000,000 | 1,989,755 |
| Uniqueness | 0.998839 | 1.000000 |
| URL-like validity | 0.996026 | 1.000000 |
| Exact memorization | 0.000171 | 0.000000 |
| Repeated-template rate | 0.000611 | 0.000000 |
| Acceptance rate | 0.994878 | — |

RelGAN beats the original two GANs on full-corpus raw uniqueness, acceptance,
memorization, template concentration, and accepted count. Its raw validity is
lower (99.6026% versus about 99.97%), so strict filtering remains essential.

## RelGAN 10% augmentation screen

Every run used all 2,055,821 unchanged real training rows plus 228,425 accepted
RelGAN URLs. Validation/test data and the threshold policy were unchanged.
Values are seed-42 augmented-minus-real-only changes.

| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN | -0.001844 | +0.000537 | -0.003772 | -0.001730 | -0.000073 |
| LSTM | -0.000492 | +0.000525 | -0.001376 | -0.000471 | +0.000123 |
| Random Forest | -0.000520 | +0.000221 | -0.001125 | -0.000503 | -0.000279 |
| Linear SVM | -0.005199 | -0.000469 | -0.008761 | -0.005113 | -0.000271 |

No classifier improved all five requested metrics. LSTM was closest, improving
precision and ROC-AUC while reducing accuracy, recall, and F1. Therefore no
RelGAN condition was promoted to seeds 43/44. These are screening results from
one seed, not confidence-interval estimates.

Evidence:

- `artifacts/generators/relgan/best.pt`
- `artifacts/generators/relgan/history.json`
- `artifacts/results/generators/checkpoint_audits/summary.json`
- `artifacts/results/generators/checkpoint_audits/comparison.png`
- `artifacts/generated/relgan/metrics.json`
- `data/augmented/relgan/index.json`
- `artifacts/results/augmentation/relgan_screen/summary.json`
- `artifacts/results/augmentation/relgan_screen/delta_heatmap.png`

## Commands

From `Version_2`, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_relgan.py' -v
PYTHONPATH=src python3 -m phishing_gans_v2.train_relgan
PYTHONPATH=src python3 -m phishing_gans_v2.train_relgan --resume
PYTHONPATH=src python3 -m phishing_gans_v2.summarize_relgan
PYTHONPATH=src python3 -m phishing_gans_v2.generate_corpus --model relgan --batch-size 64 --commit-interval 20000 --max-gpu-temperature 84
PYTHONPATH=src python3 -m phishing_gans_v2.build_augmented_sets --model relgan
PYTHONPATH=src python3 -m phishing_gans_v2.run_experiment_matrix --generators relgan --doses 10 --models cnn,lstm,random_forest,linear_svm --seeds 42 --include-augmentation
PYTHONPATH=src python3 -m phishing_gans_v2.summarize_relgan_augmentation
```

The unit-test command only checks tensor shapes and gradient flow. The training
command requires CUDA and writes resumable checkpoints after each epoch.

The `generate_corpus` command is resumable and uses a thermal guard. Because the
10% screen failed, higher doses and multi-seed confirmation are not justified
under the frozen screen-then-confirm protocol.
