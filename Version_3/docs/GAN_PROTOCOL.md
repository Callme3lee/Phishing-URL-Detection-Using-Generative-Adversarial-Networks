# Frozen Version 3 GAN Protocol

## Research question

Can character-level GAN augmentation improve phishing-distribution coverage and downstream cross-source URL detection, rather than merely produce unique strings or improve an internal accuracy number?

The result may be improvement, no change, or degradation. Ratios, models, filters, and evaluation sets must not be changed after outcomes are observed to force a positive result.

## Selected generators and discriminators

### SeqGAN

SeqGAN uses an autoregressive character LSTM generator and treats hard token generation as a stochastic policy. A multi-kernel character CNN discriminator supplies complete-sequence rewards, which are propagated to intermediate actions through strided Monte Carlo rollouts. Maximum-likelihood warm-up and regularization are retained for stability.

This is appropriate because URL characters are discrete. The original [SeqGAN paper](https://aaai.org/papers/10804-aaai-31-2017/) explicitly addresses the inability to backpropagate discriminator gradients through discrete samples. The supplied evaluation paper also reports SeqGAN as its strongest tested generator for structural similarity and classifier deception.

### RelGAN

RelGAN uses a relational-memory character generator, differentiable Gumbel-Softmax samples, and a discriminator with three independent embeddings and parallel CNN kernels. The relational memory models longer dependencies among hostname, path, and query characters; the multiple discriminator views provide a richer adversarial signal.

The [RelGAN paper](https://iclr.cc/virtual/2019/poster/966) introduces precisely these components for text generation and reports an adjustable quality/diversity trade-off. [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) provides the continuous relaxation required for gradient-based training with categorical characters.

### Why not image GANs or plain WGAN-GP

DCGAN and WGAN-GP are valuable for continuous data and critic stability, but neither by itself resolves discrete character sampling. [WGAN-GP](https://papers.nips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html) is therefore not a primary model here. The chosen pair represents the two practical text-GAN strategies: policy gradient and differentiable relaxation.

## Frozen training contract

- Generator input: only the 82,219 phishing rows from the frozen `URLs_New` training split.
- No legitimate, validation, test, calibration, external, coverage, or Version 2 row may train either generator.
- Vocabulary, maximum length 128, and canonicalization are inherited from Version 3.
- Seed 42 trains each generator; downstream classifiers use seeds 42/43/44.
- CUDA is mandatory.
- Both generators use a multi-kernel CNN discriminator over character embeddings. RelGAN uses three independently learned embedding views.
- SeqGAN uses full shuffled MLE passes. RelGAN uses 400 seeded shuffled MLE batches per epoch to keep relational attention within the RTX 3060 laptop's thermal envelope; its eligible pool remains all 82,219 phishing rows and exact consumption is logged.

## Quality gates

Every generated corpus is reported before augmentation using:

- parse validity and strict accepted rate;
- raw and accepted uniqueness;
- exact URL, hostname, and encoded-input overlap with all real project data;
- character n-gram nearest-neighbour distances;
- prefix-template concentration and cluster coverage;
- lexical distribution distances and source-classifier ROC-AUC;
- nearest-neighbour distance to real phishing;
- frozen real-only classifier challenge rate across multiple model families.

Filtering removes invalid URLs, exact/hostname/encoding overlap, duplicates, and excessive repeated templates. Filtering rules are identical for SeqGAN and RelGAN.

The raw generation budget is fixed at 500,000 samples per generator. This was chosen before downstream evaluation so the 40% condition can retain 239,079 unique samples after strict rejection filtering. Unused accepted URLs are never substituted based on classifier results.

## Augmentation conditions

For each generator, nested accepted pools create fixed 10%, 20%, 30%, and 40% generated fractions, where:

`fraction = generated rows / final augmented training rows`.

All 358,619 original real training rows remain unchanged. Validation, test, calibration, external data, thresholds, model architectures, and seeds remain fixed.

The controlled comparison includes:

1. URLs_New real-only;
2. URLs_New plus GAN phishing;
3. real outside-coverage reference;
4. real outside coverage plus GAN phishing.

The GAN-only comparison answers whether synthetic phishing can replace real shift-control rows. The final comparison tests whether synthetic phishing adds value after legitimate/phishing source coverage is improved.

## Interpretation rule

Uniqueness is necessary but insufficient. Evidence for useful distribution-shift improvement requires realistic and diverse samples, improved external recall/FNR or ranking, and no material increase in legitimate FPR. Existing external results are exploratory; a genuinely later untouched snapshot remains the confirmation requirement.
