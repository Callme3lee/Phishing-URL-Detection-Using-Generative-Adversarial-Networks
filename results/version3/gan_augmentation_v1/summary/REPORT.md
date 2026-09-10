# Version 3 Controlled GAN Augmentation Study

## Conclusion

The experiment is complete. SeqGAN generated a large, unique, non-memorized pool and was suitable for controlled augmentation. RelGAN failed the preregistered realism/volume gate and was not used to fabricate classifier results. SeqGAN augmentation did **not** consistently outperform real-only training and did **not** fix external distribution shift. High synthetic fractions generally degraded results, especially for the vanilla RNN, Random Forest, and SVM. Real source coverage and the previously developed calibrated hybrid remain the better direction.

This is a valid negative result: GAN uniqueness and lexical coverage do not by themselves prove useful augmentation.

## Frozen design

- GAN training data: all 82,219 phishing URLs from the frozen URLs_New training split only.
- Baseline real rows: 358,619 (276,400 legitimate; 82,219 phishing).
- Coverage reference rows: 361,631, with the original validation and test splits unchanged.
- Synthetic fractions: 10%, 20%, 30%, and 40% of final training rows.
- Synthetic counts held equal between matching URLs_New and coverage-plus conditions: 39,847; 89,655; 153,694; and 239,079.
- Classifiers: CNN, LSTM, vanilla RNN, Random Forest, and linear SVM.
- Seeds: 42, 43, and 44.
- Thresholds: selected only from the unchanged internal validation set under the existing maximum-1%-FPR policy.
- Evaluation: unchanged internal test plus the frozen 500-row Top-Domain/PhishTank and 400-row Tranco/OpenPhish sets.

## Why these GAN architectures were selected

SeqGAN uses an autoregressive LSTM generator and a multi-kernel character CNN discriminator. Policy-gradient rewards with Monte Carlo rollouts address the discrete-token gradient problem described in the [SeqGAN paper](https://aaai.org/papers/10804-aaai-31-2017/). The supplied phishing-GAN evaluation paper also identified SeqGAN as its strongest tested generator for structural similarity and classifier deception.

RelGAN uses a relational-memory generator, Gumbel-Softmax categorical relaxation, and a multi-representation CNN discriminator. These are the text-specific components introduced by [RelGAN](https://iclr.cc/virtual/2019/poster/966), with the relaxation grounded in [Gumbel-Softmax](https://arxiv.org/abs/1611.01144). Plain DCGAN/WGAN-GP was not selected as a primary URL-text generator because continuous-data critic improvements do not alone solve discrete character sampling; see [WGAN-GP](https://papers.nips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html).

## Generator results

| Measure | SeqGAN | RelGAN |
|---|---:|---:|
| Raw samples | 500,000 | 500,000 |
| Raw uniqueness | 99.839% | 99.388% |
| Parse-valid | 98.171% | 81.428% |
| Strictly accepted | 349,687 (69.94%) | 12,869 (2.57%) |
| Exact training memorization | 0% | 0% |
| Real-cluster coverage | 94% | 86% |
| Real-vs-generated source ROC-AUC | 0.7357 | 0.9888 |
| Majority-classifier challenging | 10.92% | 0.085% |
| Eligible for 10–40% study | Yes | No |

RelGAN's adversarial validity changed from 81.86% at epoch 1 to 34.16% at epoch 2 and 47.00% at epoch 3. The frozen selection rule correctly retained epoch 1, but even that checkpoint produced mostly non-TLD-like structures. RelGAN is therefore diverse but unrealistic and easily distinguishable, not a useful augmentation source in this implementation.

## Internal test F1 (mean across three seeds)

| Model | Real | SeqGAN 10% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|
| CNN | 0.9736 | **0.9764** | 0.9758 | 0.9751 | 0.9724 |
| LSTM | **0.9756** | 0.9722 | 0.9744 | 0.9750 | 0.9731 |
| RNN | **0.9679** | 0.9679 | 0.9532 | 0.9639 | 0.9623 |
| Random Forest | **0.8667** | 0.8625 | 0.8576 | 0.8508 | 0.8433 |
| Linear SVM | 0.6835 | 0.6924 | **0.6935** | 0.6920 | 0.6887 |

The small CNN/SVM internal gains are architecture-specific and do not transfer consistently to external data. LSTM never exceeds its real-only mean. RF degrades monotonically. RNN is unstable and degrades at larger ratios.

## External findings

URLs_New plus SeqGAN does not address the main legitimate-side shift. On Top-Domain/PhishTank, mean neural FPR remains 94.7–99.9% across ratios. On Tranco/OpenPhish it remains 97.5–100%. High phishing recall in this setting is misleading because nearly every legitimate URL is also flagged.

Adding SeqGAN after real coverage also fails to provide consistent value. For the strongest coverage-only external family, LSTM:

| Evaluation | Coverage only F1 / AUC / FPR | +10% SeqGAN | +20% | +30% | +40% |
|---|---:|---:|---:|---:|---:|
| Top-Domain/PhishTank | **0.6787 / 0.909 / 0.859** | 0.6515 / 0.827 / 0.968 | 0.6561 / 0.759 / 0.893 | 0.6603 / 0.744 / 0.864 | 0.6189 / 0.628 / 0.867 |
| Tranco/OpenPhish | **0.6460 / 0.837 / 0.883** | 0.6245 / 0.644 / 0.975 | 0.6242 / 0.590 / 0.933 | 0.6270 / 0.636 / 0.917 | 0.5924 / 0.566 / 0.918 |

Thus, the GAN does not replace real samples for correcting distribution shift. At best, low-ratio augmentation produces small internal changes; it does not deliver a reliable external robustness improvement.

## Interpretation

1. SeqGAN learned a plausible subset of phishing lexical structure without exact memorization.
2. The generated distribution is still detectably different from real phishing (source ROC-AUC 0.7357).
3. Generated phishing adds no missing legitimate-domain coverage, which is the dominant external failure in URLs_New-only classifiers.
4. Increasing the phishing-only synthetic share changes class/distribution weighting and can damage the operating threshold and decision boundary.
5. RelGAN's strong uniqueness was not enough: its near-perfect source separability and low TLD acceptance made it unsuitable.
6. The best deployment direction remains the calibrated hybrid CNN + PSL-aware lexical model with real multi-source coverage. GAN augmentation is an experimental robustness tool, not the production default.

## Reproducibility and artifacts

- `summary.json`: all means, sample SDs, seed values, and paired deltas.
- `all_results.csv`: tidy results for every evaluation, condition, model, and metric.
- `internal_all_metrics.*`, `topdomains_phishtank_all_metrics.*`, and `tranco_openphish_all_metrics.*`: separate metric panels.
- `data/generated/{seqgan,relgan}/audit.json`: raw/accepted generator audits.
- `artifacts/results/gan_quality/{seqgan,relgan}/quality.json`: distribution and classifier-challenge audits.
- `artifacts/experiments/gan_augmentation_v1/`: 120 model artifacts and metrics.

All 8 automated tests pass. A genuinely later untouched temporal snapshot is still required for a confirmatory deployment claim.
