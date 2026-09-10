# Recent-observation update: real-only versus 10% GAN

## Question

When both HYB-MS conditions receive the same small layer of recently observed real URLs, does the frozen 10% GAN augmentation provide additional benefit?

## Fair comparison

| Component | Updated real-only | Updated +10% GAN |
|---|---:|---:|
| Original multi-source real rows | 363,692 | 363,692 |
| Recent-update real rows | 3,492 | 3,492 |
| Total identical real rows | 367,184 | 367,184 |
| Frozen MS-HardGAN-C rows | 0 | 40,410 |
| Seeds | 42/43/44 | 42/43/44 |

Architecture, PSL-aware feature processing, weighted source-sampling family, CORAL weight, early stopping, validation/test identities, threshold policy, and metrics are fixed. The GAN condition targets 10% synthetic exposure. Synthetic URLs are excluded from CORAL pairs, as in the registered earlier experiment.

The update overlapped 42 registered domains represented by 994 rows in the old validation split and 35 domains represented by 602 old test rows. Those rows were removed identically from both conditions before training. The protected failed-confirmation cohort contributed zero rows and was not reevaluated.

## Internal results

Mean ± sample standard deviation across seeds 42/43/44:

| Metric | Updated real-only | Updated +10% GAN | Mean GAN delta |
|---|---:|---:|---:|
| Accuracy | 93.08% ± 0.56% | 93.63% ± 0.46% | +0.56 points |
| Precision | 99.96% ± 0.01% | 99.97% ± 0.00% | +0.02 points |
| Recall | 67.35% ± 2.67% | 69.96% ± 2.19% | +2.61 points |
| F1 | 80.46% ± 1.92% | 82.31% ± 1.53% | +1.85 points |
| ROC-AUC | 99.48% ± 0.07% | 99.48% ± 0.10% | −0.003 points |
| PR-AUC | 98.92% ± 0.12% | 98.89% ± 0.12% | −0.034 points |
| FPR | 0.0078% ± 0.0026% | 0.0052% ± 0.0000% | −0.0026 points |
| FNR | 32.65% ± 2.67% | 30.04% ± 2.19% | −2.61 points |

The +10% condition improves recall and F1 for seeds 42 and 43 but loses for seed 44. It improves precision for all three seeds and improves or ties FPR for all three. ROC-AUC and PR-AUC are essentially tied and are slightly lower on average under GAN augmentation.

## Conclusion

The controlled internal evidence is promising but mixed: 10% GAN augmentation improves the selected operating point on average after both models receive the same recent update, but the benefit is not seed-consistent and does not improve mean ranking metrics. This is not external confirmation and does not reverse the earlier failed deployment confirmation.

The next valid action is to collect a new identity-disjoint calibration cohort, freeze an ensemble threshold for each condition under the same rule, and later evaluate both once on a new future confirmation cohort. Earlier external/calibration cohorts containing update identities are retired.

## Reproduction

```bash
PYTHONPATH=src python -m phishing_gans_v3.prepare_recent_update_gan10
PYTHONPATH=src python -m phishing_gans_v3.train_fair_hybrid_gan \
  --data-root data/experiments/recent_update_gan10_v1 \
  --output-root artifacts/experiments/recent_update_gan10_v1 \
  --conditions real_only,gan_10pct --variants hyb_ms --seeds 42,43,44
PYTHONPATH=src python -m phishing_gans_v3.summarize_recent_update_gan10
```

Machine-readable results are under `artifacts/results/recent_update_gan10_v1/`.
