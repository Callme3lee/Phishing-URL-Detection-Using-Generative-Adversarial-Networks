# Two-stage URL cascade — one-shot evaluation

At the strict 0.25% calibration budget, the cascade changed recall by +23.30 points and FNR by -23.30 points; observed FPR changed by +0.028 points. This is evidence for the cascade only at this locked operating point, not proof that it eliminates false negatives.

| Calibration FPR cap | System | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25% | Single-stage GAN | 94.51% | 98.60% | 45.75% | 62.50% | 97.52% | 92.88% | 0.07% | 54.25% |
| 0.25% | Two-stage cascade | 96.81% | 98.71% | 69.05% | 81.26% | 96.43% | 91.72% | 0.10% | 30.95% |
| 0.50% | Single-stage GAN | 97.55% | 94.83% | 79.85% | 86.70% | 97.52% | 92.88% | 0.48% | 20.15% |
| 0.50% | Two-stage cascade | 97.48% | 96.98% | 77.20% | 85.97% | 96.43% | 91.72% | 0.27% | 22.80% |
| 1.00% | Single-stage GAN | 97.76% | 89.99% | 87.25% | 88.60% | 97.52% | 92.88% | 1.08% | 12.75% |
| 1.00% | Two-stage cascade | 97.56% | 90.96% | 84.00% | 87.34% | 96.43% | 91.72% | 0.93% | 16.00% |

This is identity-disjoint, not temporally newer. No post-test threshold tuning was performed.
