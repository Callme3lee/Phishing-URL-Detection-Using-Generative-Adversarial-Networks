# Training-set similarity and URL-length audit

## Exact training-set relationship

- Real-only contains **367,184 real rows**.
- The GAN condition contains the same **367,184 real rows** plus **40,410 synthetic phishing rows**.
- Real rows are identical and in the same order: **True**.
- Synthetic URLs exactly overlapping real URLs: **1**.
- Exact legitimate/phishing URL overlap inside the real set: **0**.
- Real-versus-synthetic phishing length-bucket overlap coefficient: **0.735**.

## Length summary

The model reads at most 128 characters. `over_128` therefore counts URLs whose
tail is truncated before entering the character branch.

| condition | origin | class | rows | mean_length | median_length | p95_length | maximum_length | over_100 | over_100_pct | over_128 | over_128_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Real-only | Real | Legitimate | 281,158 | 50.44 | 45.00 | 98.00 | 643 | 12,558 | 4.47% | 3,499 | 1.24% |
| Real-only | Real | Phishing | 86,026 | 55.06 | 42.00 | 140.00 | 2,307 | 8,241 | 9.58% | 5,098 | 5.93% |
| GAN condition | Real | Legitimate | 281,158 | 50.44 | 45.00 | 98.00 | 643 | 12,558 | 4.47% | 3,499 | 1.24% |
| GAN condition | Real | Phishing | 86,026 | 55.06 | 42.00 | 140.00 | 2,307 | 8,241 | 9.58% | 5,098 | 5.93% |
| GAN condition | Synthetic | Phishing | 40,410 | 52.46 | 36.00 | 128.00 | 138 | 7,347 | 18.18% | 41 | 0.10% |
| GAN condition | All | Legitimate | 281,158 | 50.44 | 45.00 | 98.00 | 643 | 12,558 | 4.47% | 3,499 | 1.24% |
| GAN condition | All | Phishing | 126,436 | 54.23 | 40.00 | 128.00 | 2,307 | 15,588 | 12.33% | 5,139 | 4.06% |

## Sorted length buckets

| condition | origin | class | length_bucket | count | percentage_within_group |
|---|---|---|---|---|---|
| Real-only | Real | Legitimate | 1-30 | 57,516 | 20.46% |
| Real-only | Real | Legitimate | 31-60 | 149,107 | 53.03% |
| Real-only | Real | Legitimate | 61-100 | 61,977 | 22.04% |
| Real-only | Real | Legitimate | 101-128 | 9,059 | 3.22% |
| Real-only | Real | Legitimate | >128 | 3,499 | 1.24% |
| Real-only | Real | Phishing | 1-30 | 26,525 | 30.83% |
| Real-only | Real | Phishing | 31-60 | 35,033 | 40.72% |
| Real-only | Real | Phishing | 61-100 | 16,227 | 18.86% |
| Real-only | Real | Phishing | 101-128 | 3,143 | 3.65% |
| Real-only | Real | Phishing | >128 | 5,098 | 5.93% |
| GAN condition | Synthetic | Phishing | 1-30 | 17,333 | 42.89% |
| GAN condition | Synthetic | Phishing | 31-60 | 10,011 | 24.77% |
| GAN condition | Synthetic | Phishing | 61-100 | 5,719 | 14.15% |
| GAN condition | Synthetic | Phishing | 101-128 | 7,306 | 18.08% |
| GAN condition | Synthetic | Phishing | >128 | 41 | 0.10% |
| GAN condition | All | Legitimate | 1-30 | 57,516 | 20.46% |
| GAN condition | All | Legitimate | 31-60 | 149,107 | 53.03% |
| GAN condition | All | Legitimate | 61-100 | 61,977 | 22.04% |
| GAN condition | All | Legitimate | 101-128 | 9,059 | 3.22% |
| GAN condition | All | Legitimate | >128 | 3,499 | 1.24% |
| GAN condition | All | Phishing | 1-30 | 43,858 | 34.69% |
| GAN condition | All | Phishing | 31-60 | 45,044 | 35.63% |
| GAN condition | All | Phishing | 61-100 | 21,946 | 17.36% |
| GAN condition | All | Phishing | 101-128 | 10,449 | 8.26% |
| GAN condition | All | Phishing | >128 | 5,139 | 4.06% |

## Threshold choices measured on the separate policy-development set

| option | threshold | development_accuracy | development_precision | development_recall | development_f1 | development_fpr | development_fnr | selection_source |
|---|---|---|---|---|---|---|---|---|
| Installed v0.5.1 | 1.00 | 98.38% | 92.45% | 73.50% | 81.89% | 0.32% | 26.50% | Existing installed policy |
| Frozen browser candidate | 0.99 | 98.42% | 89.60% | 77.50% | 83.11% | 0.47% | 22.50% | Frozen policy-development FPR rule |
| Development recall 80% | 0.97 | 98.00% | 80.00% | 80.00% | 80.00% | 1.05% | 20.00% | Policy-development recall target |
| Development recall 85% | 0.89 | 96.38% | 59.65% | 85.00% | 70.10% | 3.03% | 15.00% | Policy-development recall target |

The 10K and original frozen final-test labels were not used to calculate the
recall-target thresholds. They must not be evaluated on those same files and
then presented as new final results.
