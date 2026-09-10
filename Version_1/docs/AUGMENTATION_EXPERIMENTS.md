# Controlled augmentation experiments

## Protocol

All experiments use the same hostname-separated validation and test sets. The
test set contains 252,430 real URLs and never contains generated samples.
Thresholds are selected on validation data by maximum F1.

Each augmentation condition adds exactly 48,000 malicious-labeled training
rows:

- `oversample`: reproducibly resampled real malicious training URLs;
- `lsgan`: unique, valid, non-memorized LSTM-LSGAN outputs;
- `wgan_gp`: unique, valid, non-memorized LSTM-WGAN-GP outputs.

The generated strings are *malicious-style synthetic URLs*. They are not opened,
resolved, or claimed to host real phishing pages.

## Filtered synthetic corpora

| Corpus | Raw | Accepted | Acceptance | Memorized removed |
|---|---:|---:|---:|---:|
| LSGAN | 50,000 | 49,026 | 98.05% | 40 |
| WGAN-GP | 50,000 | 48,643 | 97.29% | 27 |

Both accepted corpora have 100% uniqueness, 100% strict structural validity,
and 0% remaining exact training memorization. The controlled experiments use
the first reproducibly selected 48,000 rows from each.

## Full-data CNN results

| Training condition | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| Real only | 97.5210% | 96.5923% | 97.0544% | 99.5803% | 3.2592% |
| Real oversampling +48K | 97.1994% | 97.0084% | 97.1038% | 99.5814% | 3.7101% |
| LSGAN +48K | 97.2964% | 96.8625% | 97.0790% | 99.5740% | 3.5727% |
| WGAN-GP +48K | 97.3362% | 96.7791% | 97.0569% | 99.5730% | 3.5155% |

With over two million real training URLs, augmentation has little effect.
Oversampling provides the largest recall/F1 increase but raises false positives.
Neither GAN clearly outperforms the equal-size real oversampling control.

## Full-data LSTM validation

| Training condition | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| Real only | 97.3184% | 97.0243% | 97.1711% | 99.6071% | 3.5487% |
| Real oversampling +48K | 97.4806% | 96.7514% | 97.1146% | 99.5672% | 3.3191% |
| LSGAN +48K | 97.3052% | 96.7604% | 97.0320% | 99.5714% | 3.5570% |

The real-only LSTM remains best by F1. This confirms augmentation effects are
architecture-dependent and that abundant real data can make synthetic data
unnecessary.

## 10%-real-data CNN results (seed 42)

Only a reproducible, class-stratified 10% of the real training split is used
(205,789 rows). Validation and test sets remain full and unchanged.

| Training condition | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| 10% real only | 96.1066% | 95.7928% | 95.9494% | 99.2680% | 5.1511% |
| Real oversampling +48K | 96.4380% | 95.7970% | 96.1164% | 99.3165% | 4.6966% |
| LSGAN +48K | 96.6294% | 95.4031% | 96.0124% | 99.2699% | 4.4172% |
| WGAN-GP +48K | 97.2223% | 94.9128% | 96.0536% | 99.2528% | 3.5994% |

Under scarcity, both GAN corpora improve F1 and reduce false positives relative
to real-only training. Oversampling still has the highest F1 and recall.
WGAN-GP has the highest precision and lowest false-positive rate, showing a
conservative detection tradeoff rather than a universal performance win.

## Three-seed scarcity results

The complete 10%-real experiment was repeated with seeds 42, 43, and 44. The
table reports mean +/- sample standard deviation over the three independently
initialized classifier runs.

| Training condition | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| 10% real only | 96.3160 +/- 0.3252% | 95.9116 +/- 0.1574% | 96.1131 +/- 0.1601% | 99.3041 +/- 0.0313% | 4.8705 +/- 0.4474% |
| Real oversampling +48K | 96.8341 +/- 0.3911% | 95.7122 +/- 0.1105% | **96.2695 +/- 0.1419%** | **99.3412 +/- 0.0237%** | 4.1554 +/- 0.5344% |
| LSGAN +48K | 96.8563 +/- 0.2049% | 95.4601 +/- 0.0538% | 96.1531 +/- 0.1281% | 99.2995 +/- 0.0297% | 4.1130 +/- 0.2746% |
| WGAN-GP +48K | **97.1067 +/- 0.2908%** | 95.2006 +/- 0.4171% | 96.1433 +/- 0.0857% | 99.2788 +/- 0.0277% | **3.7669 +/- 0.4070%** |

Student-t 95% confidence intervals for the main outcomes are:

| Training condition | F1 95% CI | FPR 95% CI |
|---|---:|---:|
| 10% real only | 95.7153--96.5109% | 3.7590--5.9820% |
| Real oversampling +48K | 95.9170--96.6220% | 2.8278--5.4830% |
| LSGAN +48K | 95.8348--96.4714% | 3.4307--4.7952% |
| WGAN-GP +48K | 95.9306--96.3561% | 2.7558--4.7781% |

Oversampling has the highest mean F1 and ROC-AUC. WGAN-GP has the highest mean
precision and lowest mean false-positive rate, at the cost of recall. LSGAN is
between those operating points and has the smallest recall and FPR standard
deviations among the four conditions. Because all reported 95% intervals
overlap, these three runs do not establish a statistically significant win.
They establish repeatable operating tradeoffs that should be tested with more
runs if formal significance is required.

The machine-readable values, per-seed observations, sample standard deviations,
and confidence intervals are in `artifacts/multiseed_10pct_summary.json`.

## External-source frozen-threshold evaluation

To measure source shift, the full-data neural checkpoints were evaluated without
retraining or threshold selection on `Data sets/URL dataset.csv`. After removing
81 within-file duplicates, this test contains 450,095 URLs: 345,738 legitimate
and 104,357 phishing. No exact case-insensitive normalized URL overlap was found
against the 2M source dataset.

| Frozen model | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| CNN, real only | 47.6592% | 31.7037% | 38.0775% | 68.5163% | 10.5094% |
| LSTM, real only | 20.4201% | 81.4579% | 32.6544% | 46.0860% | 95.8191% |
| CNN, real oversampling +48K | 41.8154% | 41.9023% | 41.8588% | 68.3681% | 17.5989% |
| CNN, LSGAN +48K | **44.3296%** | 40.1947% | **42.1610%** | **72.5555%** | **15.2361%** |
| CNN, WGAN-GP +48K | 41.5703% | 37.3928% | 39.3710% | 68.0133% | 15.8640% |

LSGAN augmentation gives the strongest external CNN result and raises ROC-AUC
by 4.04 percentage points over the real-only CNN. Its F1 is also 4.08 points
higher. WGAN-GP does not reproduce that benefit, and the real-only LSTM fails
under this shift. Absolute external performance remains poor for every model,
showing that source-specific patterns and threshold transfer are major issues.

## Second external source: `new_data_urls.csv`

This file uses `status=1` for benign and `status=0` for malicious, which was
explicitly remapped to the project's `benign=0, malicious=1` target convention.
CSV-aware preparation removed 15,823 within-file duplicates and 179,141 exact
case-insensitive normalized overlaps with the 2M source dataset. The remaining
frozen external test has 627,046 URLs: 280,614 benign and 346,432 malicious.

| Frozen model | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| CNN, real only | 55.1108% | 53.6902% | 54.3912% | 58.2742% | 53.9895% |
| LSTM, real only | **60.7459%** | **83.1774%** | **70.2136%** | **63.7499%** | 66.3563% |
| CNN, real oversampling +48K | 55.9422% | **59.3098%** | **57.5768%** | **60.7317%** | 57.6657% |
| CNN, LSGAN +48K | 55.8527% | 58.1497% | 56.9780% | 59.6685% | 56.7434% |
| CNN, WGAN-GP +48K | **55.9886%** | 57.6026% | 56.7842% | 59.9237% | **55.9006%** |

The LSTM is strongest on this source, although its 66.36% false-positive rate
would be operationally unacceptable without adaptation. Among the controlled
CNN conditions, real oversampling has the best F1 and ROC-AUC. WGAN-GP again
operates more conservatively, with the lowest augmented-CNN FPR, while LSGAN
does not reproduce its first external-dataset advantage.

Together, the two external datasets show that no model is universally best:
LSGAN is the strongest CNN on `URL dataset.csv`, oversampling is the strongest
CNN on `new_data_urls.csv`, and the real-only LSTM changes from the worst result
on the first source to the best F1 on the second. This ranking instability is
direct evidence of dataset shift and motivates source-aware calibration or
training on multiple independently collected datasets.

## Interpretation limits

Three seeds are sufficient to show direction and variability but give wide
confidence intervals. The external files provide real cross-source tests, not
temporal tests: they have no trustworthy collection timestamps. Their source
label definitions should also be confirmed against the original dataset
documentation before publication. Therefore these results support limited
cross-source generalization observations, but they do not support a claim of
improved zero-day or future-phishing detection.
