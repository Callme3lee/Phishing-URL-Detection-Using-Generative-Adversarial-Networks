# Hybrid CNN + PSL Lexical + CORAL Experiment

## Decision

The strongest immediate response to the measured source shift was a hybrid URL model, not GAN augmentation. The original Version 3 baselines and splits remain unchanged. This added condition combines:

- a character CNN, which learns local character sequences directly from the URL;
- 35 normalized, Public-Suffix-List-aware lexical and structural features;
- source- and class-balanced sampling; and
- class-conditional CORAL loss, which encourages representations of the same class from different sources to have similar covariance.

No GAN-generated URL is used here. The purpose is to establish a stronger real-data classifier before evaluating augmentation.

The design follows the general hybrid character/word principle demonstrated by [URLNet](https://arxiv.org/abs/1802.03162), uses the [Public Suffix List](https://publicsuffix.org/) to separate URL components without exposing suffix identity, and applies the covariance-alignment idea from [Deep CORAL](https://arxiv.org/abs/1607.01719).

## Frozen data

The training artifact contains 363,692 unique real URLs:

| Source group | Label | Unique stored rows | Target sampling share |
|---|---|---:|---:|
| URLs_New | Legitimate | 276,400 | 35% |
| Audited outside coverage | Legitimate | 3,012 | 15% |
| URLs_New | Phishing | 82,219 | 30% |
| PhishTank | Phishing | 2,000 | 15% |
| OpenPhish | Phishing | 61 | 5% |

Weighted sampling draws 363,692 rows per epoch with replacement. It changes source exposure, not the frozen stored row count. The small OpenPhish group is necessarily repeated and is a stated limitation.

The original hostname-held-out validation and test sets are unchanged. A separate 709-row domain-validation set contains 389 outside legitimate, 300 PhishTank, and 20 OpenPhish URLs. Every URL, hostname, and 128-character encoding from both frozen external evaluations was excluded from added training and domain-validation rows.

## Features and model

The character branch uses a 48-dimensional embedding and 128 convolution channels for kernel sizes 3, 4, and 5. The lexical branch maps 35 robust-scaled features through 64 and 32 units. The two branches are fused through a 128-unit representation.

The lexical vector contains only numeric properties such as component lengths and ratios, digit/special-character ratios, path depth, query count, entropies, IP/punycode indicators, suspicious-token ratio, and encoding/run statistics. Exact hostname, domain, suffix, and source identity are not features. A bundled PSL snapshot is used without network access. Robust scaling is fitted on training data only.

CORAL is class-conditional. It aligns URLs_New legitimate with outside legitimate, and aligns the three phishing sources with one another. The fixed CORAL weight is 0.01. Models are trained with AdamW, mixed precision, early stopping, and seeds 42/43/44 on the RTX 3060 Laptop GPU.

All three runs selected epoch 1 because later epochs improved the familiar validation distribution while degrading cross-source validation. Each run processed 363,692 sampled rows per epoch and took about 36–39 seconds through early stopping.

## Internal result

At the deliberately conservative robust threshold, the unchanged internal test result is:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9180 | 0.9998 | 0.6092 | 0.7551 | 0.9956 | 0.9906 | 0.00003 | 0.3908 |

These means are across seeds. The low recall is caused by the conservative cross-source threshold, not weak ranking: ROC-AUC and PR-AUC remain near 0.996 and 0.991.

## External result

The earlier frozen external sets informed this model design, so these results are exploratory. They are not a fresh confirmatory test.

### Comparable original-validation threshold

| External set | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-Domain / PhishTank (500) | 0.8727 | 0.8143 | 0.9667 | 0.8838 | 0.9704 | 0.2213 | 0.0333 |
| Tranco / OpenPhish (400) | 0.8300 | 0.7848 | 0.9133 | 0.8436 | 0.9276 | 0.2533 | 0.0867 |

For comparison, the original LSTM FPR was 0.9867 and 0.9883 on these sets. Its ROC-AUC was 0.8431 and 0.6857. The hybrid therefore substantially reduces—but does not eliminate—the source shift.

### Conservative robust threshold

| External set | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-Domain / PhishTank (500) | 0.7527 | 0.9980 | 0.5067 | 0.6601 | 0.9704 | 0.0013 | 0.4933 |
| Tranco / OpenPhish (400) | 0.7225 | 1.0000 | 0.4450 | 0.6093 | 0.9276 | 0.0000 | 0.5550 |

The robust threshold nearly removes false alarms but sacrifices too much phishing recall. This is an operating-point calibration issue. It must not be solved by selecting a threshold on either external test.

## Conclusion and next step

The hybrid is the strongest Version 3 response to distribution shift so far. The original model had both representation and calibration failures. Multi-source training plus PSL-aware features and CORAL greatly improves ranking and legitimate rejection; threshold calibration remains unresolved.

This experiment was followed by a separate, identity-disjoint calibration stage. The resulting three-seed ensemble threshold reduces exploratory external FPR to 0.8–2.0%; see `docs/CALIBRATED_HYBRID_ENSEMBLE.md`. A genuinely later untouched snapshot is still required for confirmation. GAN augmentation should resume only after this real-data hybrid becomes the reference baseline.

## Reproduce

From `Version_3` with `PYTHONPATH=src`:

```bash
python -m phishing_gans_v3.prepare_hybrid_domain_data
python -m phishing_gans_v3.train_hybrid_domain
python -m phishing_gans_v3.evaluate_hybrid_external \
  --dataset data/external/2026-08-11/external_urls.csv \
  --collection-report data/external/2026-08-11/collection_report.json \
  --output-dir artifacts/results/hybrid_cnn_coral_v1/external_topdomains_phishtank
python -m phishing_gans_v3.evaluate_hybrid_external \
  --dataset data/external/2026-08-11-openphish-tranco/external_urls.csv \
  --collection-report data/external/2026-08-11-openphish-tranco/collection_report.json \
  --output-dir artifacts/results/hybrid_cnn_coral_v1/external_openphish_tranco
python -m phishing_gans_v3.compare_hybrid_external
```

The preparation command refuses to overwrite the frozen experiment unless explicitly requested.
