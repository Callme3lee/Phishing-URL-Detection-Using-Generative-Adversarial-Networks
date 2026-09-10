# Large External Evaluation — Frozen Calibrated Hybrid Ensemble

Dataset: `10,000` URLs (`5,000` legitimate + `5,000` phishing).

Frozen threshold: `0.98795575`. Neither the model, ensemble, nor threshold was changed after seeing this dataset.

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9177 | 0.9804 | 0.8524 | 0.9120 | 0.9663 | 0.9749 | 0.0170 | 0.1476 |

## Source diagnostics

| Source | Rows | Positive recall / legitimate specificity | Error rate | Mean score |
|---|---:|---:|---:|---:|
| OpenPhish current community feed | 246 | 0.6870 | 0.3130 | 0.9083 |
| PhishTank verified-online | 4,754 | 0.8610 | 0.1390 | 0.9542 |
| Tranco current root proxy | 5,000 | 0.9830 | 0.0170 | 0.3008 |

## Temporal phishing diagnostic

Of the PhishTank rows, `121` were explicitly verified after the calibration freeze. Recall on this phishing-only subset is `0.8430` and FNR is `0.1570`. Because it contains no legitimate rows, it cannot provide FPR or the other two-class metrics.

## Interpretation guard

This is a large identity-disjoint source-transfer evaluation. It is not wholly temporal because the live feeds retain older URLs. Tranco membership is a legitimate proxy and the balanced prevalence is not browser prevalence.
