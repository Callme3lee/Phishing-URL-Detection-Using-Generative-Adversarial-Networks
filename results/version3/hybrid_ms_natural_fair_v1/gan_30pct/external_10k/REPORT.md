# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 10000 rows (5000 legitimate, 5000 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.6247 ± 0.0360 | 0.9776 ± 0.0167 | 0.2561 ± 0.0786 | 0.4014 ± 0.0937 | 0.8764 ± 0.0102 | 0.8864 ± 0.0065 | 0.0067 ± 0.0068 | 0.7439 ± 0.0786 |
| Original-validation | 0.6060 ± 0.0633 | 0.5645 ± 0.0442 | 0.9595 ± 0.0059 | 0.7100 ± 0.0329 | 0.8764 ± 0.0102 | 0.8864 ± 0.0065 | 0.7475 ± 0.1322 | 0.0405 ± 0.0059 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
