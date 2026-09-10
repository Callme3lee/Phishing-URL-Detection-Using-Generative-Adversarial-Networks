# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 10000 rows (5000 legitimate, 5000 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.6254 ± 0.0600 | 0.9541 ± 0.0295 | 0.2666 ± 0.1355 | 0.4033 ± 0.1590 | 0.8469 ± 0.0210 | 0.8618 ± 0.0132 | 0.0158 ± 0.0156 | 0.7334 ± 0.1355 |
| Original-validation | 0.5197 ± 0.0140 | 0.5105 ± 0.0076 | 0.9600 ± 0.0071 | 0.6665 ± 0.0079 | 0.8469 ± 0.0210 | 0.8618 ± 0.0132 | 0.9207 ± 0.0223 | 0.0400 ± 0.0071 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
