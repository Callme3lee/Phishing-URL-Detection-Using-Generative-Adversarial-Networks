# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 10000 rows (5000 legitimate, 5000 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.6577 ± 0.0136 | 0.9883 ± 0.0037 | 0.3191 ± 0.0268 | 0.4820 ± 0.0306 | 0.9351 ± 0.0081 | 0.9487 ± 0.0089 | 0.0037 ± 0.0010 | 0.6809 ± 0.0268 |
| Original-validation | 0.5347 ± 0.0367 | 0.5192 ± 0.0209 | 0.9681 ± 0.0007 | 0.6757 ± 0.0176 | 0.9351 ± 0.0081 | 0.9487 ± 0.0089 | 0.8987 ± 0.0733 | 0.0319 ± 0.0007 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
