# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 10000 rows (5000 legitimate, 5000 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.6452 ± 0.0325 | 0.9873 ± 0.0029 | 0.2941 ± 0.0650 | 0.4506 ± 0.0775 | 0.9077 ± 0.0195 | 0.9233 ± 0.0182 | 0.0037 ± 0.0005 | 0.7059 ± 0.0650 |
| Original-validation | 0.6184 ± 0.1081 | 0.5766 ± 0.0693 | 0.9607 ± 0.0083 | 0.7188 ± 0.0537 | 0.9077 ± 0.0195 | 0.9233 ± 0.0182 | 0.7239 ± 0.2238 | 0.0393 ± 0.0083 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
