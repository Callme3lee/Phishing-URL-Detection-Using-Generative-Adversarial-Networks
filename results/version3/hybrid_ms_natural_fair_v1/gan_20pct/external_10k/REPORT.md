# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 10000 rows (5000 legitimate, 5000 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.6239 ± 0.0395 | 0.9879 ± 0.0019 | 0.2510 ± 0.0802 | 0.3960 ± 0.0986 | 0.8943 ± 0.0177 | 0.9110 ± 0.0149 | 0.0031 ± 0.0013 | 0.7490 ± 0.0802 |
| Original-validation | 0.5839 ± 0.0346 | 0.5484 ± 0.0213 | 0.9607 ± 0.0013 | 0.6980 ± 0.0177 | 0.8943 ± 0.0177 | 0.9110 ± 0.0149 | 0.7929 ± 0.0680 | 0.0393 ± 0.0013 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
