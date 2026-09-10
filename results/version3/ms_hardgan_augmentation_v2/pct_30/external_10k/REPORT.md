# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4984 ± 0.0079 | 0.4992 ± 0.0041 | 0.9652 ± 0.0019 | 0.6580 ± 0.0032 | 0.7625 ± 0.0095 | 0.9684 ± 0.0172 | 0.0348 ± 0.0019 |
| LSTM | 0.5165 ± 0.0237 | 0.5089 ± 0.0129 | 0.9603 ± 0.0064 | 0.6653 ± 0.0119 | 0.9026 ± 0.0277 | 0.9273 ± 0.0442 | 0.0397 ± 0.0064 |
| RNN | 0.4943 ± 0.0143 | 0.4972 ± 0.0074 | 0.9607 ± 0.0019 | 0.6552 ± 0.0060 | 0.8261 ± 0.0396 | 0.9720 ± 0.0304 | 0.0393 ± 0.0019 |
| Random Forest | 0.4270 ± 0.0081 | 0.4491 ± 0.0048 | 0.6433 ± 0.0072 | 0.5289 ± 0.0028 | 0.5490 ± 0.0019 | 0.7893 ± 0.0216 | 0.3567 ± 0.0072 |
| Linear SVM | 0.1979 ± 0.0000 | 0.1644 ± 0.0000 | 0.1480 ± 0.0000 | 0.1558 ± 0.0000 | 0.1292 ± 0.0000 | 0.7522 ± 0.0000 | 0.8520 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
