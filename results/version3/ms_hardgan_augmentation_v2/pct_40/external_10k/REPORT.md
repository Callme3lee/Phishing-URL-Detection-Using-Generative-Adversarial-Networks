# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.5059 ± 0.0062 | 0.5031 ± 0.0032 | 0.9577 ± 0.0064 | 0.6597 ± 0.0030 | 0.7482 ± 0.0404 | 0.9460 ± 0.0148 | 0.0423 ± 0.0064 |
| LSTM | 0.5348 ± 0.0086 | 0.5188 ± 0.0048 | 0.9607 ± 0.0058 | 0.6738 ± 0.0041 | 0.9090 ± 0.0112 | 0.8910 ± 0.0188 | 0.0393 ± 0.0058 |
| RNN | 0.5193 ± 0.0328 | 0.5109 ± 0.0181 | 0.9503 ± 0.0069 | 0.6643 ± 0.0136 | 0.7967 ± 0.0028 | 0.9118 ± 0.0724 | 0.0497 ± 0.0069 |
| Random Forest | 0.4488 ± 0.0148 | 0.4627 ± 0.0100 | 0.6307 ± 0.0046 | 0.5337 ± 0.0054 | 0.5518 ± 0.0017 | 0.7330 ± 0.0334 | 0.3693 ± 0.0046 |
| Linear SVM | 0.1809 ± 0.0000 | 0.1691 ± 0.0000 | 0.1630 ± 0.0000 | 0.1660 ± 0.0000 | 0.1252 ± 0.0000 | 0.8012 ± 0.0000 | 0.8370 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
