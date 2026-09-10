# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4960 ± 0.0051 | 0.4979 ± 0.0026 | 0.9631 ± 0.0007 | 0.6565 ± 0.0024 | 0.7347 ± 0.0225 | 0.9712 ± 0.0096 | 0.0369 ± 0.0007 |
| LSTM | 0.5171 ± 0.0141 | 0.5091 ± 0.0077 | 0.9599 ± 0.0060 | 0.6653 ± 0.0069 | 0.8967 ± 0.0190 | 0.9257 ± 0.0279 | 0.0401 ± 0.0060 |
| RNN | 0.4843 ± 0.0052 | 0.4920 ± 0.0026 | 0.9611 ± 0.0025 | 0.6508 ± 0.0017 | 0.7756 ± 0.0420 | 0.9925 ± 0.0129 | 0.0389 ± 0.0025 |
| Random Forest | 0.4135 ± 0.0054 | 0.4414 ± 0.0031 | 0.6510 ± 0.0022 | 0.5261 ± 0.0018 | 0.5593 ± 0.0005 | 0.8240 ± 0.0123 | 0.3490 ± 0.0022 |
| Linear SVM | 0.2099 ± 0.0000 | 0.1789 ± 0.0000 | 0.1616 ± 0.0000 | 0.1698 ± 0.0000 | 0.1459 ± 0.0000 | 0.7418 ± 0.0000 | 0.8384 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
