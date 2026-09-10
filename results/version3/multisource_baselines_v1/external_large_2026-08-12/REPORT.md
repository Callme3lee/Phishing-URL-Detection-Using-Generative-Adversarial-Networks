# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4887 ± 0.0006 | 0.4942 ± 0.0003 | 0.9662 ± 0.0009 | 0.6539 ± 0.0001 | 0.7950 ± 0.0161 | 0.9889 ± 0.0021 | 0.0338 ± 0.0009 |
| LSTM | 0.5163 ± 0.0269 | 0.5089 ± 0.0146 | 0.9641 ± 0.0005 | 0.6660 ± 0.0124 | 0.9309 ± 0.0057 | 0.9315 ± 0.0539 | 0.0359 ± 0.0005 |
| RNN | 0.4831 ± 0.0021 | 0.4914 ± 0.0011 | 0.9614 ± 0.0008 | 0.6504 ± 0.0008 | 0.8725 ± 0.0131 | 0.9951 ± 0.0048 | 0.0386 ± 0.0008 |
| Random Forest | 0.4834 ± 0.0112 | 0.4881 ± 0.0079 | 0.6759 ± 0.0018 | 0.5668 ± 0.0053 | 0.6298 ± 0.0020 | 0.7091 ± 0.0224 | 0.3241 ± 0.0018 |
| Linear SVM | 0.3464 ± 0.0000 | 0.2670 ± 0.0000 | 0.1760 ± 0.0000 | 0.2122 ± 0.0000 | 0.2499 ± 0.0000 | 0.4832 ± 0.0000 | 0.8240 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
