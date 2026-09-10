# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4821 ± 0.0024 | 0.4908 ± 0.0013 | 0.9605 ± 0.0087 | 0.6497 ± 0.0031 | 0.5730 ± 0.0355 | 0.9963 ± 0.0039 | 0.0395 ± 0.0087 |
| LSTM | 0.4791 ± 0.0141 | 0.4890 ± 0.0077 | 0.9427 ± 0.0391 | 0.6439 ± 0.0157 | 0.8323 ± 0.0600 | 0.9845 ± 0.0140 | 0.0573 ± 0.0391 |
| RNN | 0.4805 ± 0.0004 | 0.4900 ± 0.0002 | 0.9601 ± 0.0020 | 0.6489 ± 0.0006 | 0.7771 ± 0.0081 | 0.9991 ± 0.0013 | 0.0399 ± 0.0020 |
| Random Forest | 0.3285 ± 0.0059 | 0.3859 ± 0.0048 | 0.5799 ± 0.0117 | 0.4634 ± 0.0072 | 0.2482 ± 0.0013 | 0.9229 ± 0.0004 | 0.4201 ± 0.0117 |
| Linear SVM | 0.3469 ± 0.0000 | 0.2587 ± 0.0000 | 0.1642 ± 0.0000 | 0.2009 ± 0.0000 | 0.2342 ± 0.0000 | 0.4704 ± 0.0000 | 0.8358 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
