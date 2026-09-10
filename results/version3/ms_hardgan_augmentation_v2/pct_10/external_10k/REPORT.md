# Version 3 External Evaluation

This frozen evaluation contains 5000 legitimate and 5000 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4927 ± 0.0023 | 0.4962 ± 0.0012 | 0.9653 ± 0.0016 | 0.6555 ± 0.0009 | 0.7642 ± 0.0301 | 0.9799 ± 0.0056 | 0.0347 ± 0.0016 |
| LSTM | 0.5179 ± 0.0453 | 0.5105 ± 0.0254 | 0.9587 ± 0.0110 | 0.6659 ± 0.0188 | 0.8958 ± 0.0118 | 0.9229 ± 0.1013 | 0.0413 ± 0.0110 |
| RNN | 0.4806 ± 0.0007 | 0.4901 ± 0.0004 | 0.9599 ± 0.0017 | 0.6489 ± 0.0005 | 0.8107 ± 0.0190 | 0.9986 ± 0.0023 | 0.0401 ± 0.0017 |
| Random Forest | 0.4104 ± 0.0016 | 0.4402 ± 0.0007 | 0.6588 ± 0.0033 | 0.5277 ± 0.0009 | 0.5766 ± 0.0010 | 0.8379 ± 0.0060 | 0.3412 ± 0.0033 |
| Linear SVM | 0.2235 ± 0.0000 | 0.1951 ± 0.0000 | 0.1770 ± 0.0000 | 0.1856 ± 0.0000 | 0.1753 ± 0.0000 | 0.7300 ± 0.0000 | 0.8230 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate rows are roots while phishing rows often include paths and queries.
- The balanced class prevalence does not represent browser traffic.
- Only PhishTank rows explicitly timestamped after calibration count as temporal confirmation candidates.
- The full result is an identity-disjoint source-transfer test, not a wholly post-calibration temporal test.
