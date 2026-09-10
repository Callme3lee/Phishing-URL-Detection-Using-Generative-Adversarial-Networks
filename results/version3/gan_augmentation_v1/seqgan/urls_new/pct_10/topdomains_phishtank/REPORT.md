# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4927 ± 0.0012 | 0.4962 ± 0.0006 | 0.9573 ± 0.0083 | 0.6536 ± 0.0015 | 0.6088 ± 0.0070 | 0.9720 ± 0.0106 | 0.0427 ± 0.0083 |
| LSTM | 0.4820 ± 0.0053 | 0.4908 ± 0.0027 | 0.9560 ± 0.0040 | 0.6486 ± 0.0032 | 0.7117 ± 0.0978 | 0.9920 ± 0.0069 | 0.0440 ± 0.0040 |
| RNN | 0.4800 ± 0.0020 | 0.4898 ± 0.0010 | 0.9587 ± 0.0046 | 0.6483 ± 0.0019 | 0.7744 ± 0.0329 | 0.9987 ± 0.0023 | 0.0413 ± 0.0046 |
| Random Forest | 0.2733 ± 0.0031 | 0.3330 ± 0.0025 | 0.4520 ± 0.0040 | 0.3835 ± 0.0031 | 0.2292 ± 0.0010 | 0.9053 ± 0.0023 | 0.5480 ± 0.0040 |
| Linear SVM | 0.2980 ± 0.0000 | 0.2735 ± 0.0000 | 0.2440 ± 0.0000 | 0.2579 ± 0.0000 | 0.2560 ± 0.0000 | 0.6480 ± 0.0000 | 0.7560 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
