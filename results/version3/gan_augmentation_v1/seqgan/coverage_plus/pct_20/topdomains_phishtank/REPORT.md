# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4740 ± 0.0262 | 0.4858 ± 0.0144 | 0.8653 ± 0.0129 | 0.6221 ± 0.0097 | 0.6014 ± 0.0300 | 0.9173 ± 0.0621 | 0.1347 ± 0.0129 |
| LSTM | 0.5153 ± 0.0340 | 0.5097 ± 0.0206 | 0.9240 ± 0.0487 | 0.6561 ± 0.0053 | 0.7594 ± 0.0195 | 0.8933 ± 0.1157 | 0.0760 ± 0.0487 |
| RNN | 0.4887 ± 0.0081 | 0.4941 ± 0.0041 | 0.9533 ± 0.0083 | 0.6509 ± 0.0043 | 0.7815 ± 0.0811 | 0.9760 ± 0.0174 | 0.0467 ± 0.0083 |
| Random Forest | 0.3047 ± 0.0012 | 0.3374 ± 0.0017 | 0.4053 ± 0.0046 | 0.3683 ± 0.0029 | 0.3162 ± 0.0035 | 0.7960 ± 0.0040 | 0.5947 ± 0.0046 |
| Linear SVM | 0.2900 ± 0.0000 | 0.2581 ± 0.0000 | 0.2240 ± 0.0000 | 0.2398 ± 0.0000 | 0.2371 ± 0.0000 | 0.6440 ± 0.0000 | 0.7760 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
