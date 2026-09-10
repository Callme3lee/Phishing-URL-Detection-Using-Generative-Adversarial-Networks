# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4947 ± 0.0058 | 0.4972 ± 0.0030 | 0.9480 ± 0.0000 | 0.6523 ± 0.0026 | 0.5772 ± 0.0185 | 0.9587 ± 0.0115 | 0.0520 ± 0.0000 |
| LSTM | 0.4847 ± 0.0012 | 0.4921 ± 0.0007 | 0.9507 ± 0.0231 | 0.6484 ± 0.0059 | 0.6336 ± 0.0403 | 0.9813 ± 0.0220 | 0.0493 ± 0.0231 |
| RNN | 0.4807 ± 0.0031 | 0.4901 ± 0.0016 | 0.9547 ± 0.0061 | 0.6477 ± 0.0024 | 0.7119 ± 0.0479 | 0.9933 ± 0.0061 | 0.0453 ± 0.0061 |
| Random Forest | 0.2527 ± 0.0023 | 0.3128 ± 0.0027 | 0.4133 ± 0.0061 | 0.3561 ± 0.0039 | 0.2158 ± 0.0014 | 0.9080 ± 0.0040 | 0.5867 ± 0.0061 |
| Linear SVM | 0.2860 ± 0.0000 | 0.2601 ± 0.0000 | 0.2320 ± 0.0000 | 0.2452 ± 0.0000 | 0.2330 ± 0.0000 | 0.6600 ± 0.0000 | 0.7680 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
