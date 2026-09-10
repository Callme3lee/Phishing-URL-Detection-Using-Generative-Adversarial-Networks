# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4973 ± 0.0042 | 0.4986 ± 0.0021 | 0.9653 ± 0.0061 | 0.6576 ± 0.0032 | 0.6991 ± 0.0014 | 0.9707 ± 0.0046 | 0.0347 ± 0.0061 |
| LSTM | 0.4893 ± 0.0129 | 0.4946 ± 0.0066 | 0.9613 ± 0.0023 | 0.6531 ± 0.0063 | 0.8503 ± 0.0289 | 0.9827 ± 0.0234 | 0.0387 ± 0.0023 |
| RNN | 0.4780 ± 0.0035 | 0.4887 ± 0.0018 | 0.9547 ± 0.0092 | 0.6465 ± 0.0037 | 0.7166 ± 0.1215 | 0.9987 ± 0.0023 | 0.0453 ± 0.0092 |
| Random Forest | 0.3373 ± 0.0012 | 0.3475 ± 0.0018 | 0.3707 ± 0.0061 | 0.3587 ± 0.0037 | 0.3074 ± 0.0024 | 0.6960 ± 0.0069 | 0.6293 ± 0.0061 |
| Linear SVM | 0.5720 ± 0.0000 | 0.9737 ± 0.0000 | 0.1480 ± 0.0000 | 0.2569 ± 0.0000 | 0.3246 ± 0.0000 | 0.0040 ± 0.0000 | 0.8520 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
