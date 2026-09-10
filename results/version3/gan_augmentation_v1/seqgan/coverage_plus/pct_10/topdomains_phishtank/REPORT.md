# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4753 ± 0.0061 | 0.4867 ± 0.0031 | 0.8987 ± 0.0101 | 0.6314 ± 0.0019 | 0.6514 ± 0.0158 | 0.9480 ± 0.0208 | 0.1013 ± 0.0101 |
| LSTM | 0.4913 ± 0.0090 | 0.4955 ± 0.0046 | 0.9507 ± 0.0061 | 0.6515 ± 0.0033 | 0.8271 ± 0.0139 | 0.9680 ± 0.0223 | 0.0493 ± 0.0061 |
| RNN | 0.4907 ± 0.0133 | 0.4952 ± 0.0068 | 0.9547 ± 0.0046 | 0.6521 ± 0.0062 | 0.7803 ± 0.0122 | 0.9733 ± 0.0260 | 0.0453 ± 0.0046 |
| Random Forest | 0.3180 ± 0.0035 | 0.3567 ± 0.0031 | 0.4533 ± 0.0101 | 0.3993 ± 0.0056 | 0.3644 ± 0.0030 | 0.8173 ± 0.0115 | 0.5467 ± 0.0101 |
| Linear SVM | 0.3280 ± 0.0000 | 0.2913 ± 0.0000 | 0.2400 ± 0.0000 | 0.2632 ± 0.0000 | 0.2611 ± 0.0000 | 0.5840 ± 0.0000 | 0.7600 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
