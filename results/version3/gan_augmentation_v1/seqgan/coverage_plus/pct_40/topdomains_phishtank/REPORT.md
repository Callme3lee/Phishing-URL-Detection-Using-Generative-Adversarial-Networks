# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4653 ± 0.0170 | 0.4802 ± 0.0097 | 0.8387 ± 0.0162 | 0.6107 ± 0.0121 | 0.5809 ± 0.0440 | 0.9080 ± 0.0183 | 0.1613 ± 0.0162 |
| LSTM | 0.4860 ± 0.0250 | 0.4916 ± 0.0155 | 0.8387 ± 0.0954 | 0.6189 ± 0.0323 | 0.6282 ± 0.0554 | 0.8667 ± 0.0885 | 0.1613 ± 0.0954 |
| RNN | 0.4933 ± 0.0172 | 0.4966 ± 0.0090 | 0.9467 ± 0.0101 | 0.6514 ± 0.0076 | 0.7759 ± 0.0212 | 0.9600 ± 0.0382 | 0.0533 ± 0.0101 |
| Random Forest | 0.2760 ± 0.0072 | 0.2921 ± 0.0060 | 0.3147 ± 0.0061 | 0.3030 ± 0.0059 | 0.2841 ± 0.0027 | 0.7627 ± 0.0101 | 0.6853 ± 0.0061 |
| Linear SVM | 0.2560 ± 0.0000 | 0.2277 ± 0.0000 | 0.2040 ± 0.0000 | 0.2152 ± 0.0000 | 0.1978 ± 0.0000 | 0.6920 ± 0.0000 | 0.7960 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
