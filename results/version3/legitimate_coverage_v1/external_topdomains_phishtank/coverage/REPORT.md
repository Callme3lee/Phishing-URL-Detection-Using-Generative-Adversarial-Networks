# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4953 ± 0.0061 | 0.4976 ± 0.0032 | 0.9520 ± 0.0069 | 0.6535 ± 0.0043 | 0.7192 ± 0.0184 | 0.9613 ± 0.0061 | 0.0480 ± 0.0069 |
| LSTM | 0.5480 ± 0.0104 | 0.5265 ± 0.0060 | 0.9547 ± 0.0023 | 0.6787 ± 0.0047 | 0.9087 ± 0.0125 | 0.8587 ± 0.0220 | 0.0453 ± 0.0023 |
| RNN | 0.4860 ± 0.0000 | 0.4928 ± 0.0000 | 0.9587 ± 0.0023 | 0.6510 ± 0.0005 | 0.8071 ± 0.0732 | 0.9867 ± 0.0023 | 0.0413 ± 0.0023 |
| Random Forest | 0.4187 ± 0.0114 | 0.4322 ± 0.0074 | 0.5173 ± 0.0061 | 0.4709 ± 0.0019 | 0.4678 ± 0.0038 | 0.6800 ± 0.0288 | 0.4827 ± 0.0061 |
| Linear SVM | 0.3700 ± 0.0000 | 0.3204 ± 0.0000 | 0.2320 ± 0.0000 | 0.2691 ± 0.0000 | 0.2898 ± 0.0000 | 0.4920 ± 0.0000 | 0.7680 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
