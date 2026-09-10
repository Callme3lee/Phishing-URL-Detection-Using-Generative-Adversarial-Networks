# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4673 ± 0.0099 | 0.4816 ± 0.0054 | 0.8533 ± 0.0061 | 0.6157 ± 0.0057 | 0.5910 ± 0.0304 | 0.9187 ± 0.0162 | 0.1467 ± 0.0061 |
| LSTM | 0.5267 ± 0.0674 | 0.5168 ± 0.0401 | 0.9173 ± 0.0516 | 0.6603 ± 0.0371 | 0.7444 ± 0.1111 | 0.8640 ± 0.1380 | 0.0827 ± 0.0516 |
| RNN | 0.4900 ± 0.0087 | 0.4948 ± 0.0044 | 0.9533 ± 0.0046 | 0.6515 ± 0.0036 | 0.7598 ± 0.0705 | 0.9733 ± 0.0197 | 0.0467 ± 0.0046 |
| Random Forest | 0.2827 ± 0.0090 | 0.3176 ± 0.0093 | 0.3787 ± 0.0151 | 0.3455 ± 0.0117 | 0.2980 ± 0.0065 | 0.8133 ± 0.0061 | 0.6213 ± 0.0151 |
| Linear SVM | 0.2860 ± 0.0000 | 0.2512 ± 0.0000 | 0.2160 ± 0.0000 | 0.2323 ± 0.0000 | 0.2160 ± 0.0000 | 0.6440 ± 0.0000 | 0.7840 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
