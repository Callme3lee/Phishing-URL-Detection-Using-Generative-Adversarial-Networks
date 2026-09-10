# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4793 ± 0.0155 | 0.4890 ± 0.0082 | 0.9200 ± 0.0120 | 0.6386 ± 0.0098 | 0.5469 ± 0.0283 | 0.9613 ± 0.0197 | 0.0800 ± 0.0120 |
| LSTM | 0.4853 ± 0.0237 | 0.4919 ± 0.0132 | 0.9187 ± 0.0546 | 0.6406 ± 0.0246 | 0.6239 ± 0.0381 | 0.9480 ± 0.0080 | 0.0813 ± 0.0546 |
| RNN | 0.4807 ± 0.0031 | 0.4901 ± 0.0015 | 0.9573 ± 0.0023 | 0.6483 ± 0.0014 | 0.7853 ± 0.0586 | 0.9960 ± 0.0069 | 0.0427 ± 0.0023 |
| Random Forest | 0.2493 ± 0.0058 | 0.3025 ± 0.0049 | 0.3840 ± 0.0069 | 0.3384 ± 0.0058 | 0.2133 ± 0.0018 | 0.8853 ± 0.0046 | 0.6160 ± 0.0069 |
| Linear SVM | 0.2620 ± 0.0000 | 0.2468 ± 0.0000 | 0.2320 ± 0.0000 | 0.2392 ± 0.0000 | 0.2116 ± 0.0000 | 0.7080 ± 0.0000 | 0.7680 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
