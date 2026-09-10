# Version 3 External Evaluation

This frozen evaluation contains 250 legitimate and 250 phishing proxy-labelled rows.

Sources: legitimate: local Top-Domain ranked-list root proxies; phishing: PhishTank verified-online hourly feed.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4873 ± 0.0042 | 0.4933 ± 0.0021 | 0.9333 ± 0.0185 | 0.6454 ± 0.0028 | 0.5427 ± 0.0250 | 0.9587 ± 0.0266 | 0.0667 ± 0.0185 |
| LSTM | 0.4900 ± 0.0212 | 0.4944 ± 0.0118 | 0.9267 ± 0.0577 | 0.6446 ± 0.0240 | 0.6905 ± 0.0442 | 0.9467 ± 0.0180 | 0.0733 ± 0.0577 |
| RNN | 0.4740 ± 0.0104 | 0.4866 ± 0.0056 | 0.9453 ± 0.0254 | 0.6424 ± 0.0108 | 0.7656 ± 0.0167 | 0.9973 ± 0.0046 | 0.0547 ± 0.0254 |
| Random Forest | 0.2293 ± 0.0101 | 0.2773 ± 0.0112 | 0.3373 ± 0.0180 | 0.3044 ± 0.0141 | 0.2131 ± 0.0039 | 0.8787 ± 0.0023 | 0.6627 ± 0.0180 |
| Linear SVM | 0.2560 ± 0.0000 | 0.2371 ± 0.0000 | 0.2200 ± 0.0000 | 0.2282 ± 0.0000 | 0.1938 ± 0.0000 | 0.7080 ± 0.0000 | 0.7800 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
