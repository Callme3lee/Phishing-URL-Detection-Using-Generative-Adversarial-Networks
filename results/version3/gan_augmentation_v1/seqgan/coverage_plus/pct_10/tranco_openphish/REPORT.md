# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4250 ± 0.0282 | 0.4583 ± 0.0176 | 0.8350 ± 0.0650 | 0.5917 ± 0.0311 | 0.5756 ± 0.0282 | 0.9850 ± 0.0087 | 0.1650 ± 0.0650 |
| LSTM | 0.4608 ± 0.0052 | 0.4791 ± 0.0026 | 0.8967 ± 0.0029 | 0.6245 ± 0.0015 | 0.6435 ± 0.0556 | 0.9750 ± 0.0132 | 0.1033 ± 0.0029 |
| RNN | 0.4558 ± 0.0088 | 0.4765 ± 0.0046 | 0.8950 ± 0.0087 | 0.6219 ± 0.0059 | 0.6066 ± 0.1047 | 0.9833 ± 0.0104 | 0.1050 ± 0.0087 |
| Random Forest | 0.1225 ± 0.0025 | 0.1317 ± 0.0006 | 0.1350 ± 0.0000 | 0.1333 ± 0.0003 | 0.0950 ± 0.0025 | 0.8900 ± 0.0050 | 0.8650 ± 0.0000 |
| Linear SVM | 0.2725 ± 0.0000 | 0.1360 ± 0.0000 | 0.0850 ± 0.0000 | 0.1046 ± 0.0000 | 0.1152 ± 0.0000 | 0.5400 ± 0.0000 | 0.9150 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
