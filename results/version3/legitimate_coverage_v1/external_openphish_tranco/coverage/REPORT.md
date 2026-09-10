# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4283 ± 0.0095 | 0.4602 ± 0.0056 | 0.8283 ± 0.0176 | 0.5916 ± 0.0090 | 0.5814 ± 0.0248 | 0.9717 ± 0.0058 | 0.1717 ± 0.0176 |
| LSTM | 0.5075 ± 0.0218 | 0.5044 ± 0.0122 | 0.8983 ± 0.0029 | 0.6460 ± 0.0096 | 0.8369 ± 0.0124 | 0.8833 ± 0.0454 | 0.1017 ± 0.0029 |
| RNN | 0.4575 ± 0.0066 | 0.4775 ± 0.0034 | 0.9000 ± 0.0100 | 0.6239 ± 0.0042 | 0.7763 ± 0.0399 | 0.9850 ± 0.0150 | 0.1000 ± 0.0100 |
| Random Forest | 0.1875 ± 0.0109 | 0.1680 ± 0.0035 | 0.1583 ± 0.0104 | 0.1630 ± 0.0071 | 0.1371 ± 0.0029 | 0.7833 ± 0.0321 | 0.8417 ± 0.0104 |
| Linear SVM | 0.3300 ± 0.0000 | 0.1792 ± 0.0000 | 0.0950 ± 0.0000 | 0.1242 ± 0.0000 | 0.1291 ± 0.0000 | 0.4350 ± 0.0000 | 0.9050 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
