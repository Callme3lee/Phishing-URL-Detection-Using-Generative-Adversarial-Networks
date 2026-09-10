# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4508 ± 0.0052 | 0.4738 ± 0.0029 | 0.8900 ± 0.0087 | 0.6184 ± 0.0045 | 0.5697 ± 0.0388 | 0.9883 ± 0.0029 | 0.1100 ± 0.0087 |
| LSTM | 0.4508 ± 0.0076 | 0.4741 ± 0.0042 | 0.9017 ± 0.0153 | 0.6215 ± 0.0073 | 0.5926 ± 0.0865 | 1.0000 ± 0.0000 | 0.0983 ± 0.0153 |
| RNN | 0.4458 ± 0.0072 | 0.4713 ± 0.0040 | 0.8883 ± 0.0126 | 0.6158 ± 0.0063 | 0.7545 ± 0.0574 | 0.9967 ± 0.0058 | 0.1117 ± 0.0126 |
| Random Forest | 0.0917 ± 0.0014 | 0.1277 ± 0.0003 | 0.1400 ± 0.0000 | 0.1335 ± 0.0002 | 0.0641 ± 0.0021 | 0.9567 ± 0.0029 | 0.8600 ± 0.0000 |
| Linear SVM | 0.2475 ± 0.0000 | 0.1314 ± 0.0000 | 0.0900 ± 0.0000 | 0.1068 ± 0.0000 | 0.1135 ± 0.0000 | 0.5950 ± 0.0000 | 0.9100 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
