# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4342 ± 0.0076 | 0.4642 ± 0.0046 | 0.8550 ± 0.0200 | 0.6017 ± 0.0088 | 0.5163 ± 0.0230 | 0.9867 ± 0.0058 | 0.1450 ± 0.0200 |
| LSTM | 0.4492 ± 0.0052 | 0.4726 ± 0.0029 | 0.8783 ± 0.0161 | 0.6145 ± 0.0057 | 0.4694 ± 0.0861 | 0.9800 ± 0.0150 | 0.1217 ± 0.0161 |
| RNN | 0.4458 ± 0.0063 | 0.4713 ± 0.0034 | 0.8883 ± 0.0076 | 0.6158 ± 0.0047 | 0.7694 ± 0.0180 | 0.9967 ± 0.0058 | 0.1117 ± 0.0076 |
| Random Forest | 0.0767 ± 0.0014 | 0.1092 ± 0.0024 | 0.1183 ± 0.0029 | 0.1136 ± 0.0026 | 0.0598 ± 0.0005 | 0.9650 ± 0.0000 | 0.8817 ± 0.0029 |
| Linear SVM | 0.1850 ± 0.0000 | 0.1062 ± 0.0000 | 0.0850 ± 0.0000 | 0.0944 ± 0.0000 | 0.0889 ± 0.0000 | 0.7150 ± 0.0000 | 0.9150 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
