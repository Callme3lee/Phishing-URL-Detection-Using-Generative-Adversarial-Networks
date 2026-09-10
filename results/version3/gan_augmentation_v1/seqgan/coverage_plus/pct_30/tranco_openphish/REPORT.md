# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4158 ± 0.0138 | 0.4524 ± 0.0089 | 0.8017 ± 0.0321 | 0.5783 ± 0.0156 | 0.5407 ± 0.0316 | 0.9700 ± 0.0050 | 0.1983 ± 0.0321 |
| LSTM | 0.4792 ± 0.0422 | 0.4888 ± 0.0234 | 0.8750 ± 0.0391 | 0.6270 ± 0.0249 | 0.6364 ± 0.1379 | 0.9167 ± 0.0802 | 0.1250 ± 0.0391 |
| RNN | 0.4492 ± 0.0029 | 0.4730 ± 0.0013 | 0.8900 ± 0.0050 | 0.6177 ± 0.0007 | 0.7285 ± 0.0452 | 0.9917 ± 0.0104 | 0.1100 ± 0.0050 |
| Random Forest | 0.1150 ± 0.0066 | 0.1137 ± 0.0073 | 0.1133 ± 0.0076 | 0.1135 ± 0.0075 | 0.0710 ± 0.0013 | 0.8833 ± 0.0076 | 0.8867 ± 0.0076 |
| Linear SVM | 0.2450 ± 0.0000 | 0.1136 ± 0.0000 | 0.0750 ± 0.0000 | 0.0904 ± 0.0000 | 0.0896 ± 0.0000 | 0.5850 ± 0.0000 | 0.9250 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
