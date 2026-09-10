# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4458 ± 0.0014 | 0.4712 ± 0.0010 | 0.8850 ± 0.0087 | 0.6149 ± 0.0029 | 0.5737 ± 0.0183 | 0.9933 ± 0.0058 | 0.1150 ± 0.0087 |
| LSTM | 0.4317 ± 0.0361 | 0.4620 ± 0.0225 | 0.8517 ± 0.0881 | 0.5988 ± 0.0410 | 0.6857 ± 0.0874 | 0.9883 ± 0.0161 | 0.1483 ± 0.0881 |
| RNN | 0.4467 ± 0.0014 | 0.4718 ± 0.0009 | 0.8917 ± 0.0058 | 0.6171 ± 0.0022 | 0.7534 ± 0.0180 | 0.9983 ± 0.0029 | 0.1083 ± 0.0058 |
| Random Forest | 0.0975 ± 0.0043 | 0.1484 ± 0.0064 | 0.1700 ± 0.0087 | 0.1585 ± 0.0074 | 0.0715 ± 0.0004 | 0.9750 ± 0.0000 | 0.8300 ± 0.0087 |
| Linear SVM | 0.3325 ± 0.0000 | 0.1810 ± 0.0000 | 0.0950 ± 0.0000 | 0.1246 ± 0.0000 | 0.1286 ± 0.0000 | 0.4300 ± 0.0000 | 0.9050 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
