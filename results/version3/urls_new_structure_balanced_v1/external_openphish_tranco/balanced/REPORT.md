# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4525 ± 0.0000 | 0.4746 ± 0.0002 | 0.8883 ± 0.0076 | 0.6187 ± 0.0020 | 0.5728 ± 0.0272 | 0.9833 ± 0.0076 | 0.1117 ± 0.0076 |
| LSTM | 0.4550 ± 0.0066 | 0.4763 ± 0.0035 | 0.9033 ± 0.0076 | 0.6237 ± 0.0047 | 0.5952 ± 0.0880 | 0.9933 ± 0.0076 | 0.0967 ± 0.0076 |
| RNN | 0.4508 ± 0.0063 | 0.4740 ± 0.0034 | 0.8983 ± 0.0104 | 0.6206 ± 0.0054 | 0.7179 ± 0.0463 | 0.9967 ± 0.0029 | 0.1017 ± 0.0104 |
| Random Forest | 0.1342 ± 0.0052 | 0.0987 ± 0.0011 | 0.0900 ± 0.0000 | 0.0942 ± 0.0005 | 0.0841 ± 0.0007 | 0.8217 ± 0.0104 | 0.9100 ± 0.0000 |
| Linear SVM | 0.5200 ± 0.0000 | 0.8333 ± 0.0000 | 0.0500 ± 0.0000 | 0.0943 ± 0.0000 | 0.1361 ± 0.0000 | 0.0100 ± 0.0000 | 0.9500 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
