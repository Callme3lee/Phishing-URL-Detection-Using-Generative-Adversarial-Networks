# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4250 ± 0.0125 | 0.4576 ± 0.0089 | 0.8150 ± 0.0522 | 0.5859 ± 0.0208 | 0.5319 ± 0.0647 | 0.9650 ± 0.0312 | 0.1850 ± 0.0522 |
| LSTM | 0.4717 ± 0.0322 | 0.4850 ± 0.0174 | 0.8767 ± 0.0189 | 0.6242 ± 0.0100 | 0.5904 ± 0.1019 | 0.9333 ± 0.0822 | 0.1233 ± 0.0189 |
| RNN | 0.4533 ± 0.0072 | 0.4751 ± 0.0036 | 0.8883 ± 0.0126 | 0.6190 ± 0.0043 | 0.6913 ± 0.0883 | 0.9817 ± 0.0202 | 0.1117 ± 0.0126 |
| Random Forest | 0.1217 ± 0.0029 | 0.1254 ± 0.0029 | 0.1267 ± 0.0029 | 0.1260 ± 0.0029 | 0.0801 ± 0.0003 | 0.8833 ± 0.0029 | 0.8733 ± 0.0029 |
| Linear SVM | 0.2475 ± 0.0000 | 0.1203 ± 0.0000 | 0.0800 ± 0.0000 | 0.0961 ± 0.0000 | 0.1021 ± 0.0000 | 0.5850 ± 0.0000 | 0.9200 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
