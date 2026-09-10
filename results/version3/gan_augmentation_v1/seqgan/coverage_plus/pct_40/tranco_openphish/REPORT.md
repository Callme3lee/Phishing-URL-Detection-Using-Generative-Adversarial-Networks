# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4050 ± 0.0152 | 0.4452 ± 0.0104 | 0.7750 ± 0.0409 | 0.5655 ± 0.0193 | 0.4899 ± 0.0201 | 0.9650 ± 0.0132 | 0.2250 ± 0.0409 |
| LSTM | 0.4458 ± 0.0539 | 0.4677 ± 0.0340 | 0.8100 ± 0.1058 | 0.5924 ± 0.0537 | 0.5658 ± 0.1286 | 0.9183 ± 0.0679 | 0.1900 ± 0.1058 |
| RNN | 0.4492 ± 0.0126 | 0.4726 ± 0.0068 | 0.8767 ± 0.0202 | 0.6141 ± 0.0095 | 0.6692 ± 0.1266 | 0.9783 ± 0.0225 | 0.1233 ± 0.0202 |
| Random Forest | 0.1175 ± 0.0066 | 0.0895 ± 0.0035 | 0.0833 ± 0.0029 | 0.0863 ± 0.0031 | 0.0714 ± 0.0012 | 0.8483 ± 0.0115 | 0.9167 ± 0.0029 |
| Linear SVM | 0.1900 ± 0.0000 | 0.1026 ± 0.0000 | 0.0800 ± 0.0000 | 0.0899 ± 0.0000 | 0.0799 ± 0.0000 | 0.7000 ± 0.0000 | 0.9200 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
