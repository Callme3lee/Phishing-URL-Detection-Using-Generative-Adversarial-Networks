# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4375 ± 0.0000 | 0.4661 ± 0.0001 | 0.8583 ± 0.0029 | 0.6041 ± 0.0008 | 0.5044 ± 0.0488 | 0.9833 ± 0.0029 | 0.1417 ± 0.0029 |
| LSTM | 0.4333 ± 0.0118 | 0.4632 ± 0.0076 | 0.8417 ± 0.0369 | 0.5974 ± 0.0157 | 0.4965 ± 0.0253 | 0.9750 ± 0.0132 | 0.1583 ± 0.0369 |
| RNN | 0.4450 ± 0.0025 | 0.4708 ± 0.0014 | 0.8867 ± 0.0058 | 0.6150 ± 0.0025 | 0.7928 ± 0.0265 | 0.9967 ± 0.0029 | 0.1133 ± 0.0058 |
| Random Forest | 0.0658 ± 0.0038 | 0.0845 ± 0.0028 | 0.0883 ± 0.0029 | 0.0864 ± 0.0028 | 0.0582 ± 0.0009 | 0.9567 ± 0.0058 | 0.9117 ± 0.0029 |
| Linear SVM | 0.1850 ± 0.0000 | 0.1013 ± 0.0000 | 0.0800 ± 0.0000 | 0.0894 ± 0.0000 | 0.0786 ± 0.0000 | 0.7100 ± 0.0000 | 0.9200 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
