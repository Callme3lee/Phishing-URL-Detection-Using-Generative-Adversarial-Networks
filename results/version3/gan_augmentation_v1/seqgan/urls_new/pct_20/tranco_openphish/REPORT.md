# Version 3 External Evaluation

This frozen evaluation contains 200 legitimate and 200 phishing proxy-labelled rows.

Sources: legitimate: Tranco Q2XX4 root proxies; phishing: OpenPhish free community snapshot.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4417 ± 0.0038 | 0.4686 ± 0.0022 | 0.8700 ± 0.0087 | 0.6091 ± 0.0040 | 0.5519 ± 0.0205 | 0.9867 ± 0.0029 | 0.1300 ± 0.0087 |
| LSTM | 0.4458 ± 0.0076 | 0.4710 ± 0.0041 | 0.8800 ± 0.0100 | 0.6136 ± 0.0054 | 0.5799 ± 0.0877 | 0.9883 ± 0.0115 | 0.1200 ± 0.0100 |
| RNN | 0.4442 ± 0.0063 | 0.4703 ± 0.0036 | 0.8850 ± 0.0150 | 0.6142 ± 0.0067 | 0.7333 ± 0.0488 | 0.9967 ± 0.0029 | 0.1150 ± 0.0150 |
| Random Forest | 0.0800 ± 0.0066 | 0.1169 ± 0.0102 | 0.1283 ± 0.0126 | 0.1224 ± 0.0113 | 0.0619 ± 0.0013 | 0.9683 ± 0.0029 | 0.8717 ± 0.0126 |
| Linear SVM | 0.2450 ± 0.0000 | 0.1304 ± 0.0000 | 0.0900 ± 0.0000 | 0.1065 ± 0.0000 | 0.1005 ± 0.0000 | 0.6000 ± 0.0000 | 0.9100 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
