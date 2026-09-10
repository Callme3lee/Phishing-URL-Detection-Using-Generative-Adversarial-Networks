# Version 3 External Real-Only Evaluation

This frozen evaluation contains 250 local Top-Domain root proxies and 250 current PhishTank verified-online URLs.

Every normalized URL, 128-token encoding, and hostname is unique and has zero overlap with Version 3 training/development data or earlier external snapshots.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4900 ± 0.0020 | 0.4949 ± 0.0010 | 0.9667 ± 0.0061 | 0.6546 ± 0.0014 | 0.6580 ± 0.0256 | 0.9867 ± 0.0083 | 0.0333 ± 0.0061 |
| LSTM | 0.4860 ± 0.0035 | 0.4928 ± 0.0017 | 0.9587 ± 0.0092 | 0.6510 ± 0.0019 | 0.8431 ± 0.0421 | 0.9867 ± 0.0140 | 0.0413 ± 0.0092 |
| RNN | 0.4807 ± 0.0012 | 0.4901 ± 0.0006 | 0.9600 ± 0.0000 | 0.6489 ± 0.0005 | 0.7821 ± 0.0157 | 0.9987 ± 0.0023 | 0.0400 ± 0.0000 |
| Random Forest | 0.3080 ± 0.0020 | 0.3657 ± 0.0014 | 0.5227 ± 0.0023 | 0.4303 ± 0.0017 | 0.2741 ± 0.0021 | 0.9067 ± 0.0023 | 0.4773 ± 0.0023 |
| Linear SVM | 0.3700 ± 0.0000 | 0.3184 ± 0.0000 | 0.2280 ± 0.0000 | 0.2657 ± 0.0000 | 0.2849 ± 0.0000 | 0.4880 ± 0.0000 | 0.7720 ± 0.0000 |

## Limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
