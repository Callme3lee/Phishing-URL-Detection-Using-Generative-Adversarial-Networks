# Version 3 External Evaluation

This frozen evaluation contains 20979 legitimate and 9021 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.2924 ± 0.0013 | 0.2931 ± 0.0007 | 0.9585 ± 0.0075 | 0.4489 ± 0.0016 | 0.6304 ± 0.0176 | 0.4607 ± 0.0084 | 0.9940 ± 0.0047 | 0.0415 ± 0.0075 |
| LSTM | 0.3025 ± 0.0075 | 0.2972 ± 0.0014 | 0.9667 ± 0.0074 | 0.4546 ± 0.0008 | 0.8411 ± 0.0284 | 0.7848 ± 0.0388 | 0.9831 ± 0.0139 | 0.0333 ± 0.0074 |
| RNN | 0.2886 ± 0.0001 | 0.2918 ± 0.0002 | 0.9569 ± 0.0021 | 0.4472 ± 0.0005 | 0.7439 ± 0.0144 | 0.6507 ± 0.0450 | 0.9988 ± 0.0009 | 0.0431 ± 0.0021 |
| Random Forest | 0.1899 ± 0.0006 | 0.1755 ± 0.0007 | 0.4583 ± 0.0021 | 0.2539 ± 0.0010 | 0.2160 ± 0.0009 | 0.2555 ± 0.0014 | 0.9255 ± 0.0000 | 0.5417 ± 0.0021 |
| Linear SVM | 0.4384 ± 0.0000 | 0.1796 ± 0.0000 | 0.2431 ± 0.0000 | 0.2066 ± 0.0000 | 0.2946 ± 0.0000 | 0.3286 ± 0.0000 | 0.4776 ± 0.0000 | 0.7569 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
