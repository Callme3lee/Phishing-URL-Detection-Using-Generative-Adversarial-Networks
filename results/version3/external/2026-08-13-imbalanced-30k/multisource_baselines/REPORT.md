# Version 3 External Evaluation

This frozen evaluation contains 20979 legitimate and 9021 phishing proxy-labelled rows.

Sources: openphish: openphish; phishtank: phishtank; tranco: tranco.

Every normalized URL, 128-token encoding, and hostname is unique; collection-time overlap checks are recorded in the frozen manifest.

Thresholds are the unchanged internal-validation thresholds saved with each checkpoint. The external data were not used for training, threshold selection, or model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.3048 ± 0.0027 | 0.2986 ± 0.0009 | 0.9725 ± 0.0012 | 0.4569 ± 0.0011 | 0.8201 ± 0.0129 | 0.7176 ± 0.0188 | 0.9824 ± 0.0037 | 0.0275 ± 0.0012 |
| LSTM | 0.3416 ± 0.0401 | 0.3106 ± 0.0132 | 0.9722 ± 0.0029 | 0.4707 ± 0.0148 | 0.9354 ± 0.0082 | 0.9285 ± 0.0085 | 0.9295 ± 0.0585 | 0.0278 ± 0.0029 |
| RNN | 0.2944 ± 0.0015 | 0.2948 ± 0.0006 | 0.9668 ± 0.0064 | 0.4518 ± 0.0013 | 0.8702 ± 0.0233 | 0.8145 ± 0.0384 | 0.9947 ± 0.0045 | 0.0332 ± 0.0064 |
| Random Forest | 0.3801 ± 0.0162 | 0.2631 ± 0.0062 | 0.5891 ± 0.0016 | 0.3637 ± 0.0058 | 0.5313 ± 0.0027 | 0.5105 ± 0.0025 | 0.7098 ± 0.0234 | 0.4109 ± 0.0016 |
| Linear SVM | 0.4338 ± 0.0000 | 0.1830 ± 0.0000 | 0.2550 ± 0.0000 | 0.2131 ± 0.0000 | 0.3092 ± 0.0000 | 0.3455 ± 0.0000 | 0.4893 ± 0.0000 | 0.7450 ± 0.0000 |

## Limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
