# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 30000 rows (20979 legitimate, 9021 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.7865 ± 0.0255 | 0.9616 ± 0.0321 | 0.3046 ± 0.1016 | 0.4553 ± 0.1076 | 0.8896 ± 0.0104 | 0.8204 ± 0.0122 | 0.0064 ± 0.0072 | 0.6954 ± 0.1016 |
| Original-validation | 0.4686 ± 0.0910 | 0.3610 ± 0.0428 | 0.9640 ± 0.0057 | 0.5242 ± 0.0436 | 0.8896 ± 0.0104 | 0.8204 ± 0.0122 | 0.7445 ± 0.1326 | 0.0360 ± 0.0057 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
