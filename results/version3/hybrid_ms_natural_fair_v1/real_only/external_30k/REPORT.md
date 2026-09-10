# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 30000 rows (20979 legitimate, 9021 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.8043 ± 0.0071 | 0.9847 ± 0.0099 | 0.3547 ± 0.0214 | 0.5213 ± 0.0239 | 0.9399 ± 0.0102 | 0.9172 ± 0.0191 | 0.0023 ± 0.0014 | 0.6453 ± 0.0214 |
| Original-validation | 0.3686 ± 0.0466 | 0.3202 ± 0.0165 | 0.9739 ± 0.0026 | 0.4817 ± 0.0184 | 0.9399 ± 0.0102 | 0.9172 ± 0.0191 | 0.8917 ± 0.0675 | 0.0261 ± 0.0026 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
