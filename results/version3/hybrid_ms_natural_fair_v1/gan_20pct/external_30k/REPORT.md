# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 30000 rows (20979 legitimate, 9021 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.7884 ± 0.0197 | 0.9816 ± 0.0062 | 0.3024 ± 0.0690 | 0.4594 ± 0.0779 | 0.9045 ± 0.0137 | 0.8624 ± 0.0179 | 0.0026 ± 0.0015 | 0.6976 ± 0.0690 |
| Original-validation | 0.4400 ± 0.0477 | 0.3462 ± 0.0194 | 0.9642 ± 0.0014 | 0.5093 ± 0.0213 | 0.9045 ± 0.0137 | 0.8624 ± 0.0179 | 0.7854 ± 0.0677 | 0.0358 ± 0.0014 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
