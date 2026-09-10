# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 30000 rows (20979 legitimate, 9021 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.7765 ± 0.0280 | 0.9171 ± 0.0619 | 0.2902 ± 0.1292 | 0.4271 ± 0.1360 | 0.8614 ± 0.0215 | 0.7818 ± 0.0250 | 0.0144 ± 0.0156 | 0.7098 ± 0.1292 |
| Original-validation | 0.3489 ± 0.0171 | 0.3116 ± 0.0068 | 0.9629 ± 0.0091 | 0.4708 ± 0.0087 | 0.8614 ± 0.0215 | 0.7818 ± 0.0250 | 0.9152 ± 0.0211 | 0.0371 ± 0.0091 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
