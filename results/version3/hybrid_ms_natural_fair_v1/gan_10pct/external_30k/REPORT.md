# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 30000 rows (20979 legitimate, 9021 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.8009 ± 0.0200 | 0.9851 ± 0.0049 | 0.3428 ± 0.0657 | 0.5063 ± 0.0738 | 0.9201 ± 0.0091 | 0.8867 ± 0.0173 | 0.0021 ± 0.0004 | 0.6572 ± 0.0657 |
| Original-validation | 0.4833 ± 0.1503 | 0.3725 ± 0.0624 | 0.9647 ± 0.0098 | 0.5352 ± 0.0655 | 0.9201 ± 0.0091 | 0.8867 ± 0.0173 | 0.7237 ± 0.2192 | 0.0353 ± 0.0098 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not a manual page-level safety adjudication.
- Legitimate examples are root domains while phishing examples often contain paths and queries.
- The seeded 30.07% phishing prevalence is an experimental stress-test prevalence, not measured browser prevalence.
- PhishTank and the limited OpenPhish Community feed do not represent every campaign, language, or region.
- The snapshot is temporally later than training, but source reuse and proxy labels still limit deployment claims.
