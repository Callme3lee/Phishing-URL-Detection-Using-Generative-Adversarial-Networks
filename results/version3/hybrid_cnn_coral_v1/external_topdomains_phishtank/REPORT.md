# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 500 rows (250 legitimate, 250 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.7527 ± 0.0841 | 0.9980 ± 0.0035 | 0.5067 ± 0.1698 | 0.6601 ± 0.1569 | 0.9704 ± 0.0019 | 0.9765 ± 0.0043 | 0.0013 ± 0.0023 | 0.4933 ± 0.1698 |
| Original-validation | 0.8727 ± 0.0185 | 0.8143 ± 0.0272 | 0.9667 ± 0.0046 | 0.8838 ± 0.0149 | 0.9704 ± 0.0019 | 0.9765 ± 0.0043 | 0.2213 ± 0.0395 | 0.0333 ± 0.0046 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Top-Domain membership is a legitimacy proxy rather than a manual safety audit.
- Legitimate examples are root domains, while phishing examples may contain deep paths and queries.
- The balanced 500-row snapshot does not estimate deployment prevalence.
- PhishTank is one phishing source and cannot represent every campaign or region.
