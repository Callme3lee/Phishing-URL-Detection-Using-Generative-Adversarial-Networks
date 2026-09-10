# Hybrid CNN External Evaluation

exploratory because these frozen sets informed the hybrid design; a later untouched snapshot is required for confirmation.

Dataset: 400 rows (200 legitimate, 200 phishing).

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust | 0.7225 ± 0.0601 | 1.0000 ± 0.0000 | 0.4450 ± 0.1201 | 0.6093 ± 0.1202 | 0.9276 ± 0.0064 | 0.9528 ± 0.0017 | 0.0000 ± 0.0000 | 0.5550 ± 0.1201 |
| Original-validation | 0.8300 ± 0.0313 | 0.7848 ± 0.0457 | 0.9133 ± 0.0076 | 0.8436 ± 0.0238 | 0.9276 ± 0.0064 | 0.9528 ± 0.0017 | 0.2533 ± 0.0683 | 0.0867 ± 0.0076 |

The threshold comparison is diagnostic. Neither threshold was chosen using these external labels.

## Source limitations

- Tranco membership is a legitimate proxy, not page-level manual adjudication.
- OpenPhish Community is a limited feed without the metadata available in paid tiers.
- Legitimate samples are roots while phishing samples can include paths and queries.
- A balanced 400-row set does not estimate deployment prevalence.
- This is a source-transfer test collected on the same date, not a later temporal evaluation.
