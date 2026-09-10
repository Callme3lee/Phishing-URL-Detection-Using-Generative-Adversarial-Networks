# Definitive Equal-Calibration Comparison

Compared **106 configurations / 318 checkpoints** under one calibration policy.

**Winner:** Feature-equal / weighted HYB | HYB-MS | +10%.

| Set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | 0.9310 | 0.9843 | 0.8760 | 0.9270 | 0.9689 | 0.9781 | 0.0140 | 0.1240 |
| 30K | 0.9543 | 0.9749 | 0.8705 | 0.9198 | 0.9699 | 0.9638 | 0.0096 | 0.1295 |

The 10K and 30K are now retrospective selection evidence. A new untouched temporal dataset is required for final confirmation.
