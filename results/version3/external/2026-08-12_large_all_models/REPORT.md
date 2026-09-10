# Frozen 10K External Snapshot — Baselines vs HYB

The exact frozen dataset contains 5,000 legitimate proxies and 5,000 phishing URLs. SHA-256: `fdc040874608908c6f6f89523a219e044e3977f449c66d4966d48c76ed98c1b8`.

all checkpoints and thresholds frozen; no retraining, reselection, or external threshold tuning.

Real-only baselines use their frozen internal thresholds and report mean ± sample SD across seeds 42/43/44. HYB is the frozen calibrated three-seed ensemble and therefore has one result rather than a seed SD.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4821 | 0.4908 | 0.9605 | 0.6497 | 0.5730 | 0.5976 | 0.9963 | 0.0395 |
| LSTM | 0.4791 | 0.4890 | 0.9427 | 0.6439 | 0.8323 | 0.8674 | 0.9845 | 0.0573 |
| RNN | 0.4805 | 0.4900 | 0.9601 | 0.6489 | 0.7771 | 0.8102 | 0.9991 | 0.0399 |
| RF | 0.3285 | 0.3859 | 0.5799 | 0.4634 | 0.2482 | 0.4265 | 0.9229 | 0.4201 |
| SVM | 0.3469 | 0.2587 | 0.1642 | 0.2009 | 0.2342 | 0.4280 | 0.4704 | 0.8358 |
| HYB | 0.9177 | 0.9804 | 0.8524 | 0.9120 | 0.9663 | 0.9749 | 0.0170 | 0.1476 |

## Interpretation

HYB is the only model with a practical false-positive rate on this snapshot. The real-only neural baselines retain high phishing recall but achieve it by labelling almost all legitimate roots as phishing. RF and SVM also show ranking reversal (ROC-AUC below 0.5), so their failures cannot be repaired by threshold adjustment alone.

The snapshot remains external evaluation data and must not be moved into training or threshold selection.
