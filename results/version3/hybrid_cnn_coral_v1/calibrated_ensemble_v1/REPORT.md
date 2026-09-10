# Calibrated Hybrid Ensemble

Frozen threshold: `0.98795575`.

Policy: maximum calibration F1 subject to calibration FPR <= 1.00%.

The calibration row is resubstitution performance and is not an unbiased test result. The two older external sets are exploratory because they informed the hybrid design.

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration (resubstitution) | 0.9425 | 0.9890 | 0.8950 | 0.9396 | 0.9745 | 0.9769 | 0.0100 | 0.1050 |
| Top-Domain / PhishTank | 0.9260 | 0.9908 | 0.8600 | 0.9208 | 0.9746 | 0.9821 | 0.0080 | 0.1400 |
| Tranco / OpenPhish | 0.8875 | 0.9755 | 0.7950 | 0.8760 | 0.9324 | 0.9567 | 0.0200 | 0.2050 |

A genuinely later untouched confirmation snapshot is still required before deployment claims.
