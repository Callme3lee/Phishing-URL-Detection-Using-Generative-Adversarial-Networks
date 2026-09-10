# Hybrid External Comparison

This is an exploratory comparison on the same frozen rows used in the earlier shift diagnosis. A later untouched snapshot is required for confirmation.

## Top-Domain / PhishTank (500)

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original LSTM | 0.4860 ± 0.0035 | 0.4928 ± 0.0017 | 0.9587 ± 0.0092 | 0.6510 ± 0.0019 | 0.8431 ± 0.0421 | 0.9867 ± 0.0140 | 0.0413 ± 0.0092 |
| Coverage LSTM | 0.5480 ± 0.0104 | 0.5265 ± 0.0060 | 0.9547 ± 0.0023 | 0.6787 ± 0.0047 | 0.9087 ± 0.0125 | 0.8587 ± 0.0220 | 0.0453 ± 0.0023 |
| Balanced LSTM | 0.4893 ± 0.0129 | 0.4946 ± 0.0066 | 0.9613 ± 0.0023 | 0.6531 ± 0.0063 | 0.8503 ± 0.0289 | 0.9827 ± 0.0234 | 0.0387 ± 0.0023 |
| Hybrid robust | 0.7527 ± 0.0841 | 0.9980 ± 0.0035 | 0.5067 ± 0.1698 | 0.6601 ± 0.1569 | 0.9704 ± 0.0019 | 0.0013 ± 0.0023 | 0.4933 ± 0.1698 |
| Hybrid original-threshold | 0.8727 ± 0.0185 | 0.8143 ± 0.0272 | 0.9667 ± 0.0046 | 0.8838 ± 0.0149 | 0.9704 ± 0.0019 | 0.2213 ± 0.0395 | 0.0333 ± 0.0046 |

## Tranco / OpenPhish (400)

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original LSTM | 0.4317 ± 0.0361 | 0.4620 ± 0.0225 | 0.8517 ± 0.0881 | 0.5988 ± 0.0410 | 0.6857 ± 0.0874 | 0.9883 ± 0.0161 | 0.1483 ± 0.0881 |
| Coverage LSTM | 0.5075 ± 0.0218 | 0.5044 ± 0.0122 | 0.8983 ± 0.0029 | 0.6460 ± 0.0096 | 0.8369 ± 0.0124 | 0.8833 ± 0.0454 | 0.1017 ± 0.0029 |
| Balanced LSTM | 0.4550 ± 0.0066 | 0.4763 ± 0.0035 | 0.9033 ± 0.0076 | 0.6237 ± 0.0047 | 0.5952 ± 0.0880 | 0.9933 ± 0.0076 | 0.0967 ± 0.0076 |
| Hybrid robust | 0.7225 ± 0.0601 | 1.0000 ± 0.0000 | 0.4450 ± 0.1201 | 0.6093 ± 0.1202 | 0.9276 ± 0.0064 | 0.0000 ± 0.0000 | 0.5550 ± 0.1201 |
| Hybrid original-threshold | 0.8300 ± 0.0313 | 0.7848 ± 0.0457 | 0.9133 ± 0.0076 | 0.8436 ± 0.0238 | 0.9276 ± 0.0064 | 0.2533 ± 0.0683 | 0.0867 ± 0.0076 |

## Interpretation

The hybrid sharply improves external ranking (ROC-AUC) and legitimate rejection (FPR). The conservative robust threshold nearly eliminates false positives but misses many phishing URLs. The original-validation threshold recovers high phishing recall but still has a 22–25% external FPR. This means the representation shift is greatly reduced, while operating-point calibration remains unresolved.

Do not select between these thresholds using these external labels. Freeze a deployment-like calibration set, calibrate once, and confirm once on a genuinely later untouched snapshot.
