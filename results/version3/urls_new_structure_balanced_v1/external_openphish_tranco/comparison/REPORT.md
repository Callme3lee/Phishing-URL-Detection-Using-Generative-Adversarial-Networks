# External Comparison — Original vs URLs_New structure-balanced

Frozen dataset SHA-256: `c31791b5d93722d66b306b12ff7c7857909539191036e25c5931f7b7be120793`. Mean ± sample SD across paired seeds 42/43/44; thresholds remain those selected on each run's unchanged internal validation set.

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | Original | 0.4458 ± 0.0014 | 0.4712 ± 0.0010 | 0.8850 ± 0.0087 | 0.6149 ± 0.0029 | 0.5737 ± 0.0183 | 0.9933 ± 0.0058 | 0.1150 ± 0.0087 |
| CNN | URLs_New structure-balanced | 0.4525 ± 0.0000 | 0.4746 ± 0.0002 | 0.8883 ± 0.0076 | 0.6187 ± 0.0020 | 0.5728 ± 0.0272 | 0.9833 ± 0.0076 | 0.1117 ± 0.0076 |
| LSTM | Original | 0.4317 ± 0.0361 | 0.4620 ± 0.0225 | 0.8517 ± 0.0881 | 0.5988 ± 0.0410 | 0.6857 ± 0.0874 | 0.9883 ± 0.0161 | 0.1483 ± 0.0881 |
| LSTM | URLs_New structure-balanced | 0.4550 ± 0.0066 | 0.4763 ± 0.0035 | 0.9033 ± 0.0076 | 0.6237 ± 0.0047 | 0.5952 ± 0.0880 | 0.9933 ± 0.0076 | 0.0967 ± 0.0076 |
| RNN | Original | 0.4467 ± 0.0014 | 0.4718 ± 0.0009 | 0.8917 ± 0.0058 | 0.6171 ± 0.0022 | 0.7534 ± 0.0180 | 0.9983 ± 0.0029 | 0.1083 ± 0.0058 |
| RNN | URLs_New structure-balanced | 0.4508 ± 0.0063 | 0.4740 ± 0.0034 | 0.8983 ± 0.0104 | 0.6206 ± 0.0054 | 0.7179 ± 0.0463 | 0.9967 ± 0.0029 | 0.1017 ± 0.0104 |
| Random Forest | Original | 0.0975 ± 0.0043 | 0.1484 ± 0.0064 | 0.1700 ± 0.0087 | 0.1585 ± 0.0074 | 0.0715 ± 0.0004 | 0.9750 ± 0.0000 | 0.8300 ± 0.0087 |
| Random Forest | URLs_New structure-balanced | 0.1342 ± 0.0052 | 0.0987 ± 0.0011 | 0.0900 ± 0.0000 | 0.0942 ± 0.0005 | 0.0841 ± 0.0007 | 0.8217 ± 0.0104 | 0.9100 ± 0.0000 |
| Linear SVM | Original | 0.3325 ± 0.0000 | 0.1810 ± 0.0000 | 0.0950 ± 0.0000 | 0.1246 ± 0.0000 | 0.1286 ± 0.0000 | 0.4300 ± 0.0000 | 0.9050 ± 0.0000 |
| Linear SVM | URLs_New structure-balanced | 0.5200 ± 0.0000 | 0.8333 ± 0.0000 | 0.0500 ± 0.0000 | 0.0943 ± 0.0000 | 0.1361 ± 0.0000 | 0.0100 ± 0.0000 | 0.9500 ± 0.0000 |

## Paired mean deltas: experiment minus original

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | +0.0067 | +0.0035 | +0.0033 | +0.0038 | -0.0009 | -0.0100 | -0.0033 |
| LSTM | +0.0233 | +0.0143 | +0.0517 | +0.0249 | -0.0905 | +0.0050 | -0.0517 |
| RNN | +0.0042 | +0.0023 | +0.0067 | +0.0035 | -0.0356 | -0.0017 | -0.0067 |
| Random Forest | +0.0367 | -0.0497 | -0.0800 | -0.0643 | +0.0125 | -0.1533 | +0.0800 |
| Linear SVM | +0.1875 | +0.6524 | -0.0450 | -0.0303 | +0.0075 | -0.4200 | +0.0450 |

FPR/FNR are error rates, so negative deltas are improvements. Three seeds support descriptive paired comparison, not a strong significance claim.
