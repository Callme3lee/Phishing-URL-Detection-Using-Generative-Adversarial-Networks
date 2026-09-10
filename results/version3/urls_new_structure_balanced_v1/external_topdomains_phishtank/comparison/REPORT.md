# External Comparison — Original vs URLs_New structure-balanced

Frozen dataset SHA-256: `304d809a72b63b76c80ad0e2e326f5be7df38ada5211a23b773c2b7d87eb89fa`. Mean ± sample SD across paired seeds 42/43/44; thresholds remain those selected on each run's unchanged internal validation set.

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | Original | 0.4900 ± 0.0020 | 0.4949 ± 0.0010 | 0.9667 ± 0.0061 | 0.6546 ± 0.0014 | 0.6580 ± 0.0256 | 0.9867 ± 0.0083 | 0.0333 ± 0.0061 |
| CNN | URLs_New structure-balanced | 0.4973 ± 0.0042 | 0.4986 ± 0.0021 | 0.9653 ± 0.0061 | 0.6576 ± 0.0032 | 0.6991 ± 0.0014 | 0.9707 ± 0.0046 | 0.0347 ± 0.0061 |
| LSTM | Original | 0.4860 ± 0.0035 | 0.4928 ± 0.0017 | 0.9587 ± 0.0092 | 0.6510 ± 0.0019 | 0.8431 ± 0.0421 | 0.9867 ± 0.0140 | 0.0413 ± 0.0092 |
| LSTM | URLs_New structure-balanced | 0.4893 ± 0.0129 | 0.4946 ± 0.0066 | 0.9613 ± 0.0023 | 0.6531 ± 0.0063 | 0.8503 ± 0.0289 | 0.9827 ± 0.0234 | 0.0387 ± 0.0023 |
| RNN | Original | 0.4807 ± 0.0012 | 0.4901 ± 0.0006 | 0.9600 ± 0.0000 | 0.6489 ± 0.0005 | 0.7821 ± 0.0157 | 0.9987 ± 0.0023 | 0.0400 ± 0.0000 |
| RNN | URLs_New structure-balanced | 0.4780 ± 0.0035 | 0.4887 ± 0.0018 | 0.9547 ± 0.0092 | 0.6465 ± 0.0037 | 0.7166 ± 0.1215 | 0.9987 ± 0.0023 | 0.0453 ± 0.0092 |
| Random Forest | Original | 0.3080 ± 0.0020 | 0.3657 ± 0.0014 | 0.5227 ± 0.0023 | 0.4303 ± 0.0017 | 0.2741 ± 0.0021 | 0.9067 ± 0.0023 | 0.4773 ± 0.0023 |
| Random Forest | URLs_New structure-balanced | 0.3373 ± 0.0012 | 0.3475 ± 0.0018 | 0.3707 ± 0.0061 | 0.3587 ± 0.0037 | 0.3074 ± 0.0024 | 0.6960 ± 0.0069 | 0.6293 ± 0.0061 |
| Linear SVM | Original | 0.3700 ± 0.0000 | 0.3184 ± 0.0000 | 0.2280 ± 0.0000 | 0.2657 ± 0.0000 | 0.2849 ± 0.0000 | 0.4880 ± 0.0000 | 0.7720 ± 0.0000 |
| Linear SVM | URLs_New structure-balanced | 0.5720 ± 0.0000 | 0.9737 ± 0.0000 | 0.1480 ± 0.0000 | 0.2569 ± 0.0000 | 0.3246 ± 0.0000 | 0.0040 ± 0.0000 | 0.8520 ± 0.0000 |

## Paired mean deltas: experiment minus original

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | +0.0073 | +0.0037 | -0.0013 | +0.0030 | +0.0411 | -0.0160 | +0.0013 |
| LSTM | +0.0033 | +0.0017 | +0.0027 | +0.0021 | +0.0072 | -0.0040 | -0.0027 |
| RNN | -0.0027 | -0.0014 | -0.0053 | -0.0024 | -0.0655 | +0.0000 | +0.0053 |
| Random Forest | +0.0293 | -0.0182 | -0.1520 | -0.0716 | +0.0333 | -0.2107 | +0.1520 |
| Linear SVM | +0.2020 | +0.6552 | -0.0800 | -0.0088 | +0.0397 | -0.4840 | +0.0800 |

FPR/FNR are error rates, so negative deltas are improvements. Three seeds support descriptive paired comparison, not a strong significance claim.
