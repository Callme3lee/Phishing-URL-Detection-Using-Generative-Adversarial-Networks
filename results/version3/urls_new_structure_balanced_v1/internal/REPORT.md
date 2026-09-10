# URLs_New structure-balanced — Unchanged Internal Test

Mean ± sample standard deviation across paired seeds 42/43/44. Original validation/test rows and threshold policy were unchanged.

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | Original | 0.9889 ± 0.0013 | 0.9746 ± 0.0050 | 0.9725 ± 0.0023 | 0.9736 ± 0.0031 | 0.9970 ± 0.0008 | 0.0067 ± 0.0014 | 0.0275 ± 0.0023 |
| CNN | URLs_New structure-balanced | 0.9882 ± 0.0002 | 0.9754 ± 0.0046 | 0.9679 ± 0.0039 | 0.9716 ± 0.0004 | 0.9965 ± 0.0001 | 0.0065 ± 0.0013 | 0.0321 ± 0.0039 |
| LSTM | Original | 0.9898 ± 0.0003 | 0.9817 ± 0.0024 | 0.9696 ± 0.0012 | 0.9756 ± 0.0006 | 0.9968 ± 0.0004 | 0.0048 ± 0.0006 | 0.0304 ± 0.0012 |
| LSTM | URLs_New structure-balanced | 0.9886 ± 0.0003 | 0.9823 ± 0.0031 | 0.9632 ± 0.0022 | 0.9726 ± 0.0006 | 0.9961 ± 0.0005 | 0.0046 ± 0.0008 | 0.0368 ± 0.0022 |
| RNN | Original | 0.9866 ± 0.0017 | 0.9730 ± 0.0046 | 0.9629 ± 0.0042 | 0.9679 ± 0.0041 | 0.9939 ± 0.0008 | 0.0071 ± 0.0012 | 0.0371 ± 0.0042 |
| RNN | URLs_New structure-balanced | 0.9814 ± 0.0050 | 0.9738 ± 0.0082 | 0.9365 ± 0.0298 | 0.9545 ± 0.0130 | 0.9869 ± 0.0085 | 0.0067 ± 0.0023 | 0.0635 ± 0.0298 |
| Random Forest | Original | 0.9496 ± 0.0001 | 0.9718 ± 0.0012 | 0.7822 ± 0.0006 | 0.8667 ± 0.0002 | 0.9627 ± 0.0003 | 0.0060 ± 0.0003 | 0.2178 ± 0.0006 |
| Random Forest | URLs_New structure-balanced | 0.9427 ± 0.0001 | 0.9639 ± 0.0010 | 0.7548 ± 0.0006 | 0.8466 ± 0.0003 | 0.9643 ± 0.0000 | 0.0075 ± 0.0002 | 0.2452 ± 0.0006 |
| Linear SVM | Original | 0.8975 ± 0.0000 | 0.9687 ± 0.0000 | 0.5280 ± 0.0000 | 0.6835 ± 0.0000 | 0.9298 ± 0.0000 | 0.0045 ± 0.0000 | 0.4720 ± 0.0000 |
| Linear SVM | URLs_New structure-balanced | 0.8691 ± 0.0000 | 0.9534 ± 0.0000 | 0.3947 ± 0.0000 | 0.5583 ± 0.0000 | 0.9232 ± 0.0000 | 0.0051 ± 0.0000 | 0.6053 ± 0.0000 |

## Paired mean deltas: experiment minus original

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | -0.0008 | +0.0008 | -0.0046 | -0.0019 | -0.0005 | -0.0002 | +0.0046 |
| LSTM | -0.0012 | +0.0006 | -0.0064 | -0.0030 | -0.0007 | -0.0002 | +0.0064 |
| RNN | -0.0053 | +0.0007 | -0.0264 | -0.0134 | -0.0069 | -0.0003 | +0.0264 |
| Random Forest | -0.0069 | -0.0079 | -0.0274 | -0.0201 | +0.0016 | +0.0015 | +0.0274 |
| Linear SVM | -0.0284 | -0.0153 | -0.1333 | -0.1252 | -0.0066 | +0.0006 | +0.1333 |

With only three paired seeds, these deltas are descriptive and are not presented as conclusive statistical significance.
