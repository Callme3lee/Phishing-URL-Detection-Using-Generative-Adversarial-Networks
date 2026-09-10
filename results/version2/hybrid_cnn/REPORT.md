# Hybrid CNN Real-Only Evaluation

The hybrid was added without changing the frozen real data, hostname-separated splits, existing models, seeds, or threshold policy.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Character CNN | 0.963503 ± 0.000657 | 0.992540 ± 0.000276 | 0.943149 ± 0.001230 | 0.967214 ± 0.000611 | 0.996119 ± 0.000059 | 0.009428 ± 0.000357 | 0.056851 ± 0.001230 |
| Character LSTM | 0.964454 ± 0.002135 | 0.992730 ± 0.000406 | 0.944645 ± 0.003560 | 0.968089 ± 0.001973 | 0.996460 ± 0.000083 | 0.009200 ± 0.000502 | 0.055355 ± 0.003560 |
| Random Forest | 0.950569 ± 0.000068 | 0.991746 ± 0.000118 | 0.921068 ± 0.000065 | 0.955101 ± 0.000060 | 0.987961 ± 0.000008 | 0.010195 ± 0.000147 | 0.078932 ± 0.000065 |
| Linear SVM | 0.931315 ± 0.000000 | 0.991858 ± 0.000000 | 0.886951 ± 0.000000 | 0.936476 ± 0.000000 | 0.963661 ± 0.000000 | 0.009683 ± 0.000000 | 0.113049 ± 0.000000 |
| Hybrid CNN | 0.966536 ± 0.000170 | 0.992233 ± 0.000157 | 0.948802 ± 0.000161 | 0.970032 ± 0.000153 | 0.996694 ± 0.000066 | 0.009877 ± 0.000200 | 0.051198 ± 0.000161 |

## Paired design

All models used the same 2,055,821 real training rows and unchanged validation/test sets. The hybrid contains no generated rows. Its 28 lexical features were standardized using training rows only.

The strongest existing model by mean F1 is Character LSTM. Hybrid-minus-Character LSTM mean deltas are:

- Accuracy: +0.002082
- Precision: -0.000497
- Recall: +0.004157
- F1: +0.001943
- Roc Auc: +0.000234
- False Positive Rate: +0.000678
- False Negative Rate: -0.004157

With three paired seeds, accuracy, recall, and F1 deltas have 95% intervals that cross zero. The ROC-AUC delta is +0.000234 with paired 95% CI [+0.000077, +0.000391], the only requested improvement whose interval excludes zero. Mean gains must not be presented as universal external improvement.
