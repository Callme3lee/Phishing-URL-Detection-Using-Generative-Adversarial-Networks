# Original-Test Classifier Comparison

same frozen hostname-held-out test split from the original URLs_New dataset.

CNN/LSTM/RNN/RF/SVM were trained on the same 358,619 original real rows. HYB is evaluated on the same original test rows but was trained under its registered 363,692-row multi-source domain-adaptation condition.

Abbreviations: `RF` = Random Forest; `SVM` = linear Support Vector Machine; `HYB` = calibrated three-seed Hybrid CNN + Lexical + CORAL ensemble.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.9889 | 0.9746 | 0.9725 | 0.9736 | 0.9970 | 0.9935 | 0.0067 | 0.0275 |
| LSTM | 0.9898 | 0.9817 | 0.9696 | 0.9756 | 0.9968 | 0.9931 | 0.0048 | 0.0304 |
| RNN | 0.9866 | 0.9730 | 0.9629 | 0.9679 | 0.9939 | 0.9879 | 0.0071 | 0.0371 |
| RF | 0.9496 | 0.9718 | 0.7822 | 0.8667 | 0.9627 | 0.9330 | 0.0060 | 0.2178 |
| SVM | 0.8975 | 0.9687 | 0.5280 | 0.6835 | 0.9298 | 0.8724 | 0.0045 | 0.4720 |
| HYB | 0.9608 | 0.9995 | 0.8135 | 0.8970 | 0.9964 | 0.9923 | 0.0001 | 0.1865 |

The five baseline rows are means across three seeds. HYB is one mean-score ensemble result, so no seed SD is shown.
