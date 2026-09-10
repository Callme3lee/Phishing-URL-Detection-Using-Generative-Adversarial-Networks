# Complete Classifier Dashboard

This dashboard combines all directly comparable Version 3 classifier conditions on the unchanged internal test and frozen external 10K snapshot.

- `ORIG`: 358,619 original real training URLs.
- `MS`: the same 363,692 stored multi-source real URLs for every MS condition.
- No condition in this figure contains GAN-generated training rows.
- Baselines and HYB-ORIG are mean results across seeds 42/43/44.
- `HYB-MS*` is a single three-seed mean-score ensemble using its separately frozen calibration threshold.
- HYB-MS has the same stored MS row pool but additionally uses source-balanced sampling, PSL-aware lexical fusion, conditional CORAL, ensembling, and calibration.

## Internal original test (49,229 URLs)

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-ORIG | 0.9889 | 0.9746 | 0.9725 | 0.9736 | 0.9970 | 0.9935 | 0.0067 | 0.0275 |
| CNN-MS | 0.9890 | 0.9756 | 0.9719 | 0.9737 | 0.9970 | 0.9936 | 0.0065 | 0.0281 |
| LSTM-ORIG | 0.9898 | 0.9817 | 0.9696 | 0.9756 | 0.9968 | 0.9931 | 0.0048 | 0.0304 |
| LSTM-MS | 0.9884 | 0.9790 | 0.9654 | 0.9721 | 0.9958 | 0.9914 | 0.0055 | 0.0346 |
| RNN-ORIG | 0.9866 | 0.9730 | 0.9629 | 0.9679 | 0.9939 | 0.9879 | 0.0071 | 0.0371 |
| RNN-MS | 0.9819 | 0.9534 | 0.9608 | 0.9570 | 0.9928 | 0.9858 | 0.0125 | 0.0392 |
| RF-ORIG | 0.9496 | 0.9718 | 0.7822 | 0.8667 | 0.9627 | 0.9330 | 0.0060 | 0.2178 |
| RF-MS | 0.9488 | 0.9717 | 0.7785 | 0.8644 | 0.9640 | 0.9344 | 0.0060 | 0.2215 |
| SVM-ORIG | 0.8975 | 0.9687 | 0.5280 | 0.6835 | 0.9298 | 0.8724 | 0.0045 | 0.4720 |
| SVM-MS | 0.8988 | 0.9684 | 0.5346 | 0.6889 | 0.9301 | 0.8722 | 0.0046 | 0.4654 |
| HYB-ORIG | 0.9911 | 0.9880 | 0.9691 | 0.9785 | 0.9980 | 0.9952 | 0.0031 | 0.0309 |
| HYB-MS* | 0.9608 | 0.9995 | 0.8135 | 0.8970 | 0.9964 | 0.9923 | 0.0001 | 0.1865 |

## Frozen external test (10,000 URLs)

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-ORIG | 0.4821 | 0.4908 | 0.9605 | 0.6497 | 0.5730 | 0.5976 | 0.9963 | 0.0395 |
| CNN-MS | 0.4887 | 0.4942 | 0.9662 | 0.6539 | 0.7950 | 0.8071 | 0.9889 | 0.0338 |
| LSTM-ORIG | 0.4791 | 0.4890 | 0.9427 | 0.6439 | 0.8323 | 0.8674 | 0.9845 | 0.0573 |
| LSTM-MS | 0.5163 | 0.5089 | 0.9641 | 0.6660 | 0.9309 | 0.9539 | 0.9315 | 0.0359 |
| RNN-ORIG | 0.4805 | 0.4900 | 0.9601 | 0.6489 | 0.7771 | 0.8102 | 0.9991 | 0.0399 |
| RNN-MS | 0.4831 | 0.4914 | 0.9614 | 0.6504 | 0.8725 | 0.8931 | 0.9951 | 0.0386 |
| RF-ORIG | 0.3285 | 0.3859 | 0.5799 | 0.4634 | 0.2482 | 0.4265 | 0.9229 | 0.4201 |
| RF-MS | 0.4834 | 0.4881 | 0.6759 | 0.5668 | 0.6298 | 0.7433 | 0.7091 | 0.3241 |
| SVM-ORIG | 0.3469 | 0.2587 | 0.1642 | 0.2009 | 0.2342 | 0.4280 | 0.4704 | 0.8358 |
| SVM-MS | 0.3464 | 0.2670 | 0.1760 | 0.2122 | 0.2499 | 0.4372 | 0.4832 | 0.8240 |
| HYB-ORIG | 0.4856 | 0.4926 | 0.9564 | 0.6503 | 0.7266 | 0.7104 | 0.9852 | 0.0436 |
| HYB-MS* | 0.9177 | 0.9804 | 0.8524 | 0.9120 | 0.9663 | 0.9749 | 0.0170 | 0.1476 |

## Reading the graph

Green means better. In the detection panels, high values are good. In the error panels, low FPR/FNR values are good, so their colour scale is reversed. The external panels should guide deployment decisions because the internal dataset substantially overstates cross-source performance.

The decisive external result is HYB-MS: accuracy 0.9177, F1 0.9120, ROC-AUC 0.9663, FPR 0.0170, and FNR 0.1476. The unchanged MS baselines improve some ranking metrics, but their thresholds still transfer poorly.
