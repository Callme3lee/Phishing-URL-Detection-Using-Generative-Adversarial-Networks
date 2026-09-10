# HYB-ORIG Controlled Experiment

HYB-ORIG uses exactly the same 358,619 original training URLs as CNN/LSTM/RNN/RF/SVM. It adds the hybrid architecture and 35 PSL-aware features, but no outside rows, GAN rows, CORAL, or source-balanced sampling.

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original test | 0.9911 | 0.9880 | 0.9691 | 0.9785 | 0.9980 | 0.9952 | 0.0031 | 0.0309 |
| External 10K | 0.4856 | 0.4926 | 0.9564 | 0.6503 | 0.7266 | 0.7104 | 0.9852 | 0.0436 |

## External 10K: controlled hybrid comparison

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYB-ORIG | 0.4856 | 0.4926 | 0.9564 | 0.6503 | 0.7266 | 0.7104 | 0.9852 | 0.0436 |
| HYB-MS | 0.9177 | 0.9804 | 0.8524 | 0.9120 | 0.9663 | 0.9749 | 0.0170 | 0.1476 |

Strong internal performance does not imply external robustness. The external result shows that the architecture alone is insufficient; the registered multi-source real coverage, CORAL, ensembling, and calibration are responsible for the practical external behavior.
