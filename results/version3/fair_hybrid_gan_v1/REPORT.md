# Fair Classifier and HYB-MS GAN Augmentation Study

Complete PDF: `artifacts/results/fair_hybrid_gan_v1/fair_hybrid_gan_complete_report.pdf`.

## Main result

The separately calibrated HYB-MS +10% ensemble is the best balanced augmented condition on the frozen external 10K. It is a deployment candidate, not deployment-ready.

## Calibrated external 10K

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | GAN challenge recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real | 0.9177 | 0.9804 | 0.8524 | 0.9120 | 0.9663 | 0.9749 | 0.0170 | 0.1476 | 0.6309 |
| +10% | 0.9310 | 0.9843 | 0.8760 | 0.9270 | 0.9689 | 0.9781 | 0.0140 | 0.1240 | 0.8490 |
| +20% | 0.9130 | 0.9896 | 0.8348 | 0.9056 | 0.9630 | 0.9744 | 0.0088 | 0.1652 | 0.8531 |
| +30% | 0.8839 | 0.9923 | 0.7738 | 0.8695 | 0.9585 | 0.9691 | 0.0060 | 0.2262 | 0.8455 |
| +40% | 0.8515 | 0.9930 | 0.7080 | 0.8266 | 0.9555 | 0.9664 | 0.0050 | 0.2920 | 0.8509 |
