# Final frozen browser-policy evaluation

Status: **COMPLETED ONCE; NO POST-TEST TUNING PERMITTED**

The final cohort contained 6,000 unseen URLs: 5,700 legitimate and 300 phishing.
All thresholds were selected on the separate 4,000-row policy-development split.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYB-MS real-only | 97.95% | 82.18% | 75.33% | 78.61% | 95.87% | 84.31% | 0.86% | 24.67% |
| HYB-MS +10% GAN | 98.23% | 84.15% | 79.67% | 81.85% | 95.52% | 87.18% | 0.79% | 20.33% |
| Browser ONNX v0.5.2 candidate | 98.23% | 84.15% | 79.67% | 81.85% | 95.52% | 87.18% | 0.79% | 20.33% |

GAN augmentation improved the operating-point results: 13 more phishing URLs were
detected, four fewer legitimate URLs were falsely flagged, F1 increased by 3.24
percentage points and PR-AUC increased by 2.87 points. The paired exact comparison
favored GAN augmentation overall (36 unique corrections versus 19 unique
regressions, p=0.030). The strict augmentation rule was nevertheless not met
because ROC-AUC decreased by 0.36 points.

Neither native condition passed the pre-frozen deployment gate. The GAN candidate
detected 239 of 300 phishing rows (79.67% recall), missing the required 80% by one
row. It is therefore the strongest research candidate in this evaluation, but it
is not promoted as deployment-ready and the existing v0.5.1 extension remains
unchanged.

The final dataset may not be used to change thresholds, model checkpoints,
preprocessing, features or tiers.
