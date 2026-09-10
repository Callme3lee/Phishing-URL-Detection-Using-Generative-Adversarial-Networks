# Updated HYB-MS ensembles: FP32 probability calibration

## Purpose

The recent-update internal experiment produced very low FPR and high FNR under thresholds close to 1.0. This stage tests whether that behavior was caused only by FP16 probability saturation or by a deeper low-FPR distribution-shift problem.

## Probability correction

The neural forward pass remains under CUDA FP16 autocast for efficiency. Sigmoid is now applied after converting logits to FP32:

```python
torch.sigmoid(logits.float())
```

This preserves substantially more threshold resolution near probabilities of 1.0. The same correction is applied in shared training-time validation, hybrid external evaluation, and ensemble calibration scoring.

No model weights were changed or retrained for this correction.

## Calibration cohort

`updated_models_calibration_2026-08-15_v1` contains:

- 20,600 legitimate proxy URLs;
- 2,500 verified-online PhishTank URLs;
- 23,100 unique URLs;
- 22,582 unique registered domains;
- 549 real cached Common Crawl paths, with the strict accepted structure counts recorded in the collection report;
- no training rows, protected confirmation rows, or previous evaluation identities.

A live Common Crawl expansion was attempted but stopped after repeated endpoint disconnections. The calibration therefore uses strict unseen cached Common Crawl records plus unseen Tranco roots. This limitation is preserved in the dataset report.

## Registered gate

Threshold selection uses each condition's three-seed mean FP32 score and maximizes F1 subject to:

- FPR <= 0.25%;
- recall >= 80%.

No threshold satisfies both constraints for either ensemble. The recorded fallback is the maximum-recall operating point under the FPR cap.

| Metric at FPR <= 0.25% | Updated real-only | Updated +10% GAN |
|---|---:|---:|
| Threshold | 0.999411 | 0.999849 |
| Precision | 96.30% | 95.52% |
| Recall | 52.08% | 43.48% |
| F1 | 67.60% | 59.76% |
| ROC-AUC | 98.10% | 98.03% |
| PR-AUC | 92.66% | 92.71% |
| FPR | 0.243% | 0.248% |
| FNR | 47.92% | 56.52% |

At this strict budget, real-only is better.

## Operational trade-off

| FPR cap | Real-only recall | +10% GAN recall | Real-only F1 | +10% GAN F1 |
|---|---:|---:|---:|---:|
| 0.25% | 52.08% | 43.48% | 67.60% | 59.76% |
| 0.50% | 72.12% | 79.28% | 81.84% | 86.46% |
| 1.00% | 85.84% | 87.72% | 88.57% | 89.62% |

The GAN condition becomes better once the false-positive budget is relaxed. At 0.50% FPR it nearly reaches the 80% recall goal, but the budget cannot be changed merely to make the GAN result favorable.

At an assumed 1% phishing prevalence, projected precision declines as FPR is relaxed:

| FPR cap | Real-only projected precision | +10% GAN projected precision |
|---|---:|---:|
| 0.25% | 68.43% | 63.95% |
| 0.50% | 59.30% | 61.56% |
| 1.00% | 47.18% | 47.59% |

## Conclusion

FP32 sigmoid corrects score quantization but does not solve the high-FNR problem. The ranking metrics remain strong, yet phishing and legitimate scores overlap too much in the extreme low-FPR region. If the browser extension must retain a 0.25% FPR budget, further improvement requires better temporal/source representation and low-FPR-oriented training—not another threshold-only adjustment.

The protected failed confirmation set remains untouched and must not be rerun. A future operational FPR budget must be selected before a new later confirmation snapshot is evaluated.
