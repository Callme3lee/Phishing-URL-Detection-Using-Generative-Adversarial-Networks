# FPR and FNR Audit

## Definitions

- **FPR = FP / (FP + TN):** the fraction of legitimate URLs incorrectly labelled phishing.
- **FNR = FN / (FN + TP):** the fraction of phishing URLs incorrectly labelled legitimate. FNR is exactly `1 - recall` here.
- Lower is better for both. A lower FNR is not operationally useful if FPR becomes unacceptably high.

This audit covers all 150 aggregated cells: three evaluations × ten conditions × five classifiers. Each cell is the mean ± sample standard deviation across seeds 42/43/44. The accompanying CSV contains every FPR and FNR value.

## First comparison: URLs_New real-only versus SeqGAN augmentation

### Internal test

| Model | Real FPR / FNR | 10% | 20% | 30% | 40% | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| CNN | 0.67% / 2.75% | 0.52% / 2.87% | 0.45% / 3.21% | 0.37% / 3.52% | 0.47% / 3.68% | Lower FPR, but steadily worse FNR |
| LSTM | 0.48% / 3.04% | 0.52% / 3.60% | 0.39% / 3.62% | 0.37% / 3.42% | 0.47% / 3.52% | No ratio improves both |
| RNN | 0.71% / 3.71% | 0.59% / 4.01% | 1.03% / 5.73% | 0.69% / 4.49% | 0.71% / 4.74% | Higher missed-phishing rate; 20% unstable |
| RF | 0.60% / 21.78% | 0.60% / 22.56% | 0.54% / 23.55% | 0.54% / 24.53% | 0.53% / 25.72% | FNR worsens monotonically |
| SVM | 0.45% / 47.20% | 0.43% / 46.18% | 0.44% / 46.04% | 0.47% / 46.14% | 0.50% / 46.39% | 10–20% improve both rates slightly, but ROC-AUC decreases |

The SVM is the only repeatable internal exception. At 20%, FPR falls by 0.015 percentage points and FNR by 1.15 points, but ROC-AUC decreases from 0.9298 to 0.9251. This is a small threshold-specific improvement, not a universal augmentation victory.

### External source-transfer tests

On Top-Domain/PhishTank, **no SeqGAN ratio improves both FPR and FNR for any classifier** relative to its matching baseline.

On Tranco/OpenPhish, two apparent exceptions occur:

| Condition | FPR change | FNR change | Other evidence | Decision |
|---|---:|---:|---|---|
| CNN + 10% | −0.50 points | −0.50 points | FPR still 98.83%; ROC-AUC −0.40 points | Not operationally useful |
| LSTM + 30% | −0.83 points | −2.67 points | FPR still 98.00%; ROC-AUC collapses from 0.6857 to 0.4694 | Reject |

These external exceptions do not demonstrate robustness because almost every legitimate URL is still rejected and ranking quality does not improve.

## Coverage-only versus coverage plus SeqGAN

On both external datasets, **no coverage-plus-SeqGAN condition improves both FPR and FNR for any model** relative to coverage-only.

Model-averaged external error rates also worsen:

| Evaluation | Condition | Macro FPR | Macro FNR |
|---|---|---:|---:|
| Top-Domain/PhishTank | Coverage only | 79.57% | 27.71% |
|  | Coverage + 10% | 85.81% | 30.05% |
|  | Coverage + 20% | 84.53% | 32.56% |
|  | Coverage + 30% | 84.27% | 33.63% |
|  | Coverage + 40% | 83.79% | 37.15% |
| Tranco/OpenPhish | Coverage only | 81.17% | 42.40% |
|  | Coverage + 10% | 87.47% | 43.07% |
|  | Coverage + 20% | 86.97% | 44.27% |
|  | Coverage + 30% | 86.93% | 44.90% |
|  | Coverage + 40% | 88.20% | 47.50% |

Macro averages combine classifiers with very different quality, so they are descriptive only. The per-model result is stronger: there is no external model/ratio where adding SeqGAN after coverage reduces both errors.

## Conclusion

1. Internal FPR is already constrained near 1% by the threshold policy, so small internal changes must not be overinterpreted.
2. SeqGAN often trades a slightly lower internal FPR for a higher FNR.
3. External FPR is the dominant failure: approximately 90–100% for most neural conditions.
4. Phishing-only GAN augmentation cannot supply the missing legitimate-source coverage needed to reduce that FPR.
5. Higher synthetic ratios generally raise FNR and frequently damage FPR as well.
6. The calibrated hybrid with real multi-source coverage remains the preferred browser-extension model; these GAN conditions should not replace it.

## Files

- `error_rates.csv`: all evaluation/condition/model mean and SD values.
- `error_rates.json`: the same values plus paired deltas and simultaneous-improvement flags.
- `fpr_fnr_comparison.png` and `.pdf`: dedicated six-panel error-rate comparison.
