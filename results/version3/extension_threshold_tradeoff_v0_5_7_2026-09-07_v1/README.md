# Extension threshold trade-off study

## Scope and protocol

This is a controlled operating-point analysis of the existing HYB-MS + 10% MS-HardGAN-C browser ensemble. It does not retrain or replace the model and does not modify the deployed extension policy.

- Threshold selection used only the frozen calibration/validation cohort: 23,100 URLs (20,600 legitimate and 2,500 phishing).
- The dense sweep evaluated all 22,449 distinct score operating points.
- The frozen 10,000-URL set (9,700 legitimate and 300 phishing) was evaluated only after the 1%-FPR operating point had been selected.
- Predictions use `score >= threshold` as phishing.
- All dataset and prediction hashes are recorded in `analysis.json`.

## What controls the trade-off

Each of the three neural checkpoints emits one logit. A sigmoid is embedded once in each exported ONNX model, and the extension averages the three FP32 sigmoid scores. The 35 lexical features are normalized using the stored centre and scale. These scores are ranking scores, not calibrated probabilities.

The thresholds in `models/policy.json` convert that continuous score into tiers. Lowering a threshold labels more URLs as phishing: true positives increase and false negatives decrease, but false positives can also increase. The full URL score controls the decision; the query-stripped score is diagnostic only.

Extension post-processing can change the user-visible action without changing the model score. Exact user trust, reviewed service/path rules, IP handling and passive mode can suppress interruption. In normal mode only the `block` tier interrupts navigation. Therefore model predictions and extension actions must not be treated as interchangeable measurements.

## Validation operating points

| Maximum FPR | Threshold | FPR | FNR | Precision | Recall | F1 | Specificity | Accuracy | TP | TN | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.995307744 | 0.50% | 20.68% | 95.06% | 79.32% | 86.48% | 99.50% | 97.32% | 1,983 | 20,497 | 103 | 517 |
| 1% | 0.970766783 | 1.00% | 12.24% | 91.42% | 87.76% | 89.55% | 99.00% | 97.78% | 2,194 | 20,394 | 206 | 306 |
| 2% | 0.885774910 | 1.99% | 8.08% | 84.86% | 91.92% | 88.25% | 98.01% | 97.35% | 2,298 | 20,190 | 410 | 202 |
| 3% | 0.801366985 | 2.97% | 6.76% | 79.23% | 93.24% | 85.67% | 97.03% | 96.62% | 2,331 | 19,989 | 611 | 169 |
| 5% | 0.670751750 | 4.96% | 5.24% | 69.86% | 94.76% | 80.43% | 95.04% | 95.01% | 2,369 | 19,578 | 1,022 | 131 |

The 1%-FPR point is recommended for a recall-oriented warning. Relative to the current 0.5%-FPR threshold, validation FPR increases by 0.50 percentage points (100% relative), while FNR decreases by 8.44 points (40.81% relative). Precision decreases by 3.65 points, recall increases by 8.44 points, and F1 increases by 3.07 points.

This threshold is already the extension's `caution` boundary. It should not replace the much stricter automatic-block boundary. The scientifically defensible improvement is to treat `caution` as the recall-oriented detection/warning operating point while retaining a strict block policy.

## Frozen 10K confirmation

| Operating point | Threshold | FPR | FNR | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 0.995307744 | 0.39% | 23.67% | 85.77% | 76.33% | 80.78% | 229 | 9,662 | 38 | 71 |
| Validation-selected recommendation | 0.970766783 | 1.24% | 14.00% | 68.25% | 86.00% | 76.11% | 258 | 9,580 | 120 | 42 |

The frozen confirmation moved in the expected direction without test-set tuning. FNR decreased by 9.67 percentage points (40.85% relative), while FPR increased by 0.85 points (215.79% relative). The external FPR is 1.24%, rather than exactly 1%, because the 1% constraint was imposed on validation and the later frozen cohort has a different distribution. Precision and F1 declined on this low-prevalence set, so this is a safety/recall choice rather than a claim that every metric improves.

## Score separation

Validation ROC-AUC is 0.9803 and PR-AUC is 0.9271, showing strong overall ranking. Nevertheless, the score-distribution figure reveals tail overlap: some legitimate URLs receive very high scores and some phishing URLs receive low scores. Threshold tuning can materially reduce FNR, but it cannot eliminate this overlap.

If further improvement at the same FPR is required, the smallest justified training experiment is hard-example sampling with the existing architecture and BCE-with-logits pipeline. It should add difficult real phishing positives and difficult legitimate login/query negatives, keep the current frozen splits, compare all seeds, and recalibrate on a separate validation cohort. Broad model redesign is not yet justified.

## Identical-result audit

- Native GAN versus v0.5.2: scores are not bit-identical (maximum absolute difference 0.001097), but all 10,000 binary predictions are identical under their separately derived native/ONNX thresholds. This is consistent with CUDA/ONNX numerical portability, not cached evaluation.
- v0.5.1 versus v0.5.7: scores and predictions are exactly identical. All three ONNX hashes, calibration-score hashes and the primary threshold match. This is legitimate because v0.5.7 did not replace the neural checkpoints; it changed extension reliability, service rules and UI.
- The v0.5.7 ONNX evaluation was rerun into a new dated artifact. No evidence of stale cached predictions or ineffective version selection was found.

## Reproduction

From `Version_3`:

```bash
PYTHONPATH=src python -m phishing_gans_v3.analyze_extension_threshold_tradeoff
```

The command refuses to overwrite this output directory, preserving the recorded run. Use a new output identifier in the source for a future frozen experiment.
