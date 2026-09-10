# Final frozen-model comparison

## Method

The weighted HYB-MS real-only ensemble and weighted HYB-MS +10% MS-HardGAN-C ensemble were frozen before the new datasets were scored. Their three seed checkpoints, hashes, calibration thresholds, sample counts, cohort sizes, statistical tests, and final decision rule were registered in `artifacts/final_confirmation_protocol_2026-08-22/manifest.json`.

No model was retrained. No threshold, checkpoint, URL, or decision rule was changed after inference began. Neural inference ran on the NVIDIA GPU.

## Primary 30K confirmation

The primary cohort contains 28,500 legitimate and 1,500 phishing URLs. Every selected URL, hostname, and 128-token encoded input is disjoint from all earlier training, development, calibration, and external cohorts.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYB-MS real-only | 97.85% | 77.16% | 81.07% | 79.06% | 96.58% | 84.29% | 1.26% | 18.93% |
| HYB-MS +10% GAN | **98.12%** | **80.95%** | **81.60%** | **81.27%** | **96.82%** | **87.27%** | **1.01%** | **18.40%** |

The +10% condition passed the primary directional rule. It made 240 unique corrections and 160 unique regressions relative to real-only. The overall paired exact McNemar result was significant (`p = 0.0000743`). Its F1 improvement had a stratified paired-bootstrap 95% interval of +1.05 to +3.41 percentage points, and its FPR change had an interval of -0.365 to -0.140 percentage points.

The recall change was small: +0.53 percentage points with a bootstrap interval of -0.87 to +2.00 points. The phishing-only paired comparison was not significant (`p = 0.526`). The reliable primary benefit is therefore better precision/F1 and fewer legitimate false alerts, not a proven large increase in phishing recall.

## Long and complex URL stress test

The separate stress cohort contains 2,000 legitimate Common Crawl URLs of at least 80 characters with deep paths or queries, plus 200 phishing URLs. It is deliberately difficult and is not a browser-prevalence estimate.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYB-MS real-only | **96.95%** | **88.00%** | 77.00% | 82.13% | **98.51%** | **92.45%** | **1.05%** | 23.00% |
| HYB-MS +10% GAN | 96.86% | 84.66% | **80.00%** | **82.26%** | 98.42% | 91.39% | 1.45% | **20.00%** |

The +10% model caught six more net phishing URLs but produced eight more net legitimate false alerts. Its FPR increase was +0.40 percentage points, with a bootstrap 95% interval of 0.00 to +0.80 points. The legitimate-only paired comparison was not conventionally significant (`p = 0.0768`), but the registered non-regression rule required the GAN FPR to be no higher, regardless of significance. That rule failed.

## Final decision

The formal frozen-rule outcome is to **retain weighted HYB-MS real-only as the conservative unconditional final reference**.

Weighted HYB-MS +10% MS-HardGAN-C remains the **best ordinary low-prevalence candidate** because it won the larger 30K confirmation with a significant overall paired improvement. It is not selected as the unconditional winner because it increased false alerts on the separate long-URL stress test.

This is an inconclusive deployment trade-off, not a failed GAN experiment:

- GAN augmentation is supported for the ordinary low-prevalence source mixture.
- GAN augmentation does not provide uniform false-positive improvement across difficult legitimate URL structures.
- The project must report both results instead of choosing a threshold or test set that favors one condition.
- Neither model is deployment-ready until browser latency, memory, live alert burden, failure handling, and drift monitoring are validated.

The combined PDF is `artifacts/results/final_model_decision_2026-08-22/final_model_decision.pdf`.
