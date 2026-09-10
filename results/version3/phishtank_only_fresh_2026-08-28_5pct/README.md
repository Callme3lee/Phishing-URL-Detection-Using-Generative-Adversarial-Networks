# PhishTank-only fresh external evaluation

This experiment uses the frozen 10,000-row PhishTank/Tranco snapshot collected on 2026-08-28: 9,500 Tranco legitimate-domain proxies and 500 PhishTank community-verified, online phishing URLs. OpenPhish is excluded. URL, hostname, and 128-token identities have zero measured overlap with prior project data.

## Frozen research comparison

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted HYB-MS real-only | 98.00% | 76.88% | 85.80% | 81.10% | 97.22% | 87.24% | 1.36% | 14.20% |
| Weighted HYB-MS +10% GAN | **98.45%** | **81.89%** | **88.60%** | **85.11%** | **97.30%** | **91.11%** | **1.03%** | **11.40%** |

The registered augmentation rule passed: +10% improved recall, F1, and ROC-AUC while reducing FPR. It made 71 corrections and 26 regressions relative to real-only. The overall paired exact McNemar p-value is `5.4586e-06`.

## Exact browser-extension evaluation

| Operating point | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Warning or block | 98.94% | 93.58% | 84.60% | 88.87% | 97.89% | 92.08% | 0.31% | 15.40% |
| Caution or higher | 98.48% | 81.41% | 90.20% | 85.58% | 97.89% | 92.08% | 1.08% | 9.80% |
| Secondary review or higher | 92.07% | 38.29% | 95.80% | 54.71% | 97.89% | 92.08% | 8.13% | 4.20% |

The extension produced 29 strict false alerts among 9,500 legitimate proxies and detected 423 of 500 phishing URLs at warning-or-block. At caution-or-higher it detected 451 of 500 phishing URLs with 103 legitimate cautions.

## Source diagnosis

On the preceding mixed-source 10K snapshot, the same packaged extension detected 38.16% of the 283 OpenPhish rows but 80.18% of the 217 PhishTank rows. On this new PhishTank-only snapshot it detected 84.60% at the same warning threshold. This establishes a strong source-specific mismatch in the previous result. It does not by itself prove that every missed OpenPhish label was wrong; the OpenPhish rows may be noisier, structurally different, or both.

## Files

- `research_pair/results.json`: complete research metrics, deltas, hashes, confidence intervals, and confusion matrices
- `research_pair/prospective_comparison.pdf`: generated comparison report
- `research_pair/frozen_predictions.npz`: frozen paired predictions
- `browser_extension/browser_evaluation.json`: complete packaged-extension metrics and policy hashes
- `browser_extension/tier_counts.csv`: allow/review/caution/warning/block counts
- `browser_extension/browser_predictions.npz`: frozen packaged-extension scores and tiers

No training, calibration, threshold adjustment, checkpoint selection, or post-test row selection was performed.
