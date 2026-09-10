# Frozen local cascade on the fresh 2026-08-28 snapshot

This is a one-shot transfer evaluation of the already-frozen two-stage URL cascade on the untouched 10,000-URL August 28 snapshot. No model was retrained and no threshold was selected from this test.

## Recommended matched operating point

The 0.50% calibration-budget point matches the extension's existing primary warning budget.

| System | Accuracy | Precision | Recall | F1 | FPR | FNR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Extension Stage 1 only | 97.58% | 92.16% | 56.40% | 69.98% | 0.253% | 43.60% | 24 | 218 |
| Frozen local cascade | **98.13%** | **94.59%** | **66.40%** | **78.03%** | **0.200%** | **33.60%** | **19** | **168** |

The cascade recovered 50 additional phishing URLs and removed five false alerts. Relative FNR fell by approximately 22.9%: `(43.6 - 33.6) / 43.6`.

## Architecture

- Stage 1: updated HYB-MS +10% GAN, seeds 42/43/44.
- Stage 2: independent real-only 35-feature lexical MLP, seeds 42/43/44.
- Stage 2 contains zero GAN-generated training rows.
- Only 1,220 of 10,000 URLs were routed to Stage 2.
- Both stages are local and URL-only; no DNS, reputation, page visit, redirect following, or URL upload is required.

## Other frozen operating points

| Calibration FPR budget | System | Recall | F1 | FPR | FNR |
|---:|---|---:|---:|---:|---:|
| 0.25% | Single stage | 27.60% | 43.19% | 0.011% | 72.40% |
| 0.25% | Cascade | **61.60%** | **75.49%** | 0.084% | **38.40%** |
| 1.00% | Single stage | 69.60% | 73.96% | 0.979% | 30.40% |
| 1.00% | Cascade | 69.60% | **75.73%** | **0.747%** | 30.40% |

## Interpretation

The transfer result supports the local cascade as the best current extension architecture. At the matched 0.50% budget, it improves every thresholded metric and reduces both false negatives and false positives. It does not eliminate distribution shift: 168 of 500 phishing URLs remain missed, so the extension is still a research prototype rather than production-certified protection.

The cascade has been evaluated but is not yet packaged into the distributed v0.5.1 extension.
