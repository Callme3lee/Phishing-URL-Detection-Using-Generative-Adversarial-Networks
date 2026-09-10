# Conservative tiered URL policy

## Status

`conservative_hyb_gan10_v1` is a frozen **offline research deployment candidate**, not a deployment-ready browser extension. It uses the updated three-seed HYB-MS +10% MS-HardGAN-C mean-score ensemble. Its primary threshold was selected from calibration data under a 0.5% FPR ceiling; the previously viewed 20K identity-disjoint cohort is used only for diagnostic reproduction.

## Frozen primary operating point

| Metric | 20K diagnostic result |
|---|---:|
| Accuracy | 97.55% |
| Precision | 94.83% |
| Recall | 79.85% |
| F1 | 86.70% |
| ROC-AUC | 97.52% |
| PR-AUC | 92.88% |
| FPR | 0.483% |
| FNR | 20.15% |

The primary binary threshold is `0.9953193664550781`. A score is a model-ranking output, not a literal probability or percentage of trust.

## Tier policy

| Tier | Score interval | Action | 20K rows | Legitimate | Phishing |
|---|---|---|---:|---:|---:|
| Allow | `< 0.670915127` | Allow normally | 17,113 | 16,985 | 128 |
| Secondary review | `[0.670915127, 0.970765769)` | Request independent evidence; do not block from URL score alone | 941 | 816 | 125 |
| Caution | `[0.970765769, 0.995319366)` | Passive caution; independent evidence may escalate | 262 | 112 | 150 |
| Warning | `[0.995319366, 0.999848783)` | Strong phishing warning | 756 | 74 | 682 |
| Block | `>= 0.999848783` | Interstitial block with explicit override | 928 | 13 | 915 |

The 128 phishing URLs in the Allow tier demonstrate the URL-only representation ceiling. Threshold changes cannot recover them without moving many legitimate URLs upward. The next new-model experiment should implement independent evidence for the Secondary review tier using new development data.

## Reproducibility result

The inference engine rescored all 20,000 URLs on an NVIDIA GeForce RTX 3060 Laptop GPU:

- batch inference throughput: approximately 20,560 URLs/s;
- maximum absolute difference from saved FP16 scores: `0.0000402331`;
- declared FP16/CUDA tolerance: `0.0001`;
- changed binary decisions: `0`;
- changed risk tiers: `0`;
- tests: 12 passed.

The candidate has now been exported to ONNX and integrated into a Manifest V3 research extension. Repeated Node-hosted ONNX Runtime Web checks on this development machine give 167–197 ms initialization, 7–9 ms mean and 17–27 ms p95 sequential URL inference over 12 fixtures. Interactive Chrome latency, memory, concurrency, battery use and cross-device behavior remain unmeasured. See `docs/BROWSER_EXTENSION.md`.

## Run the frozen policy

```bash
PYTHONPATH=src python -m phishing_gans_v3.tiered_url_inference \
  --input urls.csv \
  --output tiered_predictions.csv
```

The input must contain a `url` column. If it also contains a binary `label` column, the report includes full classification metrics.

## Preprocessing correction

The end-to-end run identified eight scheme-free URLs containing redirect targets such as `?target=https://...`. The prior canonicalizer incorrectly treated any `://` anywhere in the string as the outer URL scheme. It now recognizes a scheme only at the beginning of the input. No evaluation row was removed or relabelled, and a regression test protects this case.

## Next controlled experiment

1. Keep this policy, checkpoints and thresholds frozen.
2. Collect new development-only DNS/domain-age, redirect, TLS/reputation or page-content evidence for Secondary review URLs.
3. Train and calibrate the secondary verifier without using the 20K evaluation labels.
4. Lock the complete decision rule.
5. Evaluate once on a genuinely later, low-prevalence cohort.
