# Exact browser-extension audit — fresh 2026-08-28 snapshot

This is an offline audit of the exact ONNX ensemble and frozen tier policy packaged in `browser_extension_user_pilot` v0.5.1. It uses the frozen 10,000-URL snapshot at `data/external/2026-08-28-fresh-raw-5pct/external_urls.csv`.

No model, feature, checkpoint, threshold, or policy tier was changed after collection.

## Primary warning-or-block result

The extension treats the `warning` boundary as its strict binary phishing decision. At that existing threshold:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 97.58% | 92.16% | 56.40% | 69.98% | 84.35% | 74.03% | 0.25% | 43.60% |

Confusion counts: TN 9,476; FP 24; FN 218; TP 282.

This conservative browser threshold produces few strong false warnings, but misses many phishing URLs as strict warning-or-block decisions. Some of those phishing URLs still receive lower `secondary_review` or `caution` results.

## Tier distribution

| Tier | Legitimate | Phishing | Total |
|---|---:|---:|---:|
| Allow | 8,730 | 106 | 8,836 |
| Secondary review | 675 | 46 | 721 |
| Caution | 71 | 66 | 137 |
| Warning | 23 | 144 | 167 |
| Block | 1 | 138 | 139 |

Only one of 9,500 legitimate proxies reached the automatic block tier. A legitimate URL in `caution` is an unnecessary caution, not a strict classifier false positive. If caution-or-higher is counted as an alert, FPR is 1.00% and recall is 69.60%.

## Files

- `browser_evaluation.json`: complete metrics, hashes, thresholds, tier counts, and interpretation
- `tier_counts.csv`: simple table of legitimate/phishing counts by tier
- `browser_predictions.npz`: frozen labels, scores, and tier assignments

The raw phishing feeds may contain live malicious URLs. Do not open them manually in a normal browser.
