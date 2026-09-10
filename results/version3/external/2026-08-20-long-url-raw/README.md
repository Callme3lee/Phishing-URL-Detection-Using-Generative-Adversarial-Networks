# New long-URL external stress test

This frozen evaluation contains 1,550 URLs: 1,400 long legitimate-proxy URLs and 150 verified-online phishing URLs (9.68% phishing). Every legitimate URL is at least 80 characters and contains either a query or a deep path.

The two final hybrid candidates were evaluated with the same seeds (42/43/44) and their previously frozen thresholds selected at a maximum 0.5% calibration FPR. The new test labels were not used for training, threshold selection, or URL selection.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Updated HYB-MS, real only | 95.68% | 80.74% | 72.67% | 76.49% | 98.11% | 88.21% | 1.86% | 27.33% |
| Updated HYB-MS, +10% GAN | 96.06% | 81.12% | 77.33% | 79.18% | 97.92% | 86.51% | 1.93% | 22.67% |

The GAN condition caught seven more net phishing URLs and produced one more false positive. On phishing rows, its paired exact comparison was 9 uniquely corrected versus 2 uniquely worsened cases (two-sided exact McNemar p=0.065). This is promising but not conclusive at the usual 0.05 cutoff.

Important limitation: Tranco membership plus a Common Crawl 200/HTML record is a legitimate source proxy, not a manual page-level safety audit. This set deliberately stresses long URLs and does not estimate ordinary browser prevalence.

Authoritative files:

- `results.json`: metrics, frozen thresholds, checkpoint hashes, subgroup analysis and paired test.
- `predictions.csv`: row-level scores and predictions.
- `main_metrics.png`, `fpr_fnr.png`, `legitimate_length_false_positives.png`: report graphs.
- Dataset and collection record: `Version_3/data/external/2026-08-20-long-url-raw/`.
