# Post-freeze 10K evaluation at 3% phishing prevalence

The unchanged frozen models and thresholds were evaluated on 10,000 URLs:
9,700 Tranco legitimate-domain proxies and 300 PhishTank verified-online
phishing URLs. The cohort has zero URL, hostname, registered-domain or
128-token encoded-input overlap with the project identity index and the original
final evaluation dataset.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYB-MS real-only | 98.71% | 77.14% | 81.00% | 79.02% | 96.50% | 85.27% | 0.74% | 19.00% | 72 | 57 |
| HYB-MS +10% GAN | 98.80% | 78.48% | 82.67% | 80.52% | 96.61% | 87.11% | 0.70% | 17.33% | 68 | 52 |
| Browser candidate | 98.80% | 78.48% | 82.67% | 80.52% | 96.61% | 87.10% | 0.70% | 17.33% | 68 | 52 |
| Installed extension v0.5.1 | 98.91% | 85.77% | 76.33% | 80.78% | 96.61% | 87.10% | 0.39% | 23.67% | 38 | 71 |

## What happened

Compared with real-only, the GAN condition detected five additional phishing
URLs and produced four fewer false alerts. It improved recall by 1.67 points,
F1 by 1.50 points and PR-AUC by 1.83 points while reducing FNR by 1.67 points.
The direction is favorable across all reported metrics, although the paired
comparison is not statistically conclusive on this cohort (two-sided exact
McNemar p = 0.368).

The installed v0.5.1 extension uses the same GAN-augmented ONNX scores with a
higher threshold. It reduced false alerts from 68 to 38 and increased precision
to 85.77%, but detected only 229 of 300 phishing URLs. The browser candidate
detected 248 of 300. This is the central operating trade-off: the installed
threshold favors fewer false warnings, while the candidate threshold favors
phishing detection.

The manually reported 100-phishing-URL test produced 70% recall and 30% FNR.
The new v0.5.1 cohort result is 76.33% recall and 23.67% FNR. The difference can
reasonably arise from the smaller manual sample and different URL difficulty;
the manual URL list was not available for an overlap or source audit.

This evaluation does not replace the original frozen 6,000-row final result.
The legitimate portion primarily contains root-domain proxies, so it does not
fully reproduce the difficult deep links, redirects and login pages encountered
during normal browsing.
