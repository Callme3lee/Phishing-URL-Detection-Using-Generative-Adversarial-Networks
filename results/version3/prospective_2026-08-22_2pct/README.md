# Prospective low-prevalence comparison

Verdict: **AUGMENTATION BENEFIT SUPPORTED**

Dataset: 5,000 URLs at 2.00% phishing prevalence. The real-only and +10% weighted HYB-MS ensembles were frozen before collection and evaluated once.

| Frozen condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted HYB-MS real-only | 98.38% | 57.04% | 77.00% | 65.53% | 96.16% | 73.16% | 1.18% | 23.00% |
| Weighted HYB-MS +10% MS-HardGAN-C | **98.64%** | **62.70%** | **79.00%** | **69.91%** | **96.42%** | **76.94%** | **0.96%** | **21.00%** |

The +10% model found 79 of 100 phishing URLs and produced 47 false alerts among 4,900 legitimate URLs. The real-only control found 77 phishing URLs and produced 58 false alerts.

The predeclared directional rule passed because recall, F1, and ROC-AUC increased while FPR decreased. An exploratory exact McNemar test over paired correctness disagreements gives p = 0.0984 (33 cases improved and 20 worsened), so this single small cohort supports the direction but does not establish a conventional p < 0.05 difference.

Neither condition passed the predeclared practical gate of FPR <= 2% and recall >= 80%: both passed the FPR requirement, but recall reached only 77% and 79%. The +10% model therefore remains the stronger research/deployment candidate, not a deployment-ready model.

The dataset contains current OpenPhish URLs and current Tranco root-domain proxies with zero measured URL, hostname, or encoded-input overlap against all prior project cohorts. Root domains underrepresent long paths, parameters, redirects, and other difficult legitimate traffic.

Files:

- `prospective_comparison.pdf`: two-page visual report.
- `results.json`: complete metrics, hashes, thresholds, confidence intervals, and deltas.
- `frozen_predictions.npz`: immutable labels, scores, and decisions for paired follow-up analysis.
