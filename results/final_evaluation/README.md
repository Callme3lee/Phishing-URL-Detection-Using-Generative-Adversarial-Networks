# Final Supervisor Model Evaluation

This folder preserves the two frozen evaluations completed on 7 September 2026. No model or decision threshold was fitted using either evaluation dataset.

## Evaluations

- **Complete V1 2M dataset:** 2,568,183 valid URLs, containing 1,087,717 legitimate and 1,480,466 phishing URLs. This is a phishing-heavy cross-dataset stress test, not a browser-prevalence simulation.
- **Fresh online 3% dataset:** 10,000 URLs, containing 9,700 legitimate Tranco domains and 300 verified-online PhishTank URLs. It has no exact URL overlap with the V3 real training set.

The extension result treats `caution`, `warning`, and `block` as phishing. Lower tiers are treated as non-phishing.

## Main result

On the fresh online dataset, HYB-MS +10% GAN achieved 99.00% accuracy, 79.94% precision, 89.00% recall, 84.23% F1, 98.76% ROC-AUC, 89.83% PR-AUC, 0.69% FPR, and 11.00% FNR. It outperformed the corresponding real-only hybrid on all reported metrics in this test.

The complete V1 2M evaluation produced a mixed result: GAN augmentation improved recall, FNR, F1, ROC-AUC, and PR-AUC, but increased FPR and slightly reduced precision. Dataset prevalence and source distribution must therefore be stated whenever these results are presented.

## Contents

- `dataset/online_3pct_urls.csv`: the exact online dataset used for inference.
- `dataset/dataset_manifest.json`: sampling procedure, hashes, sources, overlap check, and limitations.
- `results/evaluation_results.json`: complete machine-readable results and confusion matrices.
- `results/*_metrics.csv`: table-ready metrics.
- `graphs/`: PNG and PDF versions of the all-model and hybrid comparison figures.

## Sources

- Tranco latest standard list: <https://tranco-list.eu/top-1m.csv.zip>
- PhishTank verified-online feed: <https://data.phishtank.com/data/online-valid.csv>
