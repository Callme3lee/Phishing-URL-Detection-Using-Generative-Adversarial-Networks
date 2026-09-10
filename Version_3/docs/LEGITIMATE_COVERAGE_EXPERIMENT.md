# Legitimate Structural-Coverage Experiment

## Research question

Does adding a small, independently sourced set of structurally varied legitimate URLs to training reduce external false positives without damaging phishing detection?

This is a controlled real-data experiment, not GAN augmentation. The original Version 3 pipeline, model architectures, hyperparameters, seeds, validation/test rows, and threshold policy remain unchanged.

## Training intervention

| Item | Count |
|---|---:|
| Original training rows | 358,619 |
| Audited coverage-training candidates | 3,042 |
| Rows excluded after URL-only review | 30 |
| Accepted coverage rows added once | 3,012 |
| Final experimental training rows | 361,631 |
| Coverage share of final training | 0.833% |

The 3,012 added rows contain 760 roots, 788 shallow paths, 746 deep paths, and 718 query URLs. They were taken only from the Tranco-domain-grouped training manifest. No coverage validation/test row was promoted.

An 80-row stratified sample—20 per structure—received URL-text and provenance review. Seventy-five reviewed URLs were accepted as plausible proxy-labelled legitimate development rows. Five questionable domains were conservatively excluded in full, removing 30 rows: `anuytzc.xyz`, `danatoto.com`, `ebun.top`, `exloader.net`, and `thatpervert.com`. This was not a page-content safety audit, so labels remain explicitly described as popular-domain legitimate proxies.

Every accepted row was appended exactly once. There was no oversampling, weighting, threshold tuning, or result-driven selection.

## Unchanged internal test

Mean ± sample standard deviation across seeds 42/43/44:

| Model | Condition | Accuracy | Recall | F1 | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|
| CNN | Original | 0.9889 ± 0.0013 | 0.9725 ± 0.0023 | 0.9736 ± 0.0031 | 0.9970 ± 0.0008 | 0.0067 ± 0.0014 |
| CNN | + coverage | 0.9893 ± 0.0008 | 0.9725 ± 0.0010 | 0.9744 ± 0.0018 | 0.9972 ± 0.0001 | 0.0063 ± 0.0008 |
| LSTM | Original | 0.9898 ± 0.0003 | 0.9696 ± 0.0012 | 0.9756 ± 0.0006 | 0.9968 ± 0.0004 | 0.0048 ± 0.0006 |
| LSTM | + coverage | 0.9894 ± 0.0003 | 0.9687 ± 0.0022 | 0.9745 ± 0.0006 | 0.9971 ± 0.0004 | 0.0051 ± 0.0008 |
| RNN | Original | 0.9866 ± 0.0017 | 0.9629 ± 0.0042 | 0.9679 ± 0.0041 | 0.9939 ± 0.0008 | 0.0071 ± 0.0012 |
| RNN | + coverage | 0.9822 ± 0.0048 | 0.9615 ± 0.0033 | 0.9578 ± 0.0111 | 0.9932 ± 0.0014 | 0.0123 ± 0.0058 |
| Random Forest | Original | 0.9496 ± 0.0001 | 0.7822 ± 0.0006 | 0.8667 ± 0.0002 | 0.9627 ± 0.0003 | 0.0060 ± 0.0003 |
| Random Forest | + coverage | 0.9487 ± 0.0004 | 0.7765 ± 0.0033 | 0.8639 ± 0.0015 | 0.9628 ± 0.0011 | 0.0056 ± 0.0004 |
| Linear SVM | Original | 0.8975 ± 0.0000 | 0.5280 ± 0.0000 | 0.6835 ± 0.0000 | 0.9298 ± 0.0000 | 0.0045 ± 0.0000 |
| Linear SVM | + coverage | 0.8991 ± 0.0000 | 0.5367 ± 0.0000 | 0.6904 ± 0.0000 | 0.9299 ± 0.0000 | 0.0048 ± 0.0000 |

Internal changes are small and mixed. This is expected because 99.17% of the experimental training rows retain the original distribution. CNN changes slightly favorably; LSTM is essentially stable; RNN becomes more seed-sensitive; RF trades recall for a small FPR reduction; SVM gains recall and F1 slightly.

## Second frozen external set

The confirmatory source-transfer set was frozen before evaluation:

- 200 OpenPhish Community phishing URLs;
- 200 Tranco `Q2XX4` legitimate root proxies;
- 400 unique normalized URLs, encodings, and hostnames;
- zero overlap with the original Version 3 dataset, the full 3,800-row coverage pool, and all earlier external snapshots;
- frozen SHA-256: `c31791b5d93722d66b306b12ff7c7857909539191036e25c5931f7b7be120793`.

This changes the phishing source from PhishTank to OpenPhish, but it is a same-date source-transfer test—not a later temporal snapshot. The [OpenPhish Community feed](https://www.openphish.com/phishing_feeds.html) is free but limited, and [Tranco](https://tranco-list.eu/list/Q2XX4) membership is a legitimate proxy rather than manual page adjudication.

### Results on OpenPhish/Tranco

| Model | Condition | Accuracy | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | Original | 0.4458 ± 0.0014 | 0.8850 ± 0.0087 | 0.6149 ± 0.0029 | 0.5737 ± 0.0183 | 0.9933 ± 0.0058 | 0.1150 ± 0.0087 |
| CNN | + coverage | 0.4283 ± 0.0095 | 0.8283 ± 0.0176 | 0.5916 ± 0.0090 | 0.5814 ± 0.0248 | 0.9717 ± 0.0058 | 0.1717 ± 0.0176 |
| LSTM | Original | 0.4317 ± 0.0361 | 0.8517 ± 0.0881 | 0.5988 ± 0.0410 | 0.6857 ± 0.0874 | 0.9883 ± 0.0161 | 0.1483 ± 0.0881 |
| LSTM | + coverage | **0.5075 ± 0.0218** | **0.8983 ± 0.0029** | **0.6460 ± 0.0096** | **0.8369 ± 0.0124** | **0.8833 ± 0.0454** | **0.1017 ± 0.0029** |
| RNN | Original | 0.4467 ± 0.0014 | 0.8917 ± 0.0058 | 0.6171 ± 0.0022 | 0.7534 ± 0.0180 | 0.9983 ± 0.0029 | 0.1083 ± 0.0058 |
| RNN | + coverage | 0.4575 ± 0.0066 | 0.9000 ± 0.0100 | 0.6239 ± 0.0042 | 0.7763 ± 0.0399 | 0.9850 ± 0.0150 | 0.1000 ± 0.0100 |
| Random Forest | Original | 0.0975 ± 0.0043 | 0.1700 ± 0.0087 | 0.1585 ± 0.0074 | 0.0715 ± 0.0004 | 0.9750 ± 0.0000 | 0.8300 ± 0.0087 |
| Random Forest | + coverage | 0.1875 ± 0.0109 | 0.1583 ± 0.0104 | 0.1630 ± 0.0071 | 0.1371 ± 0.0029 | 0.7833 ± 0.0321 | 0.8417 ± 0.0104 |
| Linear SVM | Original | 0.3325 ± 0.0000 | 0.0950 ± 0.0000 | 0.1246 ± 0.0000 | 0.1286 ± 0.0000 | 0.4300 ± 0.0000 | 0.9050 ± 0.0000 |
| Linear SVM | + coverage | 0.3300 ± 0.0000 | 0.0950 ± 0.0000 | 0.1242 ± 0.0000 | 0.1291 ± 0.0000 | 0.4350 ± 0.0000 | 0.9050 ± 0.0000 |

LSTM is the clear best experimental model on this set. Relative to original LSTM, coverage increases mean accuracy by 7.58 percentage points, recall by 4.67 points, F1 by 4.72 points, and ROC-AUC by 15.12 points while reducing FPR by 10.50 points. The same direction appears on the earlier PhishTank/Top-Domain set: LSTM FPR falls by 12.80 points and ROC-AUC increases by 6.55 points with only a 0.40-point recall reduction.

These are meaningful directional changes across both external sources, but they are not sufficient for deployment. LSTM still falsely blocks 88.3% of new Tranco roots. RF's FPR improves more, but its ROC-AUC remains far below 0.5 and phishing recall is only 15.8%, so it is not a viable alternative. CNN loses 5.67 points of OpenPhish recall while gaining only 2.17 points of FPR.

## What caused the remaining shift?

The accepted rows are too small a share of training to substantially change its structure. Experimental training legitimate URLs still average 50.66 characters and 28.12 path characters; the new external Tranco roots average only 14.31 and 1.00. Internal legitimate hosts average 2.28 dots, while external legitimate roots average 1.07 and OpenPhish hosts average 2.05. The original source relationship therefore remains dominant.

The phishing side also shifts. Neural recall was approximately 95.9–96.7% on PhishTank but falls to roughly 85–89% for the original models on OpenPhish. Thus the main operational failure is legitimate false positives, but phishing generalization is not solved and depends on source.

## Conclusion

The controlled experiment supports a limited claim: **a small amount of structurally targeted real legitimate data can improve external ranking and reduce false positives, especially for LSTM, without destroying internal performance.** It does not support a deployment claim, and it does not prove that 3,012 rows are enough.

The best next experiment is not GAN augmentation. It is a pre-registered real-data mixture experiment that gives legitimate roots and realistic benign paths materially greater training representation while retaining diverse phishing sources. Use fixed mixture proportions or batch sampling decided before evaluation, and collect a genuinely later snapshot on another date. Only after the real distribution is controlled should GAN-generated phishing be tested as an orthogonal robustness intervention.

With three seeds and one small confirmatory set, deltas are descriptive. They should not be described as statistically conclusive.

## Reproducible artifacts

- Experiment freeze: `data/experiments/legitimate_coverage_v1/experiment_report.json`
- Review sample: `data/experiments/legitimate_coverage_v1/manual_review_sample.csv`
- Internal comparison: `artifacts/results/legitimate_coverage_v1/internal/`
- New external data: `data/external/2026-08-11-openphish-tranco/`
- New external comparison: `artifacts/results/legitimate_coverage_v1/external_openphish_tranco/comparison/`
- Earlier external diagnostic: `artifacts/results/legitimate_coverage_v1/external_topdomains_phishtank/comparison/`
