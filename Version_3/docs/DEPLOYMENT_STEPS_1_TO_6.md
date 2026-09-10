# Deployment validation: steps 1–6

## Outcome

The controlled deployment-validation cycle is complete. The recovery model passed the development calibration gate but failed the untouched post-threshold confirmation gate. It must therefore remain an **offline research candidate**, not be described as deployment-ready.

## 1. Frozen starting model

- Model: three-seed weighted `HYB-MS + 10% MS-HardGAN-C` mean-score ensemble.
- Seeds: 42, 43, and 44.
- Every checkpoint was fixed by SHA-256 before calibration.
- Earlier operating threshold: `0.9890950322151184`.

Freeze manifest: `data/deployment/2026-08-15-precision/frozen_model_manifest.json`.

## 2. Identity-disjoint deployment calibration data

- 24,000 URLs: 21,500 legitimate proxies and 2,500 verified-online phishing URLs.
- 22,515 unique registered domains.
- Legitimate composition: 20,000 Tranco roots and 1,500 Common Crawl paths.
- Exact URL, 128-character encoding, hostname, and registered-domain overlaps were excluded against training, validation, test, prior calibration, previous external cohorts, and prior development coverage.
- Calibration prevalence is methodological; it is not claimed to be browser prevalence.

Dataset report: `data/deployment/2026-08-15-precision/calibration_report.json`.

## 3. Threshold trade-off and registered gate

The rule was fixed before confirmation: maximize calibration F1 while requiring FPR <= 0.25% and recall >= 80%.

The original ensemble reached 78.80% recall at 0.247% FPR, so it missed the recall gate. Its calibration ROC-AUC was 97.01%.

## 4. False-positive analysis

At the low-FPR calibration threshold, the original ensemble made 53 false positives:

| Legitimate structure | Rows | False positives | FPR |
|---|---:|---:|---:|
| Root | 20,000 | 29 | 0.145% |
| Shallow path | 138 | 3 | 2.174% |
| Deep path | 1,318 | 16 | 1.214% |
| Query URL | 44 | 5 | 11.364% |

The path/query estimates have much smaller denominators, but they identify the clearest weakness: complex legitimate URL structure causes more false alerts than root domains.

Full analysis: `artifacts/results/deployment_precision_2026-08-15/calibration_analysis.json` and `false_positives.csv`.

## 5. Triggered recovery retraining

Because the gate failed, retraining was required. The original 404,102 stored rows were preserved and supplemented with a separate development pool:

- 5,000 newer, identity-disjoint PhishTank phishing domains;
- 317 identity-disjoint legitimate domains, including 92 real Common Crawl pages and 225 Tranco roots;
- zero calibration rows;
- the registered 10% synthetic exposure was retained by weighted sampling.

Three new GPU models were trained from scratch with seeds 42/43/44. Recalibration selected threshold `0.9951171875` and passed the development gate:

| Metric | Recovery calibration |
|---|---:|
| Accuracy | 97.76% |
| Precision | 97.44% |
| Recall | 80.60% |
| F1 | 88.22% |
| ROC-AUC | 97.54% |
| PR-AUC | 93.34% |
| FPR | 0.247% |
| FNR | 19.40% |

Calibration report: `artifacts/results/deployment_recovery_v1/calibration/deployment_precision_calibration.pdf`.

## 6. One-shot untouched confirmation

After the recovery threshold was locked, fresh OpenPhish and Tranco files were downloaded. Strict overlap removal left 146 phishing domains. They were combined with 14,454 unseen Tranco domains, giving exactly 14,600 URLs at 1% phishing prevalence. The selected model and threshold were applied once.

| Metric | Untouched confirmation |
|---|---:|
| Accuracy | 99.20% |
| Precision | 61.42% |
| Recall | 53.42% |
| F1 | 57.14% |
| ROC-AUC | 84.74% |
| PR-AUC | 49.95% |
| FPR | 0.339% |
| FNR | 46.58% |

Confusion counts: TN 14,405, FP 49, FN 68, TP 78. This corresponds to about 33.6 false alerts, 53.4 detected phishing URLs, and 46.6 missed phishing URLs per 10,000 visits in this cohort.

The confirmation failed both gates: recall was below 80% and FPR exceeded 0.25%. The result was not used for threshold adjustment or retraining.

Final report: `artifacts/results/deployment_confirmation_2026-08-15/deployment_confirmation_report.pdf`.

## Scientific conclusion

The cycle demonstrates why a development calibration pass is not sufficient evidence for deployment. Adding newer recovery data improved the controlled calibration result, but transfer to a genuinely untouched OpenPhish cohort remained weak. The main unresolved issue is phishing-source and temporal distribution shift, with a secondary legitimate path/query false-positive problem. Future work needs a prospectively collected multi-day development window and a later, separately locked confirmation window; this failed confirmation cohort must never become a tuning set for a claim based on the same test.
