# Graduation Project Scientific Freeze

Freeze date: 1 September 2026

## Status

The modelling and controlled-evaluation stage is complete. The 6,000-row final
test was evaluated once after the policy-development thresholds, model hashes,
statistical tests and deployment gate had been fixed.

HYB-MS with 10% MS-HardGAN-C augmentation is the strongest final research
candidate. It improved seven of the eight reported metrics over real-only
HYB-MS, but it is not described as production-ready. ROC-AUC was slightly lower,
and recall was 239/300 (79.67%), one detection below the fixed 80% deployment
gate.

The submitted browser prototype remains Version 0.5.1. Its packaged models and
policy are unchanged. The experimental v0.5.2 candidate policy was not promoted.

## Frozen scientific components

- real-only and 10% GAN HYB-MS checkpoints for seeds 42, 43 and 44;
- the final 4,000/6,000 development and test split identities;
- URL canonicalization, 128-character encoding and 35-feature lexical input;
- model ensembling and policy-development threshold rules;
- final predictions, metrics, statistical tests and deployment decision;
- the Version 0.5.1 ONNX models and browser policy.

## Prohibited after this freeze

- using the final 6K labels to lower or otherwise select a threshold;
- replacing a checkpoint or feature pipeline and reporting it on the same final set;
- changing the final test rows, labels or source balance;
- rerunning alternative candidate selection until a preferred model wins;
- describing the browser prototype as a production-certified security product.

Administrative packaging, presentation work, spelling corrections and clearly
labelled future experiments on genuinely new data remain allowed. Any future
model change must use new development data and a separate later test set.

## Authoritative records

- Final report: `Documentation/Graduation_Project_Report_Final_2026-09-01.pdf`
- Frozen protocol: `Version_3/artifacts/final_browser_policy_v0_5_2_2026-09-01/manifest.json`
- Final metrics: `Version_3/artifacts/results/final_browser_evaluation_2026-09-01_v1/results.json`
- Final evaluation PDF: `Version_3/artifacts/results/final_browser_evaluation_2026-09-01_v1/final_evaluation_report.pdf`
- Submitted extension: `Version_3/Phishing_URL_Guard_User_Pilot_v0.5.1.zip`

The exact SHA-256 values are recorded in `PROJECT_FREEZE_2026-09-01.sha256`.
