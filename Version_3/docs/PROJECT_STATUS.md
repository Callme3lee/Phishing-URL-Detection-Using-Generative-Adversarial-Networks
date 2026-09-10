# Version 3 Project Status

## Completed

- Created a separate Version 3 project without changing Version 2.
- Frozen `URLs_New.csv` by SHA-256 as the only real dataset.
- Removed the HTTP/HTTPS label shortcut from character inputs and lexical features.
- Rejected invalid and unsupported-scheme rows.
- Quarantined duplicates, ambiguous encoded collisions, and exact Version 2 label disagreements.
- Built deterministic, hostname-disjoint train/validation/test splits.
- Verified all hashes, shapes, label counts, malicious GAN indices, scheme removal, and hostname separation.
- Built 27 scheme-independent lexical features.
- Measured remaining single-feature shortcuts.
- Trained RF and linear SVM real-only baselines for seeds 42/43/44.
- Implemented a shared character CNN, character LSTM, and new vanilla tanh RNN.
- Trained CNN, LSTM, and RNN on CUDA with mixed precision for seeds 42/43/44.
- Saved validation-selected thresholds, checkpoints, histories, test predictions, and all requested metrics.
- Froze a 500-row, overlap-free Top-Domain/PhishTank external snapshot collected on 2026-08-11.
- Evaluated all five real-only models without retraining or changing thresholds.
- Confirmed severe external false-positive and structural distribution shift; the current models are not deployment-ready.
- Collected a separate 3,800-row legitimate structural-coverage development pool: 950 root, shallow, deep, and query URLs each.
- Audited the pool against Version 3, the frozen external snapshot, PhishTank, and a current 300-URL OpenPhish snapshot; all measured URL/input/hostname overlaps are zero.
- Created deterministic Tranco-domain-grouped development manifests and a lexical structural profile without changing the original training or evaluation freezes.
- Froze a controlled legitimate-coverage experiment with 3,012 accepted training-only rows and unchanged validation/test sets.
- Retrained CNN, LSTM, RNN, Random Forest, and linear SVM for seeds 42/43/44 using unchanged hyperparameters; neural training completed on CUDA.
- Froze a second 400-row, overlap-free Tranco/OpenPhish external source-transfer set and evaluated original and coverage-trained checkpoints once.
- Found consistent directional LSTM improvement across both external sets, but an 88.3% FPR on the new set remains unsuitable for deployment.
- Froze a 201,747-row `URLs_New.csv`-only structure-balanced condition with no outside or repeated rows.
- Retrained all five classifiers for seeds 42/43/44 and evaluated them on both frozen external sets.
- Confirmed that balancing reduces structural distance but not lexical source separability; neural external FPR remains approximately 97–100%.
- Established that independently sourced legitimate coverage helps LSTM more than internal structure downsampling alone.
- Frozen a 363,692-row multi-source real-data experiment while preserving the original validation/test sets and excluding all 900 external evaluation identities.
- Added a character CNN + 35 PSL-aware lexical-feature hybrid with source-balanced sampling and class-conditional CORAL alignment.
- Trained the hybrid on CUDA for seeds 42/43/44; all runs selected epoch 1 under joint original/domain validation.
- Reduced exploratory external FPR from roughly 99% to 22–25% at the comparable original-validation threshold while retaining 91–97% recall.
- Raised exploratory external ROC-AUC to 0.9704 on Top-Domain/PhishTank and 0.9276 on Tranco/OpenPhish.
- Confirmed that a stricter cross-source threshold can nearly eliminate false positives but loses too much recall; calibration remains unresolved.
- Frozen a post-model 400-row calibration set with zero URL, hostname, or encoded-input overlap against all training, development, and prior external identities.
- Froze a three-seed mean-score ensemble threshold of 0.98795575 using a predeclared maximum-1%-FPR calibration policy.
- Applied the threshold unchanged to both older external sets: exploratory FPR is now 0.8–2.0%, F1 is 87.6–92.1%, and recall is 79.5–86.0%.
- Added a confirmation-readiness guard that rejects unchanged OpenPhish/Tranco feeds and insufficient post-calibration PhishTank rows.
- Completed a formal pre-GAN audit: 15/15 baseline runs, 30/30 controlled-experiment metrics, 3/3 hash-valid hybrid checkpoints, and zero missing required artifacts.
- Classified the project as engineering-ready for GAN implementation but not yet ready for a scientific GAN-improvement claim because later temporal confirmation remains unavailable.
- Implemented and trained character-level SeqGAN and RelGAN with text-appropriate CNN discriminators on the same 82,219 frozen phishing training rows.
- Generated 500,000 URLs per GAN and applied identical TLD, validity, duplication, real-overlap, and template filters.
- SeqGAN passed the augmentation gate with 349,687 accepted unique URLs; RelGAN failed it with only 12,869 accepted URLs and clear adversarial instability.
- Measured generator uniqueness, memorization, lexical shift, cluster coverage, character n-gram distance, and challenge rates against all frozen real-only classifier families.
- Froze SeqGAN 10%, 20%, 30%, and 40% nested augmentation datasets with 39,847, 89,655, 153,694, and 239,079 generated phishing URLs.
- Trained 120 augmented classifiers across URLs_New-only and coverage-plus bases, five model families, four ratios, and seeds 42/43/44.
- Evaluated every augmented checkpoint on the unchanged internal test and both frozen external snapshots.
- Found no consistent GAN advantage: small internal CNN/SVM gains did not transfer externally, RF degraded monotonically, RNN was unstable, and high ratios generally harmed performance.
- Confirmed that phishing-only GAN augmentation does not fix the dominant legitimate-side/source distribution shift and does not replace real source coverage.
- Added `URL-GAN`, a controlled two-layer LSTM generator with Gumbel-Softmax relaxation and a character-CNN discriminator, without replacing the original GAN artifacts.
- Trained URL-GAN on the same 82,219 original phishing training URLs for five MLE and four adversarial epochs; adversarial epoch 3 was frozen as best.
- Generated and audited 500,000 URL-GAN samples: 239,368 unique rows passed the strict filter, with zero exact real-URL memorization and enough rows for all registered augmentation ratios.
- Found URL-GAN much more stable and realistic than RelGAN, but worse than the original compact SeqGAN in filter yield, lexical similarity, cluster coverage, and classifier challenge rate. SeqGAN remains the preferred augmentation generator.
- Added professor-ready short names (`SG`, `URL-GAN`, `2L-LSTM-G`, `Char-CNN-D`, `HYB`) and comparison reports without renaming or invalidating old checkpoints.
- Evaluated the calibrated three-seed hybrid ensemble on the same 49,229-row original frozen test set used by the real-only baselines and produced an eight-metric CNN/LSTM/RNN/RF/SVM/HYB comparison graph.
- Evaluated all 15 frozen real-only baseline checkpoints and the calibrated HYB ensemble on the exact 10,000-row external snapshot. HYB achieved 91.77% accuracy, 91.20% F1, 96.63% ROC-AUC, and 1.70% FPR; all five baselines remained unsuitable under external shift.
- Froze and trained `HYB-ORIG` for seeds 42/43/44 on exactly the same 358,619 original rows as the baselines, with the same validation/test splits and no outside rows, GAN rows, CORAL, or source-balanced sampling.
- HYB-ORIG achieved 99.11% internal accuracy and 99.80% ROC-AUC, but failed on the external 10K with 48.56% accuracy and 98.52% FPR. This controlled result shows that the hybrid architecture alone does not solve source shift; multi-source real coverage and adaptation are essential.
- Retrained the unchanged CNN, LSTM, RNN, RF, and linear SVM baselines for seeds 42/43/44 on the exact 363,692 stored multi-source real rows used by HYB-MS; no GAN rows were included.
- Evaluated all 15 new checkpoints on the unchanged 49,229-row internal test and frozen 10,000-row external snapshot without external threshold tuning.
- Found that ordinary multi-source training improves external ranking—especially LSTM ROC-AUC to 0.9309 and RF ROC-AUC to 0.6298—but fails to transfer a safe threshold: LSTM FPR remains 93.15%, RF FPR 70.91%, and CNN/RNN FPR approximately 99%.
- Established that HYB-MS performance is not explained by the stored row set alone; source-balanced sampling, PSL-aware fusion, conditional CORAL, ensembling, and separate calibration are material parts of that condition.
- Froze a separate 84,280-row multi-source phishing-only corpus for defensive GAN training, with zero validation/test/calibration/external/generated rows.
- Implemented MS-HardGAN with a conditioned two-layer character LSTM, Gumbel-Softmax, and a multi-head character-CNN discriminator for authenticity, eight structural families, and three source categories.
- Added a rank-normalized challenge reward from frozen CNN-MS, LSTM-MS, and RNN-MS classifiers; raw cross-family scores are never directly combined.
- Retained MS-HardGAN v1 as an explicit ablation after requested-pattern agreement reached only 28.95%.
- Trained the corrected MS-HardGAN-C with stronger pattern supervision, pattern-aware sampling, and delayed challenge training. The final 630,000-candidate audit accepted 309,208 unique rows with zero accepted real overlap, 72.22% pattern agreement, and 96.85% source agreement.
- Measured 90% real structural-cluster coverage, source-classifier ROC-AUC 0.7992, 9.95% majority challenge, and 52.83% challenge to at least one frozen classifier.
- Permanently froze 61,728 accepted rows as a synthetic challenge holdout and 247,480 as the augmentation-eligible pool.
- Froze nested MS-HardGAN-C pools of 40,410, 90,923, 155,868, and 242,461 generated URLs for exact 10%, 20%, 30%, and 40% shares with the 363,692-row MS real training set.
- Explicitly limited the claim to zero-day-like defensive stress examples; the GAN does not predict fundamentally unseen zero-day concepts.
- Materialized all four MS-HardGAN-C augmentation conditions with the same 363,692 real rows and unchanged validation/test splits.
- Trained 60 augmented classifiers: CNN, LSTM, RNN, RF, and linear SVM at four ratios for seeds 42/43/44.
- Evaluated 75 condition/model/seed runs on the unchanged internal real test and another 75 on the frozen external 10K snapshot with checkpoint thresholds unchanged.
- Found that real-only remains strongest overall internally; RF and SVM degrade with synthetic dose, while the 40% LSTM nearly matches but does not exceed real-only F1.
- Confirmed that augmentation does not solve external source shift: LSTM remains the strongest ranker, but external FPR is 89.10% even at its best 40% condition and ROC-AUC remains below real-only.
- Evaluated all 75 runs on the permanently excluded 61,728-row MS-HardGAN-C phishing challenge holdout. CNN challenge recall rises from 94.36% real-only to 98.20% at 10% and 98.95% at 40%; similar generator-specific gains occur for LSTM, RNN, and RF.
- Separated generator-specific challenge robustness from real-world generalization: the former improves, while the frozen external evidence does not support a zero-day or deployment-improvement claim.
- Audited feature fairness and confirmed that the earlier baselines and HYB-MS used the same real rows but not identical inputs or training procedures: RF/SVM had 27 lexical features, while HYB-MS had 35 PSL-aware features plus character fusion, source balancing, CORAL, domain validation, and calibration.
- Froze a feature-equal comparison in which RF-PSL, SVM-PSL, CNN+PSL, and HYB-MS receive the exact same 35 robust-scaled PSL-aware values; character-only CNN/LSTM/RNN remain explicitly labelled sequence baselines.
- Trained 57 new controlled checkpoints: 30 RF-PSL/SVM-PSL, 15 CNN+PSL, and 12 augmented HYB-MS runs. The three existing real-only HYB-MS checkpoints remain the exact control.
- Found that feature equality materially strengthens real-only RF internally (mean F1 approximately 0.945 versus 0.864 with 27 features) and SVM (0.869 versus 0.689), confirming that feature access was a substantive confound.
- Confirmed that richer features alone do not solve external shift: feature-equal RF/SVM still have severe external FPR, reaching 100% RF FPR after GAN augmentation.
- Applied the same separate 400-row calibration protocol to five three-seed HYB-MS ensembles and evaluated them on the frozen external 10K.
- Identified calibrated HYB-MS +10% as the best balanced augmentation condition: external accuracy 93.10%, precision 98.43%, recall 87.60%, F1 92.70%, ROC-AUC 96.89%, PR-AUC 97.81%, FPR 1.40%, and FNR 12.40%.
- Relative to calibrated real-only HYB-MS, +10% improves every reported external discrimination/error metric: accuracy +1.33 points, precision +0.38, recall +2.36, F1 +1.50, ROC-AUC +0.26, PR-AUC +0.31, while FPR falls 0.30 and FNR falls 2.36 points.
- Kept the deployment claim limited: HYB-MS +10% is a deployment candidate, not deployment-ready, because later temporal confirmation, realistic browser prevalence, latency/memory, alert burden, and drift monitoring remain untested.
- Pre-registered a random imbalanced 30,000-row external class count with seed 20260813 before model scoring: 20,979 Tranco legitimate proxies and 9,021 current-feed phishing URLs (30.07% phishing).
- Downloaded new official Tranco, PhishTank, and OpenPhish snapshots and froze 30,000 unique URLs, hostnames, and encoded identities with zero measured overlap against training, development, calibration, and all earlier external snapshots.
- Evaluated 30 frozen configurations spanning original baselines, ordinary multi-source baselines, feature-fair RF/SVM/CNN conditions, and calibrated HYB-MS ensembles at 0--40% GAN augmentation; no model was retrained or recalibrated on the 30K set.
- Confirmed HYB-MS +10% as the best balanced configuration: accuracy 95.43%, precision 97.49%, recall 87.05%, F1 91.98%, ROC-AUC 96.99%, PR-AUC 96.38%, FPR 0.96%, and FNR 12.95%.
- Confirmed the +10% direction relative to calibrated real-only on all eight metrics; real-only reaches 94.64% accuracy, 90.53% F1, 96.94% ROC-AUC, 1.33% FPR, and 14.75% FNR.
- Preserved the temporal limitation: PhishTank's current verified-online feed contains still-active historical reports, so the new download is identity-disjoint and later-collected but not a wholly later-created phishing cohort.
- Froze the final comparison before prospective data selection: weighted HYB-MS real-only versus weighted HYB-MS +10% MS-HardGAN-C, including checkpoint hashes, thresholds, decision rules, and a prohibition on retraining or post-test adjustment.
- Downloaded current OpenPhish, PhishTank, and Tranco snapshots on 2026-08-22 and froze a new 5,000-row browser-prevalence cohort containing 4,900 legitimate and 100 phishing URLs (2% phishing).
- Verified zero URL, hostname, and encoded-input overlap against training, validation, calibration, development, and all previous external cohorts; the selected phishing subset came from the current OpenPhish feed and the legitimate subset used current Tranco root-domain proxies.
- Evaluated both frozen three-seed ensembles exactly once on CUDA. Real-only achieved 98.38% accuracy, 57.04% precision, 77.00% recall, 65.53% F1, 96.16% ROC-AUC, 1.18% FPR, and 23.00% FNR.
- The +10% candidate achieved 98.64% accuracy, 62.70% precision, 79.00% recall, 69.91% F1, 96.42% ROC-AUC, 0.96% FPR, and 21.00% FNR. It detected two additional phishing URLs and reduced false alerts from 58 to 47.
- Passed the predeclared directional augmentation rule, but an exploratory exact McNemar comparison was not conventionally significant (p = 0.0984), and neither model passed the recall >=80% practical gate. The +10% condition remains the preferred research/deployment candidate, not deployment-ready.
- Froze an expanded final-confirmation protocol before further inference: a 30,000-row primary cohort at 5% phishing, a separate 2,200-row long-URL stress cohort, 2,000 stratified paired-bootstrap replications, exact paired tests, unchanged thresholds/checkpoints, and a two-part final-choice rule.
- Froze the 30K primary cohort with 28,500 Tranco legitimate root proxies and 1,500 OpenPhish/PhishTank phishing URLs; measured URL, hostname, and encoded-input overlap against every previous project cohort is zero.
- Confirmed the +10% GAN condition on the 30K primary cohort: accuracy 98.12%, precision 80.95%, recall 81.60%, F1 81.27%, ROC-AUC 96.82%, FPR 1.01%, and FNR 18.40%, versus real-only accuracy 97.85%, precision 77.16%, recall 81.07%, F1 79.06%, ROC-AUC 96.58%, FPR 1.26%, and FNR 18.93%.
- The 30K overall paired gain is significant (exact McNemar p=0.0000743); paired-bootstrap intervals support the F1 and FPR improvements. The small recall change is not separately significant, so the strongest confirmed benefit is precision/F1 and legitimate false-alert reduction.
- Froze a separate stress cohort of 2,000 real Common Crawl long/deep/query legitimate URLs across 83 domains plus 200 phishing URLs, with exact URL, hostname, and encoded-input exclusions retained.
- On the long-URL stress set, +10% improved recall from 77% to 80% but increased FPR from 1.05% to 1.45%. The increase was not conventionally significant (legitimate paired p=0.0768), but it failed the predeclared directional non-regression rule.
- Applied the unchanged final-choice rule: retain weighted HYB-MS real-only as the conservative unconditional reference; keep weighted HYB-MS +10% MS-HardGAN-C as the best ordinary low-prevalence candidate. Neither is declared deployment-ready.
- Diagnosed the browser prototype's disruptive false positives: v0.1 used the earlier browser-calibrated +10% bundle and opened an interstitial for both warning and block tiers even though URL text had no independent confirmation signal.
- Implemented browser extension v0.2 without changing model scores or thresholds: only the strict block tier interrupts, lower risk tiers remain passive, wording no longer presents scores as confirmed phishing, exact-host trust is explicit/reversible/local, and inference errors remain fail-open.
- Added preprocessing, policy, and WebAssembly parity tests. All pass; maximum ONNX Runtime Web fixture score difference is 2.384e-7 with approximately 7.8 ms mean and 18.3 ms p95 URL inference in the Node-hosted WASM test on this machine.
- Post-hoc policy diagnostics reduced false interruptions from 88 to 6 on 28,500 primary legitimate URLs and from 22 to 0 on 2,000 long legitimate URLs. This improves deployment usability but intentionally sends fewer phishing URLs to an interstitial; passive warnings remain.

## Next controlled stage

1. Repeat the unchanged frozen comparison on a larger, independently adjudicated prospective cohort to narrow uncertainty around phishing recall and the paired model difference.
2. Add a separate browser-traffic stress cohort containing long legitimate paths, parameters, redirects, login pages, and regional domains; do not mix this with the completed root-domain prevalence test.
3. Measure browser-extension latency, memory use, alert burden, and failure behavior without retraining on deployment traffic.
4. Keep weighted HYB-MS +10% MS-HardGAN-C as the deployment reference while retaining the real-only condition as the permanent control.

Complete GAN report: `artifacts/results/gan_augmentation_v1/summary/REPORT.md`.

URL-GAN report: `artifacts/results/urlgan_experiment/REPORT.md`.

MS-HardGAN report: `docs/MS_HARDGAN.md`; final audit: `artifacts/results/ms_hardgan_v2/quality/quality.json`; controlled downstream comparison: `artifacts/results/ms_hardgan_augmentation_v2/comparison/REPORT.md`.

Feature-fair hybrid/GAN report: `artifacts/results/fair_hybrid_gan_v1/fair_hybrid_gan_complete_report.pdf`; machine-readable metrics: `artifacts/results/fair_hybrid_gan_v1/summary.json` and `all_metrics.csv`.

Imbalanced 30K confirmation report: `artifacts/results/external/2026-08-13-imbalanced-30k/imbalanced_30k_complete_report.pdf`; complete metrics: `all_models_metrics.csv` and `comparison.json` in the same directory.

Final prospective 2% report: `artifacts/results/prospective_2026-08-22_2pct/prospective_comparison.pdf`; complete metrics and frozen predictions are in the same directory. The frozen model manifest is `artifacts/final_candidate_2026-08-22/manifest.json`.

Final two-cohort decision report: `artifacts/results/final_model_decision_2026-08-22/final_model_decision.pdf`; protocol and complete interpretation: `artifacts/final_confirmation_protocol_2026-08-22/manifest.json` and `docs/FINAL_FROZEN_COMPARISON_2026-08-22.md`.

Browser v0.2 false-positive mitigation: `docs/BROWSER_FALSE_POSITIVE_MITIGATION_V020.md`; diagnostic PDF: `artifacts/results/browser_v0_2_policy_diagnostic/browser_v0_2_policy_diagnostic.pdf`; unpacked extension: `browser_extension/`.

Professor model-comparison report: `artifacts/results/professor_model_comparison/REPORT.md`.

## GPU status

CUDA training completed on an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. The neural trainer explicitly rejects CPU fallback.
