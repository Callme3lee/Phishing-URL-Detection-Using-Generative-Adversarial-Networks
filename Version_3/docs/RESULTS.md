# Version 3 Results

## Real-only baselines

Every model uses the same 358,619 real training rows, the same frozen validation/test sets, and validation-selected thresholds constrained to at most 1% validation FPR. Results are mean ± sample standard deviation across seeds 42/43/44.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | **0.9898 ± 0.0003** | **0.9817 ± 0.0024** | 0.9696 ± 0.0012 | **0.9756 ± 0.0006** | 0.9968 ± 0.0004 | **0.0048 ± 0.0006** | 0.0304 ± 0.0012 |
| CNN | 0.9889 ± 0.0013 | 0.9746 ± 0.0050 | **0.9725 ± 0.0023** | 0.9736 ± 0.0031 | **0.9970 ± 0.0008** | 0.0067 ± 0.0014 | **0.0275 ± 0.0023** |
| Vanilla RNN | 0.9866 ± 0.0017 | 0.9730 ± 0.0046 | 0.9629 ± 0.0042 | 0.9679 ± 0.0041 | 0.9939 ± 0.0008 | 0.0071 ± 0.0012 | 0.0371 ± 0.0042 |
| Random Forest | 0.9496 ± 0.0001 | 0.9718 ± 0.0012 | 0.7822 ± 0.0006 | 0.8667 ± 0.0002 | 0.9627 ± 0.0003 | 0.0060 ± 0.0003 | 0.2178 ± 0.0006 |
| Linear SVM | 0.8975 ± 0.0000 | 0.9687 ± 0.0000 | 0.5280 ± 0.0000 | 0.6835 ± 0.0000 | 0.9298 ± 0.0000 | 0.0045 ± 0.0000 | 0.4720 ± 0.0000 |

The LSTM is the current best overall internal model: it has the highest mean accuracy, precision, and F1 and the lowest neural-model FPR. The CNN has the highest recall and ROC-AUC and the lowest FNR. The new vanilla tanh RNN performs well but is less accurate and more seed-sensitive than the gated LSTM, supporting the value of LSTM memory gates for URL character sequences.

The RNN is nevertheless a useful compact comparison: its checkpoints are approximately 94 KB, compared with 262 KB for LSTM and 336 KB for CNN. RF remains the strongest non-neural baseline. The zero SVM variation is expected from its convex optimization on identical rows; its seed does not change the fitted solution here.

These are internal hostname-held-out results, not proof of external generalization. The completed controlled GAN study is summarized below.

## Controlled GAN augmentation

SeqGAN and RelGAN were trained only on the same 82,219 phishing rows from the frozen URLs_New training split. SeqGAN produced 349,687 strictly accepted unique URLs from a 500,000-sample budget; RelGAN produced only 12,869 and failed the frozen minimum-volume/realism gate. RelGAN was therefore reported as a failed candidate rather than padded, recycled, or filtered less strictly.

SeqGAN was evaluated at fixed 10%, 20%, 30%, and 40% generated/final-training fractions across five classifiers and seeds 42/43/44. Internal effects were small and model-specific: CNN peaked at 10% F1 0.9764 versus 0.9736 real-only, SVM peaked at 20% F1 0.6935 versus 0.6835, while LSTM never exceeded real-only, RF degraded monotonically, and RNN became unstable at higher ratios.

These small internal gains did not transfer consistently. URLs_New+SeqGAN neural FPR remained roughly 95–100% on the external sets. After real legitimate coverage was added, SeqGAN generally reduced LSTM external F1/ROC-AUC and did not deliver a dependable FPR benefit. Therefore, GAN uniqueness does not prove distribution-shift improvement, and phishing-only synthetic data cannot replace real multi-source coverage.

See `artifacts/results/gan_augmentation_v1/summary/REPORT.md`, its machine-readable `summary.json`, and the separate all-metric graphs.

A dedicated error-rate audit checks every FPR/FNR result and paired baseline comparison. It finds no Top-Domain/PhishTank case and no coverage-plus-GAN external case where both errors improve together. The few apparent Tranco/OpenPhish exceptions retain approximately 98–99% FPR or lose substantial ROC-AUC, so they are not operational improvements. See `artifacts/results/gan_augmentation_v1/error_rate_analysis/REPORT.md`.

## Frozen external evaluation

A later 500-row overlap-free snapshot containing 250 Top-Domain legitimate root proxies and 250 current PhishTank verified-online URLs exposes severe distribution shift. Neural phishing recall remains 95.9–96.7%, but neural FPR is 98.7–99.9%; RF and SVM ROC-AUC falls below 0.5. These external results supersede the internal metrics for deployment-readiness decisions. See `docs/EXTERNAL_REAL_ONLY_EVALUATION.md`.

Machine-readable summary: `artifacts/results/baseline_summary.json`.

## Legitimate structural-coverage experiment

A controlled training-only intervention added 3,012 audited proxy-labelled legitimate roots, paths, and query URLs once, increasing the training set from 358,619 to 361,631 rows. The original validation/test rows, models, seeds, hyperparameters, and threshold policy were unchanged.

Internal results remained mostly stable. On a second overlap-free 400-row OpenPhish/Tranco source-transfer set, LSTM improved most: mean ROC-AUC increased from 0.6857 to 0.8369 and FPR decreased from 0.9883 to 0.8833 while recall increased from 0.8517 to 0.8983. This is useful evidence that legitimate structural coverage matters, but the remaining 88.3% FPR is not deployment-ready.

See `docs/LEGITIMATE_COVERAGE_EXPERIMENT.md` for the complete protocol, all models, the earlier external diagnostic, limitations, graphs, and reproducible artifacts.

## URLs_New-only structure balancing

A third controlled condition used only original `URLs_New.csv` training rows: exactly 29,882 legitimate roots, shallow paths, deep paths, and query URLs each, plus all 82,219 phishing rows. The final training set contains 201,747 unique rows; no outside rows or duplicates were added.

Structural balancing reduced legitimate structure distance to both external sets from 0.8644 to 0.7500, but lexical source separability remained 0.9907–0.9952 ROC-AUC. Neural external FPR remained approximately 97–100%. On OpenPhish/Tranco, balanced LSTM reached 0.9033 recall but 0.9933 FPR and only 0.5952 ROC-AUC.

Therefore, structure imbalance inside `URLs_New.csv` is real but is not the primary remaining cause. The 3,012 outside-coverage rows improved LSTM external FPR and ROC-AUC much more, indicating that independent source/domain coverage is required. See `docs/URLS_NEW_STRUCTURE_BALANCING.md`.

## Hybrid CNN + PSL-aware lexical + CORAL

The next fixed experiment combined a character CNN, 35 normalized PSL-aware lexical features, source/class-balanced sampling, and class-conditional CORAL alignment. It used 363,692 stored real rows from URLs_New plus strictly overlap-excluded real coverage sources; no GAN URLs were used. The original validation/test sets remained unchanged.

At the comparable original-validation threshold, exploratory external FPR fell from the original LSTM's 98.67% to 22.13% on Top-Domain/PhishTank and from 98.83% to 25.33% on Tranco/OpenPhish. Recall was 96.67% and 91.33%; ROC-AUC was 97.04% and 92.76%. A stricter domain-robust threshold reduced FPR to 0.13% and 0%, but recall fell to 50.67% and 44.50%.

This is the strongest distribution-shift result so far: representation quality improved substantially, but threshold calibration is not solved. Since both external sets informed model design, a later untouched snapshot is required before making a confirmatory claim. See `docs/HYBRID_CNN_CORAL_EXPERIMENT.md`.

## Frozen calibrated ensemble

A separate 400-row, post-model calibration set was frozen with 200 Tranco legitimate root proxies and 200 PhishTank verified-online URLs. Every URL, hostname, and encoded input is disjoint from training, development, and previous external data. It was used only to choose the decision threshold for the mean score of seeds 42/43/44.

The fixed policy selected threshold 0.98795575 by maximizing calibration F1 subject to at most 1% calibration FPR. Applied unchanged to the older external sets, exploratory FPR is 0.8% on Top-Domain/PhishTank and 2.0% on Tranco/OpenPhish. Corresponding F1 is 92.08% and 87.60%, while recall is 86.0% and 79.5%.

This is a much more practical operating point, but the old external sets are not confirmatory because they informed hybrid design. A feed-freshness guard reports that genuinely later confirmation data are not yet available. See `docs/CALIBRATED_HYBRID_ENSEMBLE.md`.

## Unchanged baselines on the multi-source rows

CNN, LSTM, RNN, RF, and linear SVM were also retrained for seeds 42/43/44 on the exact 363,692 stored real rows available to HYB-MS. Their architectures, hyperparameters, original validation/test sets, and validation threshold policy were unchanged; no GAN-generated rows were used.

On the frozen 10,000-row external snapshot, LSTM was the strongest of these ordinary baselines (F1 0.6660, ROC-AUC 0.9309), but its FPR remained 0.9315. RF obtained the largest improvement over its original-only version (F1 +0.1034, ROC-AUC +0.3816, FPR -0.2137) but still had FPR 0.7091 and FNR 0.3241. CNN and RNN gained ranking quality but continued to flag almost every external legitimate root as phishing.

Therefore, merely appending the 5,073 outside rows is insufficient. HYB-MS additionally uses source-balanced exposure, PSL-aware lexical fusion, conditional CORAL, a three-seed ensemble, and separate outside-domain threshold calibration. See `artifacts/results/multisource_baselines_v1/summary/REPORT.md`.
