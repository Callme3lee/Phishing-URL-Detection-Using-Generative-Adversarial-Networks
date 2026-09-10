# URLs_New-Only Structure-Balancing Experiment

## Question

Can the external distribution shift be repaired using only the original frozen `URLs_New.csv`, without adding any outside training rows?

## Controlled training condition

The original training split contained 276,400 legitimate and 82,219 phishing rows. Legitimate deep paths dominated:

| Legitimate structure | Original rows | Original share | Balanced rows | Balanced share |
|---|---:|---:|---:|---:|
| Root | 37,493 | 13.57% | 29,882 | 25.00% |
| Shallow | 47,288 | 17.11% | 29,882 | 25.00% |
| Deep | 161,737 | 58.52% | 29,882 | 25.00% |
| Query | 29,882 | 10.81% | 29,882 | 25.00% |

The experiment deterministically retained the 29,882 lowest BLAKE2b hashes per legitimate structure using selection seed 42. All 82,219 phishing rows were retained. The final training set contains 201,747 unique rows: 119,528 legitimate and 82,219 phishing.

Every row comes from the original `URLs_New.csv` Version 3 split. No outside URL was added, no row was repeated, and no external test row was used. Validation/test hashes, architectures, hyperparameters, seeds 42/43/44, and the threshold policy are unchanged.

This intervention changes both structural proportions and the class ratio. It tests deterministic downsampling, not weighting or oversampling.

## Unchanged internal test

Mean ± sample standard deviation across seeds 42/43/44:

| Model | Condition | Accuracy | Recall | F1 | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|
| CNN | Original | 0.9889 ± 0.0013 | 0.9725 ± 0.0023 | 0.9736 ± 0.0031 | 0.9970 ± 0.0008 | 0.0067 ± 0.0014 |
| CNN | Balanced | 0.9882 ± 0.0002 | 0.9679 ± 0.0039 | 0.9716 ± 0.0004 | 0.9965 ± 0.0001 | 0.0065 ± 0.0013 |
| LSTM | Original | 0.9898 ± 0.0003 | 0.9696 ± 0.0012 | 0.9756 ± 0.0006 | 0.9968 ± 0.0004 | 0.0048 ± 0.0006 |
| LSTM | Balanced | 0.9886 ± 0.0003 | 0.9632 ± 0.0022 | 0.9726 ± 0.0006 | 0.9961 ± 0.0005 | 0.0046 ± 0.0008 |
| RNN | Original | 0.9866 ± 0.0017 | 0.9629 ± 0.0042 | 0.9679 ± 0.0041 | 0.9939 ± 0.0008 | 0.0071 ± 0.0012 |
| RNN | Balanced | 0.9814 ± 0.0050 | 0.9365 ± 0.0298 | 0.9545 ± 0.0130 | 0.9869 ± 0.0085 | 0.0067 ± 0.0023 |
| Random Forest | Original | 0.9496 ± 0.0001 | 0.7822 ± 0.0006 | 0.8667 ± 0.0002 | 0.9627 ± 0.0003 | 0.0060 ± 0.0003 |
| Random Forest | Balanced | 0.9427 ± 0.0001 | 0.7548 ± 0.0006 | 0.8466 ± 0.0003 | 0.9643 ± 0.0000 | 0.0075 ± 0.0002 |
| Linear SVM | Original | 0.8975 ± 0.0000 | 0.5280 ± 0.0000 | 0.6835 ± 0.0000 | 0.9298 ± 0.0000 | 0.0045 ± 0.0000 |
| Linear SVM | Balanced | 0.8691 ± 0.0000 | 0.3947 ± 0.0000 | 0.5583 ± 0.0000 | 0.9232 ± 0.0000 | 0.0051 ± 0.0000 |

Aggressive downsampling slightly reduces CNN/LSTM internal recall and F1, materially hurts RF/SVM recall, and increases vanilla RNN seed sensitivity. It does not produce an internal advantage over the original dataset.

## External results

### OpenPhish/Tranco set

| Model | Condition | Accuracy | Recall | F1 | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|
| CNN | Original | 0.4458 | 0.8850 | 0.6149 | 0.5737 | 0.9933 |
| CNN | Balanced | 0.4525 | 0.8883 | 0.6187 | 0.5728 | 0.9833 |
| LSTM | Original | 0.4317 | 0.8517 | 0.5988 | 0.6857 | 0.9883 |
| LSTM | Balanced | 0.4550 | 0.9033 | 0.6237 | 0.5952 | 0.9933 |
| RNN | Original | 0.4467 | 0.8917 | 0.6171 | 0.7534 | 0.9983 |
| RNN | Balanced | 0.4508 | 0.8983 | 0.6206 | 0.7179 | 0.9967 |
| Random Forest | Original | 0.0975 | 0.1700 | 0.1585 | 0.0715 | 0.9750 |
| Random Forest | Balanced | 0.1342 | 0.0900 | 0.0942 | 0.0841 | 0.8217 |
| Linear SVM | Original | 0.3325 | 0.0950 | 0.1246 | 0.1286 | 0.4300 |
| Linear SVM | Balanced | 0.5200 | 0.0500 | 0.0943 | 0.1361 | 0.0100 |

Balanced CNN reduces FPR by only 1.0 percentage point. Balanced LSTM gains 5.17 points of phishing recall but FPR worsens by 0.5 points and ROC-AUC falls by 9.05 points. RF and SVM reduce FPR by predicting far fewer URLs as phishing; their phishing recall collapses to 9% and 5%, respectively. Low FPR alone is therefore misleading.

### Earlier PhishTank/Top-Domain set

Balanced CNN improves ROC-AUC from 0.6580 to 0.6991 and reduces FPR from 0.9867 to 0.9707. Balanced LSTM changes little: ROC-AUC 0.8431 to 0.8503 and FPR 0.9867 to 0.9827. Neural FPR remains approximately 97–100%, so the browser-extension failure remains.

## Direct comparison with outside legitimate coverage

The earlier intervention added only 3,012 audited outside legitimate rows. On OpenPhish/Tranco, LSTM results were:

| LSTM condition | Accuracy | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| Original `URLs_New.csv` | 0.4317 | 0.8517 | 0.5988 | 0.6857 | 0.9883 |
| `URLs_New.csv` + outside coverage | **0.5075** | 0.8983 | **0.6460** | **0.8369** | **0.8833** |
| `URLs_New.csv` structure-balanced | 0.4550 | **0.9033** | 0.6237 | 0.5952 | 0.9933 |

Although outside coverage formed only 0.83% of training, it improved external LSTM ranking and FPR much more than removing 156,872 internally overrepresented rows. This is evidence that source/domain coverage matters more than structure proportions alone.

## Quantified distribution shift

Structural total-variation distance (TVD) ranges from 0 to 1; lower is closer. A lexical source-classifier ROC-AUC of 0.5 means sources are indistinguishable; 1.0 means perfectly separable.

| External set | Class | Structure TVD original | Structure TVD balanced | Source AUC original | Source AUC balanced |
|---|---|---:|---:|---:|---:|
| Top-Domain/PhishTank | Legitimate | 0.8644 | 0.7500 | 0.9944 | 0.9907 |
| Top-Domain/PhishTank | Phishing | 0.6244 | 0.6244 | 0.9067 | 0.9067 |
| Tranco/OpenPhish | Legitimate | 0.8644 | 0.7500 | 0.9972 | 0.9952 |
| Tranco/OpenPhish | Phishing | 0.7344 | 0.7344 | 0.9342 | 0.9342 |

Balancing meaningfully reduces legitimate structural TVD, but external legitimate URLs remain almost perfectly distinguishable from balanced training URLs using only 27 lexical features. This means hostname vocabulary, TLDs, collection source, length relationships, and other lexical patterns remain shifted.

Phishing shift is unchanged by design because all original phishing rows were retained. OpenPhish remains strongly separable from `URLs_New.csv` phishing, confirming that the phishing side also requires multiple real sources.

## Conclusion

`URLs_New.csv` can be structurally rebalanced, but **that alone does not repair external distribution shift**. The experiment rejects the idea that deep-path overrepresentation is the sole cause.

The strongest current evidence supports this path:

1. retain `URLs_New.csv` as the base dataset;
2. add independently sourced, audited legitimate roots and realistic paths with materially greater—not merely 0.83%—representation;
3. add multiple real phishing sources such as PhishTank and OpenPhish to training-only source groups;
4. preserve hostname/source-disjoint validation and future external tests;
5. control mixture proportions in advance rather than tuning them against external results.

GAN augmentation remains orthogonal: synthetic phishing cannot supply missing legitimate source coverage.

## Artifacts

- Freeze: `data/experiments/urls_new_structure_balanced_v1/experiment_report.json`
- Checkpoints: `artifacts/experiments/urls_new_structure_balanced_v1/`
- Internal comparison: `artifacts/results/urls_new_structure_balanced_v1/internal/`
- New external comparison: `artifacts/results/urls_new_structure_balanced_v1/external_openphish_tranco/comparison/`
- Earlier external comparison: `artifacts/results/urls_new_structure_balanced_v1/external_topdomains_phishtank/comparison/`
- Three-condition graphs: `artifacts/results/urls_new_structure_balanced_v1/external_*/three_conditions/`
- Shift report: `artifacts/results/urls_new_structure_balanced_v1/distribution_shift/`
