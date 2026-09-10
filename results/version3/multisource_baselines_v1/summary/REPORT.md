# Multi-source Baseline Experiment

## Controlled question

Does giving the unchanged CNN, LSTM, RNN, RF, and linear SVM baselines the same stored multi-source real row set used by HYB-MS improve source-transfer performance?

The training set contains **363,692 unique real URLs**: 358,619 original URLs_New training rows plus 3,012 outside legitimate, 2,000 PhishTank, and 61 OpenPhish rows. It contains **zero GAN-generated rows**. The original validation (39,483) and test (49,229) sets, model hyperparameters, seeds 42/43/44, and threshold rule are unchanged.

## Internal original test — multi-source baselines

| Model | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.9890 ± 0.0007 | 0.9756 ± 0.0043 | 0.9719 ± 0.0015 | 0.9737 ± 0.0017 | 0.9970 ± 0.0002 | 0.9936 ± 0.0002 | 0.0065 ± 0.0012 | 0.0281 ± 0.0015 |
| LSTM | 0.9884 ± 0.0009 | 0.9790 ± 0.0050 | 0.9654 ± 0.0018 | 0.9721 ± 0.0021 | 0.9958 ± 0.0008 | 0.9914 ± 0.0010 | 0.0055 ± 0.0013 | 0.0346 ± 0.0018 |
| RNN | 0.9819 ± 0.0031 | 0.9534 ± 0.0105 | 0.9608 ± 0.0047 | 0.9570 ± 0.0073 | 0.9928 ± 0.0015 | 0.9858 ± 0.0026 | 0.0125 ± 0.0029 | 0.0392 ± 0.0047 |
| RF | 0.9488 ± 0.0001 | 0.9717 ± 0.0008 | 0.7785 ± 0.0004 | 0.8644 ± 0.0002 | 0.9640 ± 0.0011 | 0.9344 ± 0.0014 | 0.0060 ± 0.0002 | 0.2215 ± 0.0004 |
| SVM | 0.8988 ± 0.0000 | 0.9684 ± 0.0000 | 0.5346 ± 0.0000 | 0.6889 ± 0.0000 | 0.9301 ± 0.0000 | 0.8722 ± 0.0000 | 0.0046 ± 0.0000 | 0.4654 ± 0.0000 |

## Frozen 10K external snapshot — multi-source baselines

| Model | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4887 ± 0.0006 | 0.4942 ± 0.0003 | 0.9662 ± 0.0009 | 0.6539 ± 0.0001 | 0.7950 ± 0.0161 | 0.8071 ± 0.0143 | 0.9889 ± 0.0021 | 0.0338 ± 0.0009 |
| LSTM | 0.5163 ± 0.0269 | 0.5089 ± 0.0146 | 0.9641 ± 0.0005 | 0.6660 ± 0.0124 | 0.9309 ± 0.0057 | 0.9539 ± 0.0045 | 0.9315 ± 0.0539 | 0.0359 ± 0.0005 |
| RNN | 0.4831 ± 0.0021 | 0.4914 ± 0.0011 | 0.9614 ± 0.0008 | 0.6504 ± 0.0008 | 0.8725 ± 0.0131 | 0.8931 ± 0.0105 | 0.9951 ± 0.0048 | 0.0386 ± 0.0008 |
| RF | 0.4834 ± 0.0112 | 0.4881 ± 0.0079 | 0.6759 ± 0.0018 | 0.5668 ± 0.0053 | 0.6298 ± 0.0020 | 0.7433 ± 0.0014 | 0.7091 ± 0.0224 | 0.3241 ± 0.0018 |
| SVM | 0.3464 ± 0.0000 | 0.2670 ± 0.0000 | 0.1760 ± 0.0000 | 0.2122 ± 0.0000 | 0.2499 ± 0.0000 | 0.4372 ± 0.0000 | 0.4832 ± 0.0000 | 0.8240 ± 0.0000 |

## Main finding

The multi-source **LSTM is the best of the five unchanged baselines externally**: its mean ROC-AUC is 0.9309 and F1 is 0.6660. However, its FPR remains 0.9315, so it still falsely flags most external legitimate roots.

The added real rows improved ranking information without producing reliable threshold transfer. For example, external LSTM ROC-AUC rose by +0.0987, while its FPR changed by -0.0529. CNN and RNN also gained ROC-AUC but retained near-total legitimate false-positive rates.

This means the multi-source **row set alone is insufficient**. HYB-MS also depends on source-balanced exposure, PSL-aware lexical features, conditional CORAL alignment, three-seed ensembling, and a separate frozen calibration set. Therefore HYB-MS must not be described as a fair architecture-only comparison with these ordinary-sampling baselines.

## Deployment conclusion

None of these five unchanged multi-source baselines is deployment-ready on this snapshot. HYB-MS remains the only tested condition with an externally usable balance (FPR 0.0170, FNR 0.1476, F1 0.9120), but that result is an ensemble calibrated on a separate outside-domain calibration set and must be reported as such.

## Artifacts

- `internal_original_vs_multisource.png`: original-only versus multi-source on the unchanged internal test.
- `external_original_vs_multisource.png`: the same controlled comparison on the frozen external 10K snapshot.
- `comparison.csv` and `summary.json`: machine-readable means, SDs, and deltas.
