# Version 3 Pre-GAN Evaluation

## Decision

The project is **engineering-ready to implement a Version 3 GAN**, but it is **not yet scientifically ready to claim GAN improvement**. The classifier/data pipeline is reproducible; the remaining blocker is a genuinely later untouched confirmation snapshot for the calibrated real-data reference.

## Data foundation

URLs_New contributed 450,176 raw rows. After validation, conflict handling, canonicalization, and encoded deduplication, the frozen splits contain 358,619 training, 39,483 validation, and 49,229 test rows. Hostnames are disjoint and all 28 verification checks pass.

The GAN contract is already fixed: train generators only on phishing rows from the frozen Version 3 training split.

## Internal real-only baselines

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Lstm | 0.9898 | 0.9817 | 0.9696 | 0.9756 | 0.9968 | 0.0048 |
| Cnn | 0.9889 | 0.9746 | 0.9725 | 0.9736 | 0.9970 | 0.0067 |
| Rnn | 0.9866 | 0.9730 | 0.9629 | 0.9679 | 0.9939 | 0.0071 |
| Random Forest | 0.9496 | 0.9718 | 0.7822 | 0.8667 | 0.9627 | 0.0060 |
| Linear Svm | 0.8975 | 0.9687 | 0.5280 | 0.6835 | 0.9298 | 0.0045 |

The LSTM is the strongest original internal baseline; the CNN has marginally higher recall and ROC-AUC. These internal values do not establish cross-source generalization.

## External progression

### Top-Domain / PhishTank (500)

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original LSTM | 0.4860 | 0.4928 | 0.9587 | 0.6510 | 0.8431 | 0.9867 |
| Coverage LSTM | 0.5480 | 0.5265 | 0.9547 | 0.6787 | 0.9087 | 0.8587 |
| Balanced LSTM | 0.4893 | 0.4946 | 0.9613 | 0.6531 | 0.8503 | 0.9827 |
| Hybrid original threshold | 0.8727 | 0.8143 | 0.9667 | 0.8838 | 0.9704 | 0.2213 |
| Calibrated ensemble | 0.9260 | 0.9908 | 0.8600 | 0.9208 | 0.9746 | 0.0080 |

### Tranco / OpenPhish (400)

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Original LSTM | 0.4317 | 0.4620 | 0.8517 | 0.5988 | 0.6857 | 0.9883 |
| Coverage LSTM | 0.5075 | 0.5044 | 0.8983 | 0.6460 | 0.8369 | 0.8833 |
| Balanced LSTM | 0.4550 | 0.4763 | 0.9033 | 0.6237 | 0.5952 | 0.9933 |
| Hybrid original threshold | 0.8300 | 0.7848 | 0.9133 | 0.8436 | 0.9276 | 0.2533 |
| Calibrated ensemble | 0.8875 | 0.9755 | 0.7950 | 0.8760 | 0.9324 | 0.0200 |

## Readiness gates

| Gate | Status | Evidence |
|---|---|---|
| Dataset Integrity | PASS | 28/28 verification checks pass |
| Baseline Matrix | PASS | 15/15 complete model-seed runs |
| Controlled Shift Experiments | PASS | 30/30 metrics artifacts |
| Hybrid Reproducibility | PASS | 3/3 checkpoints; frozen hashes valid=True |
| Calibration Separation | PASS | zero measured URL/hostname/encoding overlap |
| Source Shortcut Risk | CAUTION | original strongest one-feature stump ROC-AUC=0.7696; hybrid mitigates but does not prove invariance |
| External Claim Validity | CAUTION | current hybrid/calibrated external results informed model design and are exploratory |
| Later Temporal Confirmation | BLOCKED | confirmation readiness=False |
| Version 3 Gan Absence | PASS | no Version 3 generator/trainer or augmentation result exists yet |

## Before GAN training

1. Keep the real-only baselines, hybrid checkpoints, ensemble rule, threshold, and all splits frozen.
2. Obtain the genuinely later confirmation snapshot and evaluate the calibrated ensemble once.
3. Pre-register GAN quality gates before viewing augmentation outcomes: validity, uniqueness, novelty, training overlap, lexical realism, source diversity, and real-only classifier challenge rate.
4. Train generators only on the 82,219 phishing rows in the frozen original training split unless a separately named multi-source generator condition is explicitly registered.
5. Keep 10%, 20%, 30%, and 40% augmentation counts fixed and compare every condition against the same real-only reference across seeds 42/43/44.
6. Treat no improvement or degradation as valid evidence; do not change ratios, samples, or thresholds to force a win.
