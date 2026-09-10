# Immediate Independent External Evaluation Dataset

## Status

The requested evaluation data was collected, frozen, and evaluated once with
the six pre-selected modern CNN checkpoints (three real-only and three
10%-LSGAN-augmented). Stored validation-selected thresholds were used without
recalibration on this dataset.

## Composition

| Class/source | URLs |
|---|---:|
| PhishTank verified-online phishing | 276 |
| PhishStats phishing | 24 |
| Tranco unused legitimate controls | 300 |
| **Total** | **600** |

The malicious URLs were observed between 29 April and 1 August 2026. This is an
independently assembled current-active-feed evaluation set. It must not be
described as 300 URLs all first observed after model training; 24 are from the
strict post-training PhishStats collection, while the remaining verified-online
PhishTank records were unused contemporary records from the current archive.

## Leakage controls

Every URL belongs to a different canonical hostname. All 600 hostnames were
checked against the immutable 2,149,042-host exclusion index covering:

- every URL in the three original datasets;
- the 31 July evaluation;
- all modern malicious training, validation and test URLs;
- all 50,000 modern benign hard negatives; and
- all 4,378 generated LSGAN training URLs.

Verified overlap count: zero.

## Frozen artifacts

- Encoded tokens: `test_tokens.npy`, shape `(600, 128)`.
- Labels: `test_labels.npy`, shape `(600,)`, with 300 examples per class.
- Auditable URL manifest: `frozen_urls.csv`.
- Provenance and hashes: `metadata.json`.

All files are under
`artifacts/immediate_external_evaluation_2026-08-01/frozen_300/`.

The frozen URL-manifest SHA-256 is
`e35074a92a01befb1ab96b19ca57ff460829ac99e211dcfe6d7380e9e922b5df`.

## Evaluation rule

Do not add these URLs to training, validation, hard-negative mining or threshold
selection. When evaluation is authorized, use the six already-frozen modern
real-only/10%-LSGAN checkpoints and their stored thresholds. Evaluate all six
once, aggregate the three paired seeds, and preserve the raw predictions.

## Frozen-checkpoint results

Mean over paired seeds 42, 43 and 44:

| Metric | Real-only | +10% modern LSGAN | Change |
|---|---:|---:|---:|
| Accuracy | 72.67% | **75.78%** | **+3.11 pp** |
| Precision | 99.09% | **99.18%** | **+0.10 pp** |
| Recall | 45.78% | **52.00%** | **+6.22 pp** |
| F1-score | 62.28% | **68.05%** | **+5.78 pp** |
| ROC-AUC | 96.43% | **97.29%** | **+0.86 pp** |
| False-positive rate | 0.44% | 0.44% | 0.00 pp |

Augmentation improved accuracy, recall, F1 and ROC-AUC in every paired seed;
precision was equal or slightly higher, and the false-positive rate was exactly
unchanged within each pair. The three-seed paired 95% Student-t intervals
exclude zero for accuracy (+0.29 to +5.93 pp), recall (+0.59 to +11.86 pp), and
ROC-AUC (+0.26 to +1.46 pp). The F1 mean gain is consistent across all three
seeds, but its three-seed interval (-0.51 to +12.06 pp) narrowly includes zero,
so it should not be called statistically conclusive from only three training
seeds.

These results support the project claim that selected quality-filtered modern
LSGAN augmentation improved generalization on this independently assembled
current-feed test. They do not establish that every GAN, augmentation ratio, or
future feed will improve performance. The relatively low frozen-threshold
recall in both conditions also shows that deployment work should prioritize
recall/threshold trade-offs while maintaining the false-positive constraint.

## Result artifacts

- Consolidated JSON: `../artifacts/immediate_external_evaluation_2026-08-01/summary/summary.json`
- Consolidated CSV: `../artifacts/immediate_external_evaluation_2026-08-01/summary/summary.csv`
- Presentation graph: `../artifacts/immediate_external_evaluation_2026-08-01/summary/comparison.png`
- Per-seed metrics and raw probability/prediction arrays:
  `../artifacts/immediate_external_evaluation_2026-08-01/results/`

The evaluator requires CUDA and uses mixed-precision GPU inference. The raw
prediction arrays preserve the labels, probabilities, binary decisions, and
the checkpoint's frozen threshold for each run.
