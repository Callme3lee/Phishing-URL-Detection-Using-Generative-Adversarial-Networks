# Augmented pretraining followed by real-only fine-tuning

## Motivation

Quality-filtered LSGAN training improves balanced-test CNN F1 and source-macro
ROC-AUC, but real-only still has the best source-macro F1. This suggests that
synthetic examples improve the learned ranking while shifting the final
classifier boundary. The next experiment therefore uses two stages:

1. pretrain the CNN on real plus quality-filtered LSGAN URLs;
2. fine-tune the resulting checkpoint for two epochs on the identical 90,000
   real URLs, using a lower learning rate of `1e-4`.

The validation set selects the checkpoint and threshold. Test labels are not
used for selection. LSGAN 10%, 20%, and 30% initializations are repeated with
seeds 42, 43, and 44.

## Implementation

`train_classifier.py` now accepts `--initial-checkpoint`. It verifies the model
type and vocabulary size before loading weights and records the initialization
checkpoint in the saved artifact.

The complete matrix can be run with:

```bash
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans.run_augmented_finetuning
```

Outputs are written under:

```text
artifacts/robust_augmentation/finetuned/seed<SEED>/<CONDITION>_then_real/cnn/
```

## Completed results

All nine GPU runs completed with seeds 42, 43, and 44. Every fine-tuned model
improved balanced-test F1 over its matching real-only seed.

| LSGAN pretraining dose | Balanced F1 | Change | Source-macro F1 | Change | Source-macro ROC-AUC | Change |
|---|---:|---:|---:|---:|---:|---:|
| 10% then real | 94.7674% | +0.1062 pp | 95.0729% | +0.0203 pp | 98.1140% | +0.1444 pp |
| **20% then real** | **94.8442%** | **+0.1830 pp** | **95.0915%** | **+0.0389 pp** | 98.1531% | +0.1834 pp |
| 30% then real | 94.8307% | +0.1695 pp | 95.0773% | +0.0247 pp | **98.2076%** | **+0.2380 pp** |

Quality-LSGAN 20% followed by real-only fine-tuning is the best measured F1
condition. Its balanced-test paired F1 improvement has a three-run 95% t
interval of +0.0850 to +0.2810 percentage points. Its source-macro F1 mean is
also higher than real-only, but the paired interval includes zero; this smaller
cross-source F1 gain is not yet statistically reliable. Source-macro ROC-AUC
improves at all three doses, with the largest gain at 30%.

Machine-readable results are in
`artifacts/robust_augmentation/finetuned/summary.json` and `.csv`. The comparison
figure is `artifacts/robust_augmentation/finetuned/comparison.png`.

## Conclusion

The two-stage schedule solves part of the earlier trade-off. Synthetic
pretraining adds ranking coverage, while the real-only stage repairs much of the
decision-boundary shift. For the browser-extension candidate, use the 20%-then-
real CNN when selecting by the best measured F1. Keep the real-only CNN as the
conservative reference until the small source-macro F1 gain is repeated with
more seeds or another genuinely unseen collection source.
