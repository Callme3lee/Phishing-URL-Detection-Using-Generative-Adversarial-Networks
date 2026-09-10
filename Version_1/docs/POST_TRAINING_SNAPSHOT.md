# Genuinely Post-Training Snapshot (1 August 2026)

## Status

This is a genuine post-training smoke test, but it is too small for a strong
statistical conclusion. The direction favors modern LSGAN augmentation.

## Snapshot construction

Fresh OpenPhish, PhishTank and Tranco downloads were requested at approximately
09:04 UTC. They were byte-for-byte identical to the files already used by the
modern experiment, so they were correctly rejected as new evidence.

An independent PhishStats API query then returned records timestamped after the
previous training source's 07:22:49 UTC cutoff. Five API pages (500 records)
were frozen with hashes. After enforcing all requirements:

- timestamp strictly after the training cutoff;
- one URL per canonical hostname;
- exclusion of every hostname in the original three datasets;
- exclusion of the 31 July frozen snapshot;
- exclusion of all modern train, validation and temporal-test URLs;
- exclusion of all 50,000 modern benign hard negatives; and
- exclusion of all 4,378 accepted generated training URLs;

only six genuinely post-cutoff, unseen-host phishing URLs remained. They were
paired with six completely unused Tranco hostname controls. The final snapshot
therefore contains 12 examples. It was never used for training, validation,
threshold calibration or checkpoint selection.

Exact provenance, counts and hashes are in
`artifacts/post_training_snapshot_2026-08-01_0904Z/evaluation/metadata.json`.

## Frozen-threshold result

Mean over seeds 42, 43 and 44:

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Modern real-only | 75.00% | 100.00% | 50.00% | 66.67% | 99.07% | 0% |
| Modern + 10% LSGAN | **80.56%** | **100.00%** | **61.11%** | **75.56%** | 99.07% | 0% |
| Augmented minus real-only | +5.56 pp | 0.00 pp | +11.11 pp | +8.89 pp | 0.00 pp | 0.00 pp |

Seed 42 ties. In seeds 43 and 44, augmentation detects four of six phishing
URLs while real-only detects three. Every seed and condition correctly rejects
all six benign controls. ROC-AUC is nearly perfect, indicating good ranking,
but the deliberately conservative frozen threshold sacrifices recall.

Because there are only six positives and six negatives, each example changes
class recall or specificity by 16.67 percentage points. Paired seed confidence
intervals for the accuracy, recall and F1 differences cross zero. This result
must not be described as statistically significant or as a final external
validation.

The graph and exact values are under
`artifacts/post_training_snapshot_2026-08-01_0904Z/summary/`.

## Next evidence requirement

Continue freezing independent post-training snapshots without retraining until
at least several hundred unseen-host phishing URLs and matching benign controls
have accumulated. Evaluate the already-frozen checkpoints once on the combined
set. Do not repeatedly tune thresholds or models against the accumulating
snapshot.
