# Later-Collected External Evaluation (31 July 2026)

## Conclusion

The current CNN models do **not** generalize safely to this later-collected,
exact-hostname-unseen snapshot. The result supersedes any claim that the earlier
held-out experiments alone establish deployment readiness. On this external
set, the real-only CNN ranks examples better than the augmentation-pretrained
CNN, but neither frozen classifier has an acceptable operating point.

## Frozen test protocol

- Collection time: `2026-07-31T20:31:13Z`.
- Phishing source: the OpenPhish community feed snapshot (300 rows).
- Benign proxy: the daily Tranco top-one-million snapshot.
- Leakage screen: canonical exact hostname comparison against **every row** in
  `urls_dataset2M.csv`, `URL dataset.csv`, and `new_data_urls.csv`—not only the
  training folds. Canonicalization lowercases and IDNA-normalizes the hostname,
  removes a trailing dot, and removes a leading `www.`.
- After screening and within-feed hostname deduplication: 255 phishing URLs and
  255 matched benign domain-root URLs (510 total).
- The data was never used for training, validation, model selection, threshold
  calibration, or threshold adjustment.
- Evaluated on GPU with each checkpoint's already-frozen threshold over seeds
  42, 43, and 44.

Source snapshots and hashes are recorded in
`artifacts/later_collected_2026-07-31/evaluation/metadata.json`.

## Primary results

Mean over the three seeds:

| Frozen model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Real-only CNN | 45.23% | 47.47% | 89.41% | 62.01% | 64.78% |
| 20% LSGAN pretrain + real fine-tune | 44.71% | 47.19% | 89.02% | 61.68% | 55.88% |
| Augmented minus real-only | -0.52 pp | -0.27 pp | -0.39 pp | -0.33 pp | -8.90 pp |

The augmented-minus-real paired 95% seed confidence interval for ROC-AUC is
`-13.41` to `-4.39` percentage points. Accuracy, precision, recall, and F1
differences have intervals crossing zero. The apparently moderate recall and F1
are misleading: the models label nearly everything as phishing. Across seeds,
the real-only model recognizes only 0–5 of the 255 benign examples; the augmented
model recognizes only 0–2. False-positive rates are therefore 98.0–100% and
98.8–100%, respectively.

The figure is
`artifacts/later_collected_2026-07-31/summary/comparison.png`, with exact values
in `summary.json` and `summary.csv` beside it.

## Host-only sensitivity check

For diagnosis only, both classes were reduced to their canonical hostname and
reevaluated without changing thresholds. Every checkpoint predicted all 510
examples as phishing: 50% accuracy, 50% precision, 100% recall, 66.67% F1, and
100% false-positive rate. Mean ROC-AUC was 41.71% for real-only and 36.36% for
augmented pretraining. This shows that the primary failure is not explained only
by the `https://` prefix. It is consistent with strong source/domain shift and
over-reliance on familiar legitimate hosts.

## Interpretation and limitations

This is a strict and useful stress test, but it is small. “Later-collected” means
the snapshot was obtained after model development; the public files do not prove
that every domain itself was first registered or first observed after training.
OpenPhish intelligence may contain label noise. Tranco membership is a benign
proxy, not a safety guarantee, and its entries were represented as canonical
`https://domain/` root URLs. These limitations do not rescue the deployment
result: a browser extension cannot tolerate an approximately 99% false-positive
rate on unfamiliar popular domains.

## Required follow-up

Do not recalibrate on this frozen snapshot or add it to training; preserve it as
an honest regression test. Build a separate development dataset with recent
benign full URLs and recent phishing URLs, normalize URL representation
consistently, train with source-balanced batches and hard benign negatives, and
select the threshold under an explicit low-false-positive constraint. Only then
evaluate once more on a newly collected future snapshot.
