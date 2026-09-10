# Why the 40% WGAN-GP Results Differ

## Executive conclusion

The two projects do not produce contradictory results because they do not test
the same data distribution or the same scientific question. The newer project
reports an **in-domain** 52,946-URL test split assembled from the same two
sources used for training. Its real-only CNN already reaches 99.4188% mean F1.
WGAN-GP 40% raises this to 99.4375%, a gain of only **0.0187 percentage points**.
The paired 95% confidence interval is -0.0089 to +0.0464 points, so the observed
gain is not statistically supported.

The selected result in this project is based on a much harder modern/current-feed
protocol. Its final 600-URL evaluation uses hostnames excluded from 2,149,042
protected training, generated, and earlier evaluation hostnames. Under that
distribution shift, real-only CNN mean F1 is 62.28% and 10% quality-filtered
modern LSGAN mean F1 is 68.05%. The lower absolute score reflects test
difficulty; it does not mean that the classifier architecture is intrinsically
worse.

Therefore, the correct interpretation of the newer 40% experiment is:

> CNN + WGAN-GP 40% is the highest observed in-domain condition, but it is
> statistically indistinguishable from the real-only CNN and has not yet been
> externally evaluated.

## Direct comparison

| Item | New 40% experiment | This project's selected experiment | Why it matters |
|---|---|---|---|
| Development data | 543,384 retained URLs from two new-project sources | 39,402 balanced modern URLs for final adaptation | Training scale and source composition differ substantially. |
| Test data | 52,946 in-domain URLs from the same two development sources | 600 independently assembled current-feed URLs | In-domain tests are normally much easier than source/time-shifted tests. |
| Test class balance | 31,712 legitimate; 21,234 malicious | 300 legitimate; 300 phishing | Accuracy and pooled metrics are affected by class/source composition. |
| Classifier | CNN: 48-d embedding, 128 channels for kernels 3/4/5 | CNN: 32-d embedding, 64 channels for kernels 3/5/7 | The new CNN is larger and was trained for up to eight epochs. |
| Real training rows | 432,662 | 39,402 | The new model has roughly eleven times more real rows than the final modern experiment. |
| Generated rows | 288,441 WGAN-GP URLs | 4,378 quality-filtered LSGAN URLs | “40%” and “10%” also represent very different absolute quantities. |
| Final training size | 721,103 | 43,780 | The experiments are not dose-only variants of one another. |
| GAN choice | WGAN-GP selected by mean validation F1 among 12 augmented conditions | Modern LSGAN selected by strict quality and downstream temporal performance | GAN utility changes with source, filter, and target distribution. |
| Threshold | Validation F1 with CNN FPR <= 1% | Same high-level policy | This part is comparable. |
| External evaluation | Not yet loaded or used | Completed once with frozen checkpoints and thresholds | The current evidence levels are different. |

## The largest data-distribution difference

The new project uses `URL dataset.csv` as a development source. Its SHA-256 is
`18067bad500580a7def376c5a4438c9886adda5eb739fc209fdda4c6309966a7`,
which exactly matches this project's `Data sets/URL dataset.csv`. In this
project, that file was originally used to expose cross-source failure; in the
new project it contributes training, validation, and test rows. Once a source
is included in training, a held-out-hostname split from the same source is no
longer a completely unseen-source evaluation.

The new frozen test also contains a strong source/label association:

| Test source | Legitimate | Malicious | Total |
|---|---:|---:|---:|
| Main (`URL dataset.csv`) | 31,671 | 10,451 | 42,122 |
| Additional real (`URLs for ML.csv`) | 41 | 10,783 | 10,824 |

The additional source is 99.62% malicious in the test set. Hostname grouping
prevents direct hostname leakage, but it cannot remove source-specific
formatting, collection, and label cues. This makes pooled metrics easier to
inflate through between-source separation.

A diagnostic recomputation from the saved per-URL CNN predictions illustrates
the issue. The real-only CNN's reported pooled ROC-AUC is 99.9255%, but its
mean source-wise ROC-AUC is approximately 94.23%. The additional source's
three-seed mean ROC-AUC is only about 88.52%, compared with about 99.94% on the
main source. Because the additional source has only 41 legitimate examples,
its source-specific ROC-AUC is also uncertain. The pooled ROC-AUC is valid for
that exact mixture, but it should not be presented as cross-source robustness.

## What “40% works” means numerically

| Metric | Real-only CNN | WGAN-GP 40% CNN | Change |
|---|---:|---:|---:|
| Accuracy | 99.5341% | 99.5492% | +0.0151 pp |
| Precision | 99.4860% | 99.5298% | +0.0437 pp |
| Recall | **99.3517%** | 99.3454% | -0.0063 pp |
| F1 | 99.4188% | **99.4375%** | +0.0187 pp |
| ROC-AUC | **99.9255%** | 99.9233% | -0.0022 pp |
| PR-AUC | 99.8986% | **99.9081%** | +0.0095 pp |
| False-positive rate | 0.3437% | **0.3143%** | -0.0294 pp |

On 52,946 test URLs, the accuracy difference corresponds to approximately eight
fewer errors per run on average. Recall and ROC-AUC are slightly lower with
40% augmentation. Every reported paired CNN interval for this comparison
includes zero. The model therefore “works” in the sense that adding the large
synthetic corpus did not damage the already excellent in-domain CNN and it
produced a marginal observed F1/FPR improvement. It does **not** yet show that
40% is reliably better than real-only.

There is also a multiple-comparison consideration. The reported CNN condition
was selected by the highest mean validation F1 from 12 augmented conditions.
Validation-based selection is correct, but when many nearly tied conditions
are searched, the winning difference may still be ordinary seed variation.
The untouched external test is the appropriate next discriminator.

## Relationship to this project's earlier 40% result

The projects are qualitatively more similar than the absolute scores suggest.
In this project's original controlled percentage experiment:

- LSGAN 40% improved CNN F1 by 0.1285 percentage points in one seed;
- WGAN-GP 40% reduced CNN F1 by 0.1545 points; and
- the report explicitly treated these sub-quarter-point changes as exploratory.

The later source-balanced experiment found a three-seed LSGAN 30% CNN gain of
0.1493 points. The modern temporal experiment then used only 10% generated data
but obtained a much larger +3.15-point F1 gain because the synthetic data was
targeted to a real temporal distribution gap. On the independent 600-URL
current-feed set, the mean F1 gain was +5.78 points. This shows that augmentation
percentage alone does not determine success. Target relevance, real-source
coverage, filtering, and evaluation distribution matter more.

## Model and training differences

The CNN families are related but not identical:

- New experiment: vocabulary 191, embedding dimension 48, 128 channels per
  kernel, kernels 3/4/5, dropout 0.5, up to eight epochs.
- This project's original CNN: vocabulary 115, embedding dimension 32, 64
  channels per kernel, kernels 3/5/7, dropout 0.3, normally five epochs.
- The new model checkpoint is about 333 KB and has substantially more
  convolution capacity.
- The new 40% training loss dynamically down-weights the now-majority positive
  class using `BCEWithLogitsLoss(pos_weight=negative/positive)`. This prevents
  the 61.3%-malicious augmented training set from merely forcing all-positive
  predictions.

These changes help explain the stronger real-only baseline, but the dominant
explanation remains the different data and in-domain evaluation.

## Required next validation

Before treating WGAN-GP 40% as the final model:

1. Freeze the three real-only and three WGAN-GP 40% CNN checkpoints and their
   existing thresholds.
2. Open the declared PhiUSIIL external dataset only after confirming the freeze.
3. Verify zero exact-URL, canonical-hostname, and encoded-row overlap.
4. Report pooled and per-source metrics; include source-macro ROC-AUC so source
   imbalance cannot dominate the conclusion.
5. Evaluate the six frozen checkpoints exactly once without recalibration.
6. Also run the planned quality-filtered WGAN-GP 10% comparison. This tests
   whether 288,441 generated URLs are necessary or whether a smaller, more
   targeted dose performs as well.
7. Select the browser candidate using external recall/F1, ROC-AUC, precision,
   FPR, seed consistency, model size, and latency.

## Final finding

No implementation defect is required to explain the difference. The newer
report measures near-saturated performance on a same-source development split;
this project emphasizes temporal and external generalization. The 40% model is
a reasonable candidate for external testing, but its current evidence does not
supersede the independently evaluated 10% modern LSGAN result.
