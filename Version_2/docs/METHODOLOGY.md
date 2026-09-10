# Version 2 Methodology

## Research question

Does adding strictly filtered GAN-generated malicious URLs improve classifiers
trained on the same real URLs, when preprocessing, validation, testing,
threshold selection, and random seeds are controlled?

## Master data contract

Version 2 accepts one real source only: `data/raw/urls_dataset2M.csv`. The
preparation command first verifies its SHA-256. Any mismatch stops execution.

URLs are normalized locally, exact duplicates are removed, conflicting labels
are rejected, and canonical hostnames are calculated. Whole hostnames are
assigned deterministically to train, validation, or test using a seeded BLAKE2b
hash. Therefore no canonical hostname may cross a split.

The character vocabulary is learned from training URLs only. Encoded sequences
are 128 tokens and include PAD, unknown, beginning-of-sequence, and
end-of-sequence symbols. Encoded collisions are audited across splits.

## Shared experimental views

| View | Rows allowed | Purpose |
|---|---|---|
| Real classifier train | All train labels | CNN/LSTM/RF/SVM real-only control |
| GAN train | Train label 1 only | Generator pretraining and adversarial fitting |
| Validation | All validation labels | Checkpoints, thresholds, early stopping |
| Test | All test labels | One evaluation after selection |
| Augmented train | Same real train + accepted generated label 1 rows | Paired augmentation experiment |

The GAN never consumes validation/test malicious URLs. Generated-sample filters
may compare against protected hashes from all real splits to reject leakage,
but may not use validation/test labels or classifier performance to select a
generator.

## Classifier metrics

Every classifier reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
false-positive rate, confusion matrix, training time, inference throughput, and
saved model size. Thresholds are chosen on validation data by maximum F1 subject
to validation FPR no greater than 1%. The threshold is frozen for test data.

## Generator metrics

Both generators receive the same malicious training subset, character
vocabulary, maximum length, pretraining budget, generation budget, and protected
hash sets. Reported metrics are:

- raw and accepted uniqueness;
- URL-like validity and acceptance rate;
- exact and encoded memorization;
- protected-host overlap;
- character, bigram, prefix, and template diversity;
- length-distribution distance;
- train/validation negative log-likelihood and perplexity; and
- downstream classifier change after identical filtering and dose construction.

Uniqueness alone is never treated as proof that mode collapse is fixed.

## Frozen-classifier generated challenge

Generated-URL evasion is evaluated separately from augmentation. Strictly
accepted generated URLs are scored by the saved real-only classifiers with
their original thresholds and without retraining. Since the generated challenge
set has one intended class (phishing), detection rate/recall and false-negative
rate/evasion are reported. Accuracy, precision, FPR, and ROC-AUC require both
classes and are not reported for this test.

Consensus across classifier families and seeds is combined with URL/TLD-like
structure, lexical feature inlier checks, and distance to real malicious
training examples. Evasion alone is not called generator quality because
malformed or out-of-distribution strings can also cross a classifier boundary.

## Quality-controlled augmentation selection

The controlled follow-up uses SeqGAN for every augmented condition and fixes the
10% count at 228,425 rows. Realism filtering uses only strict accepted samples,
TLD structure, real-training lexical statistics, and real-training structural
quantiles. Difficulty uses all real-only classifier families and seeds. Each
checkpoint score is transformed to an empirical percentile before aggregation,
so heterogeneous raw score scales are not mixed.

A 64-cluster lexical diversity cap prevents one artifact or cluster from
dominating. A separate source-pool tail outside the existing random condition is
reserved before selection; its 20,000-row challenge subset is excluded from all
training. The selection rule and artifact hashes are frozen before augmented
test evaluation.

## Statistical comparison

Classifier experiments use seeds 42, 43, and 44. Each augmented run is paired
with the same-seed real-only control. Reports include the mean paired change and
a two-sided 95% Student-t confidence interval. Exploratory searches across GAN
and dose conditions are identified as exploratory; a final external test must
use a predeclared selected condition.

## Augmentation dose definition

The dose is the generated share of the **final augmented training rows**:
`generated / (unchanged real train + generated)`. With 2,055,821 real training
rows, the 10%, 20%, 30%, and 40% conditions require 228,425, 513,955, 881,066,
and 1,370,547 accepted generated rows (subject to exact integer rounding).
Conditions are nested slices of one seeded shuffled pool, so higher doses add
samples without changing or removing any lower-dose or real row.
