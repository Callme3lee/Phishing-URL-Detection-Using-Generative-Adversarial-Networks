# Version 2 Project Report

## Executive summary

Version 2 resets the phishing-URL augmentation study around one controlled real
dataset. Every baseline uses the same frozen training rows. Both GANs learn only
from malicious rows inside that training split, and every augmented condition
adds accepted generated rows without deleting, resampling, or changing the real
rows. Validation and test data are never used to fit a classifier or GAN.

This structure fixes the largest interpretability problem in Version 1:
results from different source mixtures or split definitions can no longer be
mistaken for an augmentation effect.

## The one real dataset

The only permitted real source is `data/raw/urls_dataset2M.csv`:

- source rows: 2,568,184;
- columns: `url,label`;
- normalized labels: benign = 0, malicious = 1; and
- SHA-256: `f934406ec02c317b6eccab8c4f4cf7ac080e8981ececa25fbca863d79239defd`.

After normalization, exact deduplication, conflict rejection, and 128-token
collision auditing, 2,562,284 model-unique URLs remain.

| Frozen split | Benign | Malicious | Total |
|---|---:|---:|---:|
| Train | 869,227 | 1,186,594 | 2,055,821 |
| Validation | 107,873 | 146,425 | 254,298 |
| Test | 108,227 | 143,938 | 252,165 |

Canonical hostnames are assigned as whole groups using seeded BLAKE2b hashing.
This prevents the same hostname from leaking across train, validation, and test.
The character vocabulary is learned from train only. All automated verification
checks pass in `data/processed/verification_report.json`.

## Implemented solution

The real-only controls are a character CNN, character LSTM, Random Forest, and
linear SVM. CNN/LSTM consume the shared 128-token character arrays on CUDA.
RF/SVM consume the same 28 lexical features extracted from exactly the same
manifest rows. Thresholds are selected on validation by maximum F1 while
constraining validation false-positive rate to at most 1%, then frozen for test.

Both URL generators begin with the same autoregressive LSTM language model,
pretrained on a seeded cap of malicious training rows. The two adversarial
methods are:

- Gumbel-LSGAN: differentiable Gumbel-Softmax token distributions, least-squares
  adversarial loss, language-model regularization, feature matching, and a
  diversity term; and
- SeqGAN: hard token sampling, discriminator reward, strided Monte Carlo
  rollouts, policy-gradient updates, and language-model regularization.

A compact RelGAN was subsequently trained as an exploratory third generator.
It uses a relational-memory generator, differentiable Gumbel-Softmax path, and
multi-embedding discriminator. It has since completed full-corpus filtering and
a controlled 10% augmentation screen. Details are in `docs/RELGAN_MODEL.md`.

Neural entry points refuse to run without CUDA. Generator selection does not
use downstream test performance. It rewards validity and uniqueness while
penalizing exact memorization and repeated templates.

Final corpora contain two million raw samples per GAN. Accepted URLs must pass
normalization, generated deduplication, exact-real rejection, encoded-real
rejection, protected-host rejection, and a repeated-template cap. Filters never
rewrite a generated URL; they accept or reject it.

## Current verified results

The shared generator pretraining reached validation NLL 1.8646 and perplexity
6.4537. This is a diagnostic, not evidence that augmentation improves a
classifier.

The completed three-seed mean real-only baselines are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Character LSTM | 0.9645 | 0.9927 | 0.9446 | 0.9681 | 0.9965 |
| Character CNN | 0.9635 | 0.9925 | 0.9431 | 0.9672 | 0.9961 |
| Random Forest | 0.9506 | 0.9917 | 0.9211 | 0.9551 | 0.9880 |
| Linear SVM | 0.9313 | 0.9919 | 0.8870 | 0.9365 | 0.9637 |

Seeds 42, 43, and 44 are complete. LSTM leads all five requested mean metrics,
with CNN a close second.

Gumbel-LSGAN's selected 2,000-sample checkpoint audit scored 99.95% uniqueness,
100% URL-like validity, 0.05% exact memorization, and no repeated-template mass.
Its full two-million-sample raw corpus then achieved 99.3193% uniqueness,
99.9731% validity, 0.0980% exact memorization, and 99.0808% acceptance. The
1,981,616 accepted URLs are 100% unique and have zero exact real overlap.

SeqGAN's selected checkpoint audit scored 100% uniqueness, 100% validity, 0.05%
exact memorization, and no repeated-template mass. Its full raw corpus achieved
99.1599% uniqueness, 99.9745% validity, 0.1162% memorization, and 98.8160%
acceptance. Its 1,976,320 accepted URLs are 100% unique with zero exact-real
overlap. Gumbel-LSGAN is the current generator choice because it improves raw
uniqueness, acceptance, memorization, and template concentration; SeqGAN is
only 0.00145 percentage points higher in raw validity.

RelGAN completed three MLE warm-up epochs and three adversarial epochs on CUDA.
The bounded loader consumed 105,600 distinct rows from the seeded 200,000-row
malicious training view. Its selected 2,000-sample checkpoint audit scored 100%
uniqueness, 99.55% validity, 0.05% exact memorization, zero repeated-template
mass, and 99.5002% overall selection score. On the same checkpoint audit,
Gumbel-LSGAN and SeqGAN both scored 99.95%.

RelGAN's full two-million-sample corpus achieved 99.8839% raw uniqueness,
99.6026% validity, 0.0171% memorization, 0.0611% repeated-template mass, and
99.4878% acceptance. The 1,989,755 accepted URLs are 100% unique and valid,
with zero exact-real overlap and zero repeated-template mass. RelGAN leads the
other GANs on full-corpus uniqueness, acceptance, memorization, and template
concentration, but trails on raw validity.

## Augmentation experiment

The generated dose is defined as generated rows divided by final augmented
training rows. With the fixed 2,055,821 real training rows:

| Dose | Generated rows | Final rows | Role |
|---|---:|---:|---|
| 10% | 228,425 | 2,284,246 | Predeclared primary comparison |
| 20% | 513,955 | 2,569,776 | Exploratory |
| 30% | 881,066 | 2,936,887 | Exploratory |
| 40% | 1,370,547 | 3,426,368 | Exploratory |

The conditions are nested prefixes of one seeded shuffled accepted pool. Thus
the 20% condition contains the 10% generated set, and so forth; every condition
contains all 2,055,821 original real training rows.

Each augmented run is paired with its same-model, same-seed real-only control.
The final comparison reports accuracy, precision, recall, F1, and ROC-AUC
separately, along with mean paired changes and two-sided 95% Student-t confidence
intervals over seeds 42, 43, and 44.

### Completed three-seed 10% confirmation

Both GANs are complete for every classifier and seeds 42, 43, and 44. Values
below are mean paired changes from the same-seed real-only test score:

| GAN | Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Gumbel-LSGAN | CNN | -0.000854 | -0.000451 | -0.001072 | -0.000778 | -0.000250 |
| Gumbel-LSGAN | LSTM | -0.000944 | -0.000574 | -0.001112 | -0.000855 | -0.000269 |
| Gumbel-LSGAN | Random Forest | -0.000156 | -0.000054 | -0.000225 | -0.000146 | -0.000099 |
| Gumbel-LSGAN | Linear SVM | -0.000948 | -0.000130 | -0.001556 | -0.000926 | -0.000430 |
| SeqGAN | CNN | -0.001102 | -0.000183 | -0.001769 | -0.001018 | -0.000395 |
| SeqGAN | LSTM | -0.000022 | -0.000141 | +0.000097 | -0.000014 | -0.000173 |
| SeqGAN | Random Forest | -0.000485 | +0.000472 | -0.001301 | -0.000482 | -0.000275 |
| SeqGAN | Linear SVM | -0.007083 | -0.000370 | -0.012179 | -0.006995 | -0.000710 |

No condition improves all requested metrics. SeqGAN-LSTM is closest: recall
rises by 0.000097, but its paired 95% interval [-0.007962, 0.008157] crosses
zero, while ROC-AUC falls by 0.000173 with interval [-0.000289, -0.000058].

All 20%, 30%, and 40% conditions were also screened at seed 42. None beat its
matching real-only control overall, and degradation generally increased with
dose. Therefore, clearly inferior higher-dose candidates were not promoted to
the optional seed-43/44 confirmation stage.

### RelGAN 10% screen

The RelGAN pool was materialized into the same nested 10%, 20%, 30%, and 40%
manifests. The predeclared 10% condition was screened first with all four
classifiers at seed 42. CNN improved precision only; LSTM improved precision
and ROC-AUC but lost accuracy, recall, and F1; Random Forest improved precision
only; and SVM declined on all five metrics. The LSTM changes were -0.000492
accuracy, +0.000525 precision, -0.001376 recall, -0.000471 F1, and +0.000123
ROC-AUC.

No classifier satisfied the rule requiring improvement across all five metrics.
Consequently, RelGAN was not promoted to seeds 43/44 and its higher-dose
classifier conditions were not run. The result is a completed one-seed screen,
not a three-seed estimate.

### Completed low-data weighted experiment

A frozen, hostname-grouped subset retained 210,434 real training URLs (10.24%
of the full train split), while validation and test remained unchanged. SeqGAN
doses of 1%, 2%, 5%, 10%, and 15% were screened for CNN and LSTM with synthetic
sample weight 0.5. Class balancing was derived only from real rows.

Seed 42 promoted CNN-2% and LSTM-10%. Their matched real-only and augmented
runs were then completed for seeds 42, 43, and 44. CNN-2% was effectively tied:
mean accuracy changed by +0.0013 percentage points (95% CI -0.1174 to +0.1200)
and F1 by +0.0022 points (-0.1121 to +0.1165). LSTM-10% was worse on four of
five means: accuracy changed by -0.1284 points (-0.5460 to +0.2893) and F1 by
-0.1209 points (-0.5162 to +0.2743). Every interval crossed zero.

Reducing the real-data amount and weighting synthetic rows therefore did not
produce a reproducible advantage. The favorable seed-42 screen remains a
screening observation, not proof. Full evidence is in
`docs/LOW_DATA_EXPERIMENT.md` and
`artifacts/low_data/real_10_percent/results/`.

## Interpretation and limits

### Hybrid character/lexical classifier

An isolated fifth classifier was added without altering the original four
models or any dataset split. It fuses the character CNN representation with a
small MLP over the existing 28 lexical features. Across seeds 42/43/44 it
achieved 96.654% accuracy, 99.223% precision, 94.880% recall, 97.003% F1, and
99.669% ROC-AUC on the frozen internal test. This is the best internal mean for
accuracy, recall, F1, and ROC-AUC; the LSTM retains slightly better precision
and FPR.

The hybrid scored only 41.456% mean ROC-AUC on the frozen later
PhishTank/Top-Million snapshot. Consequently, feature fusion improves the
same-source task but does not resolve temporal/source generalization. Full
architecture, metrics, and graphs are in `docs/HYBRID_CNN_EXPERIMENT.md`.

The evidence supports two claims: the Version 2 dataset is leakage-controlled,
and all three GANs can create large, highly unique filtered corpora without retaining
exact real URLs. It does **not** show that naive 10–40% augmentation improves
phishing detection. The completed confirmations and screens instead show small
to substantial degradation depending on the model and dose.

High uniqueness is necessary but not sufficient. A generator can produce many
unique but unrealistic strings. This is why Version 2 also reports validity,
memorization, protected-host overlap, template concentration, language-model
fit, and ultimately downstream classifier changes on unchanged validation/test
sets.

After internal selection was complete, a one-time external snapshot was frozen
on 2026-08-09. It contains 250 OpenPhish Community Feed URLs and 250 legitimate
root-URL proxies sampled from the Tranco top 10,000. Exact URLs, 128-token
encoded inputs, and canonical hostnames appearing anywhere in the master data
were excluded. The final external dataset therefore has zero overlap under all
three checks.

The frozen real-only LSTM and SeqGAN-10% LSTM checkpoints for seeds 42/43/44
were evaluated without retraining, threshold adjustment, or external tuning:

| Model family | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Real-only LSTM | 0.3913 | 0.3269 | 0.3667 | 0.3362 | 0.3671 |
| SeqGAN-10% LSTM | 0.3587 | 0.3369 | 0.4733 | 0.3861 | 0.3032 |

These are three-seed means and show large seed variability. Augmentation raises
mean recall and F1 but lowers accuracy and ROC-AUC, largely by predicting more
URLs as phishing and producing more false positives. Neither model family is
deployment-ready on this snapshot. The large decline from internal results is
evidence of strong temporal/source distribution shift and reinforces that the
current augmentation method does not provide robust external improvement.

### Do the real-only models fail on generated URLs?

No. A direct frozen-classifier challenge was added to answer this separately
from the augmentation experiment. All 12 real-only checkpoints (four model
families, seeds 42/43/44) scored 100,000 accepted unique URLs from each GAN with
their original validation-selected thresholds. They detected 84.66%–93.68% of
the intended-phishing generated URLs. Mean evasion/FNR ranged from 6.32%–9.31%
for Gumbel-LSGAN, 9.46%–12.94% for SeqGAN, and 11.10%–15.34% for RelGAN.

RelGAN's greater evasion is not automatically better. Its standardized lexical
nearest-neighbor distance to real malicious URLs was 1.435, compared with 0.371
for SeqGAN, 0.237 for Gumbel-LSGAN, and a 0.265 real-to-real reference. Its hard
subset also has a strong abnormal double-slash shift. The result is partial
evasion mixed with realism mismatch, not classifier failure. Full methods and
results are in `docs/GENERATED_CHALLENGE_EVALUATION.md`.

### Does quality-aware selection help?

A controlled four-condition experiment now answers this directly. All
augmented conditions used SeqGAN and exactly 228,425 synthetic rows. The two new
pools applied a frozen realism rule; one sampled randomly and the other ranked
normalized evidence from all classifier families/seeds while limiting every
lexical cluster to 3%. A disjoint 20,000-URL challenge was excluded from every
training condition.

With common real-only thresholds, challenging selection improved challenge
recall over random augmentation by +11.05 percentage points for CNN, +12.01 for
LSTM, +9.23 for RF, and +6.02 for SVM. This gain was not free: real-test FPR
rose for every classifier, precision fell, and CNN/LSTM/RF ROC-AUC declined.
When each condition used the established validation-selected threshold to keep
FPR near 1%, real-only retained the best mean F1 for every classifier.

The result proves that sample quality/difficulty changes synthetic robustness,
but not that it universally improves real phishing detection. Full paired
statistics are in `docs/QUALITY_SELECTION_EXPERIMENT.md`.

### Later prospective validity check

A second external snapshot was frozen on 2026-08-11 after the models and
quality-selection conditions already existed. It combines 250 PhishTank
verified-online phishing URLs with 250 root-URL legitimate proxies sampled
from the local Top-Million CSV. Exact URL, model encoding, and hostname overlap
with the master dataset is zero; URL/hostname overlap with the earlier external
snapshot is also zero.

All four classifiers and the real-only, random-10%, realism-filtered-10%, and
filtered-plus-challenging-10% conditions were evaluated across seeds 42/43/44
without retraining or threshold adjustment. The best ROC-AUC was only 0.580
(realism-filtered CNN), with phishing recall 0.325. The highest CNN recall was
0.377, at FPR 0.233. Random-10% LSTM reached recall 0.636 but incorrectly
flagged 69.7% of legitimate roots. All SVM conditions flagged every legitimate
root, while Random Forest missed 91.6–98.8% of current phishes.

This result is a useful negative validity check: none of the existing models or
augmentation strategies provides adequate cross-source generalization. It also
prevents a misleading conclusion from isolated recall, precision, or accuracy
values. Full metrics are in `docs/PHISHTANK_TOPMILLION_EVALUATION.md`.

## Recommended next iteration

### Root cause of the external distribution shift

The completed attribution analysis shows that the real master data is the
primary cause and GAN data is secondary. Training contains a strong reversed
path shortcut: 89.0% of phishing URLs but only 3.4% of benign URLs are
root-like. The external evaluation instead uses 100% legitimate root proxies,
while current PhishTank URLs are longer and contain more paths and queries.
All external URLs also contain explicit scheme markers, compared with roughly
0.01% of training rows.

A source classifier separates internal from external examples with ROC-AUC
1.000 within both labels. Removing schemes without retraining does not restore
performance, confirming that the problem extends beyond formatting. GAN pools
remain far closer to old training phishing than to current PhishTank URLs, so
they reproduce and can amplify the original mismatch. See
`docs/DISTRIBUTION_SHIFT_ROOT_CAUSE.md` for the full controlled analysis.

Earlier results pointed to distribution mismatch rather than mode collapse, so
the controlled Version 2 experiments preserved the master split while testing
synthetic selection. Those experiments are complete. The new root-cause
evidence now shows that a future corrected version must also repair the real
development data rather than continue optimizing augmentation against the same
source-confounded split:

1. **Completed:** test 1%, 2%, and 5% doses (plus 10% and 15%) on the frozen
   10%-real subset;
2. **Completed:** preserve the real class prior through sample weighting and
   calculate class balance from real rows only;
3. **Completed diagnostic:** directly score generated URLs with the frozen
   real-only classifiers and identify a realistic, consensus-hard subset;
4. **Completed:** freeze a realism- and difficulty-aware rule and compare it
   with real-only and random 10% augmentation;
5. **Completed:** control lexical diversity with 64 clusters and evaluate on a
   disjoint 20,000-URL synthetic challenge; and
6. **Completed:** collect a new later real snapshot for one prospective
   evaluation; the result exposed severe source-shift failure modes; and
7. **Completed:** diagnose representation/source shift and attribute the main
   failure to real-data source/path shortcuts, with GAN mismatch secondary; and
8. **Next:** freeze a new single, consistently canonicalized and source-labelled
   development dataset with both root and deep URLs in both classes, then
   validate real-only cross-source performance before retraining GANs.

These steps improve the *usefulness* of generated data while keeping the GANs,
real dataset, validation, and test contract unchanged.

## Reproducibility and evidence

- Configuration: `configs/project.json`
- Methodology: `docs/METHODOLOGY.md`
- Model reasoning: `docs/MODEL_SELECTION.md`
- Decision history: `docs/DECISIONS.md`
- Live result table: `docs/RESULTS.md`
- Frozen-classifier challenge: `docs/GENERATED_CHALLENGE_EVALUATION.md`
- Quality-selection experiment: `docs/QUALITY_SELECTION_EXPERIMENT.md`
- Later external evaluation: `docs/PHISHTANK_TOPMILLION_EVALUATION.md`
- Machine-readable artifacts: `artifacts/`
- Historical, excluded work: `../Version_1/`

Version 1 results and datasets remain preserved but cannot be imported into a
Version 2 training or model-selection run.
