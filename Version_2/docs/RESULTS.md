# Version 2 Results

## Experimental contract

Every result uses the same frozen real dataset and hostname-disjoint split. All
baselines train on the same 2,055,821 real rows. GANs train only on the
1,186,594 malicious rows in that training split. Augmentation adds generated
malicious URLs without changing any real row; validation and test remain fixed.

| Split | Benign | Malicious | Total |
|---|---:|---:|---:|
| Train | 869,227 | 1,186,594 | 2,055,821 |
| Validation | 107,873 | 146,425 | 254,298 |
| Test | 108,227 | 143,938 | 252,165 |

Source SHA-256: `f934406ec02c317b6eccab8c4f4cf7ac080e8981ececa25fbca863d79239defd`.
All checks in `data/processed/verification_report.json` pass, including zero
encoded overlap and zero hostname overlap across splits.

## Real-only baselines (three-seed means)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Character LSTM | **0.964454** | **0.992730** | **0.944645** | **0.968089** | **0.996460** |
| Character CNN | 0.963503 | 0.992540 | 0.943149 | 0.967214 | 0.996119 |
| Random Forest | 0.950569 | 0.991746 | 0.921068 | 0.955101 | 0.987961 |
| Linear SVM | 0.931315 | 0.991858 | 0.886951 | 0.936476 | 0.963661 |

Seeds 42, 43, and 44 are complete for every classifier. The LSTM is the best
original real-only model by all five requested metrics; the CNN is a close
second. The isolated hybrid below was added later and did not replace these
controls.

## Isolated hybrid CNN real-only result

A fifth classifier now combines the existing character CNN representation with
the existing 28 lexical features. It uses all 2,055,821 unchanged real training
rows, no synthetic data, and the same validation/test and threshold policy.

Across seeds 42/43/44, the hybrid achieves 0.966536 accuracy, 0.992233
precision, 0.948802 recall, 0.970032 F1, and 0.996694 ROC-AUC. It is the best
internal real-only classifier for every listed metric except precision and FPR,
where the LSTM remains slightly better. On the frozen PhishTank/Top-Million
snapshot, however, hybrid ROC-AUC is only 0.414560. The hybrid improves
same-distribution detection but does not fix external distribution shift. Of
the paired internal gains, only ROC-AUC has a three-seed 95% interval excluding
zero. See `docs/HYBRID_CNN_EXPERIMENT.md`.

## Generator quality

| Generator | Raw uniqueness | Accepted uniqueness | Validity | Acceptance | Memorization | Template rate |
|---|---:|---:|---:|---:|---:|---:|
| Gumbel-LSGAN | 0.9932 | 1.0000 | **0.9997** | 0.9908 | 0.0010 | 0.0032 |
| SeqGAN | 0.9916 | 1.0000 | 0.9997 | 0.9882 | 0.0012 | 0.0043 |
| RelGAN | **0.9988** | 1.0000 | 0.9960 | **0.9949** | **0.0002** | **0.0006** |

The strictly accepted corpora contain 1,981,616 Gumbel-LSGAN URLs and
1,976,320 SeqGAN URLs. They are 100% unique after filtering and have zero exact
overlap with real data. RelGAN adds 1,989,755 strictly accepted URLs and leads
raw uniqueness, acceptance, memorization, and template concentration, while its
raw validity is lower. High generator quality alone still does not guarantee
useful classifier augmentation.

## RelGAN 10% augmentation screen

RelGAN's 10% condition adds 228,425 accepted generated URLs to all 2,055,821
real training rows. All four classifiers were screened at seed 42. Values are
paired augmented-minus-real-only changes.

| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN | -0.001844 | +0.000537 | -0.003772 | -0.001730 | -0.000073 |
| LSTM | -0.000492 | +0.000525 | -0.001376 | -0.000471 | +0.000123 |
| Random Forest | -0.000520 | +0.000221 | -0.001125 | -0.000503 | -0.000279 |
| Linear SVM | -0.005199 | -0.000469 | -0.008761 | -0.005113 | -0.000271 |

No classifier improved all five metrics. LSTM was closest but lost accuracy,
recall, and F1. The screen therefore failed its promotion rule; no RelGAN
seed-43/44 confirmation or higher-dose classifier runs were performed. These
are one-seed screening results and do not support confidence intervals.

## Confirmed 10% augmentation effect

Values are three-seed means. Delta is augmented minus the paired real-only
score for the same model and seed.

| Generator | Classifier | Accuracy delta | Precision delta | Recall delta | F1 delta | ROC-AUC delta |
|---|---|---:|---:|---:|---:|---:|
| Gumbel-LSGAN | CNN | -0.000854 | -0.000451 | -0.001072 | -0.000778 | -0.000250 |
| Gumbel-LSGAN | LSTM | -0.000944 | -0.000574 | -0.001112 | -0.000855 | -0.000269 |
| Gumbel-LSGAN | Random Forest | -0.000156 | -0.000054 | -0.000225 | -0.000146 | -0.000099 |
| Gumbel-LSGAN | Linear SVM | -0.000948 | -0.000130 | -0.001556 | -0.000926 | -0.000430 |
| SeqGAN | CNN | -0.001102 | -0.000183 | -0.001769 | -0.001018 | -0.000395 |
| SeqGAN | LSTM | -0.000022 | -0.000141 | +0.000097 | -0.000014 | -0.000173 |
| SeqGAN | Random Forest | -0.000485 | +0.000472 | -0.001301 | -0.000482 | -0.000275 |
| SeqGAN | Linear SVM | -0.007083 | -0.000370 | -0.012179 | -0.006995 | -0.000710 |

SeqGAN-10% with LSTM is the closest augmented condition: mean accuracy and F1
are effectively tied with real-only, and recall rises by 0.000097. However, its
paired 95% recall interval is [-0.007962, 0.008157], so the change is not a
reliable improvement. Its ROC-AUC decreases by 0.000173 with a paired 95%
interval of [-0.000289, -0.000058]. No 10% condition improves all requested
metrics.

## Dose screening

Both generators were screened at 10%, 20%, 30%, and 40% using all four
classifiers at seed 42. Every one of the 32 generator/dose/classifier screening
conditions underperformed its matching real-only control on the overall metric
set. Degradation generally grew with dose. For example, SeqGAN CNN accuracy
dropped from 0.961981 at 10% to 0.958087 at 40%; SeqGAN SVM accuracy dropped
from 0.924232 to 0.900514.

The full-data experimental record contains 60 trained runs: 12 real-only runs,
24 three-seed confirmation runs for the two 10% GAN conditions, and 24
seed-42 higher-dose screening runs. The unused cells in the optional 108-run
brute-force matrix are deliberately unrun confirmations of candidates that had
already failed screening.

## Low-data weighted augmentation

A second experiment used one frozen hostname-grouped subset of 210,434 real
training URLs (10.24% of the training split), unchanged full validation/test
sets, and synthetic sample weight 0.5. CNN and LSTM screened SeqGAN at 1%, 2%,
5%, 10%, and 15% using seed 42. CNN-2% and LSTM-10% were then confirmed with
paired seeds 42, 43, and 44.

| Condition | Metric | Real-only mean | Augmented mean | Mean paired change, pp (95% CI) |
|---|---|---:|---:|---:|
| CNN + SeqGAN 2% | Accuracy | 0.953188 | 0.953201 | +0.0013 (-0.1174, +0.1200) |
| CNN + SeqGAN 2% | F1 | 0.957550 | 0.957571 | +0.0022 (-0.1121, +0.1165) |
| CNN + SeqGAN 2% | ROC-AUC | 0.993799 | 0.993853 | +0.0053 (-0.0215, +0.0322) |
| LSTM + SeqGAN 10% | Accuracy | 0.953355 | 0.952071 | -0.1284 (-0.5460, +0.2893) |
| LSTM + SeqGAN 10% | F1 | 0.957734 | 0.956524 | -0.1209 (-0.5162, +0.2743) |
| LSTM + SeqGAN 10% | ROC-AUC | 0.993291 | 0.993265 | -0.0026 (-0.0315, +0.0262) |

Every 95% interval crosses zero. CNN-2% is effectively tied with real-only;
LSTM-10% is worse on four of five mean metrics. The favorable seed-42 screen
did not become reproducible evidence. Full precision/recall results are in
`docs/LOW_DATA_EXPERIMENT.md` and the adjacent machine-readable artifacts.
This low-data experiment adds 20 trained runs, bringing the Version 2 total to
80 classifier runs. The four RelGAN screen runs bring that total to 84. The 24
new realism-filtered and challenging runs brought that total to 108. The three
real-only hybrid-CNN seeds bring the current Version 2 total to 111 trained
classifier runs.

## One-time external snapshot

After internal model selection, the frozen real-only LSTM and SeqGAN-10% LSTM
were evaluated on 500 newly collected URLs (250 OpenPhish phishing URLs and 250
Tranco legitimate root-URL proxies). The snapshot has zero exact, encoded, or
hostname overlap with the master dataset.

| Model family | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Real-only LSTM | 0.3913 | 0.3269 | 0.3667 | 0.3362 | 0.3671 |
| SeqGAN-10% LSTM | 0.3587 | 0.3369 | 0.4733 | 0.3861 | 0.3032 |

Values are means over seeds 42/43/44 and have high between-seed variability.
SeqGAN improves mean recall and F1 but lowers accuracy and ROC-AUC; neither
family generalizes adequately. These data are evaluation-only and cannot be
used to tune the existing models.

## Later PhishTank / Top-Million snapshot

On 2026-08-11, a second prospective snapshot was frozen from independent
sources: 250 verified-online PhishTank URLs and 250 legitimate root-URL proxies
sampled from the local Top-Million CSV. It has zero exact, encoded, or hostname
overlap with the master data and zero URL/hostname overlap with the earlier
external snapshot.

All four classifiers, all four controlled quality conditions, and seeds
42/43/44 were evaluated using their unchanged validation-selected thresholds.
No condition generalized adequately. The best ROC-AUC was 0.580 for the
realism-filtered CNN, but its recall was only 0.325. The highest CNN recall was
0.377 for filtered-plus-challenging, with FPR 0.233. Random-10% LSTM recall was
0.636 but FPR was 0.697. Every SVM condition flagged all legitimate URLs
(FPR 1.000), while Random Forest missed 91.6–98.8% of phishes.

These results show severe source/temporal shift and do not validate any model
for deployment. Some augmentation conditions move individual metrics in a
favorable direction, but none creates an acceptable or consistently superior
operating point. Full results are in
`docs/PHISHTANK_TOPMILLION_EVALUATION.md` and
`artifacts/results/external/2026-08-11_phishtank_topmillion/`.

## Frozen real-only classifier challenge

To test generated URLs directly, 100,000 strictly accepted unique URLs from
each GAN were passed through all real-only CNN, LSTM, Random Forest, and linear
SVM checkpoints for seeds 42/43/44. Checkpoints and thresholds were unchanged.
Because each set contains intended-phishing examples only, the valid direct
metrics are detection/recall and false-negative/evasion rate—not accuracy,
precision, FPR, or ROC-AUC.

| Generator | CNN FNR | LSTM FNR | RF FNR | SVM FNR | Majority evasion | Plausibly hard |
|---|---:|---:|---:|---:|---:|---:|
| Gumbel-LSGAN | 6.32% | 6.54% | 7.47% | 9.31% | 7.38% | 4.73% |
| SeqGAN | 9.46% | 9.55% | 11.13% | 12.94% | 10.97% | 6.89% |
| RelGAN | 11.10% | 11.37% | 12.69% | 15.34% | 12.35% | 7.64% |

The classifiers detect 84.66%–93.68% of generated URLs, so they do not simply
fail on GAN output. RelGAN is most evasive, but its mean lexical nearest-neighbor
distance is 1.435 versus a 0.265 real-to-real reference and its hard subset has
an extreme double-slash shift. Thus higher RelGAN evasion is partly confounded
by realism mismatch. See `docs/GENERATED_CHALLENGE_EVALUATION.md` for the full
protocol, realism table, limitations, and interpretation.

## Controlled quality-selection experiment

The next predeclared experiment held the source generator (SeqGAN), 228,425-row
10% synthetic count, real rows, splits, models, and seeds fixed. It compared
random augmentation with realism-filtered random selection and
realism-filtered, ensemble-challenging selection. A disjoint 20,000-URL
synthetic challenge was reserved before selection and never used for training.

Under common real-only thresholds, challenging selection improved held-out
synthetic recall relative to random augmentation by +11.05 percentage points
for CNN, +12.01 for LSTM, +9.23 for RF, and +6.02 for SVM. However, it also
increased real-test FPR for every classifier and reduced CNN/LSTM/RF ROC-AUC.

Under each condition's established validation-selected threshold, mean
real-test F1 remained best for real-only: 96.721% CNN, 96.809% LSTM, 95.510% RF,
and 93.648% SVM. The challenging condition scored 96.342%, 96.470%, 95.237%,
and 92.630%, respectively. Quality-aware selection therefore improves
synthetic robustness but does not establish a universal real-test advantage.
Full paired results, confidence intervals, selection rules, and limitations are
in `docs/QUALITY_SELECTION_EXPERIMENT.md`.

## Conclusion

### Distribution-shift attribution

A dedicated lexical/source analysis establishes that the primary external
failure comes from the real dataset and acquisition protocol, not from GAN
samples. The internal master associates root-like URLs with phishing (89.0%
root phishing versus 3.4% root benign), while the external Top-Million half is
100% legitimate roots and PhishTank contains longer, deeper URLs. A lexical
diagnostic falls from internal ROC-AUC 0.963 to external 0.346, and source can
be identified with ROC-AUC 1.000 within either label.

SeqGAN has mean lexical KS distance 0.020 from old training phishing but 0.333
from external phishing. The other GAN and filtered pools show the same pattern.
Thus GANs inherit and sometimes amplify the old-data mismatch; they do not
originate it. Full profiles, canonicalization ablations, and graphs are in
`docs/DISTRIBUTION_SHIFT_ROOT_CAUSE.md`.

The current experiment does **not** demonstrate that GAN augmentation improves
phishing-URL detection. It demonstrates that all three GANs generate large,
non-memorized, highly unique corpora, while naive malicious-only addition at
10–40% does not improve the frozen real test distribution. Reducing the real
training set, testing 1–15% doses, and down-weighting synthetic samples also did
not produce a stable improvement. Quality-aware selection improves detection of
held-out generated challenges, but its real-test recall gain trades against FPR,
precision, and ROC-AUC. The evidence points to synthetic/real distribution
mismatch rather than classical mode collapse alone.

RelGAN's full corpus, seed-42 augmentation screen, and frozen-classifier
challenge reinforce the conclusion: better uniqueness and stronger evasion do
not necessarily translate into realistic samples or downstream improvement.

Machine-readable results are in `artifacts/results/augmentation/summary.json`
and `summary.csv`. Direct generated-challenge evidence is in
`artifacts/results/generated_challenge/`. Controlled quality-selection results
are in `artifacts/results/quality_experiment/`. The later external results are
in `artifacts/results/external/2026-08-11_phishtank_topmillion/`. Presentation
graphs are stored beside their corresponding machine-readable summaries.
