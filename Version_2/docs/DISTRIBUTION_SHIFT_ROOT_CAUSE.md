# Distribution-Shift Root-Cause Analysis

## Direct answer

The primary cause is the **real dataset and evaluation-source mismatch**, not
the GAN samples. GAN augmentation is a secondary amplifier because the GANs
faithfully reproduce the old malicious training distribution.

This conclusion follows from three independent observations:

1. Frozen real-only models, which never saw GAN data, already fail on the
   PhishTank/Top-Million snapshot.
2. Internal train and test lexical distributions are nearly identical, while
   both external classes are perfectly separable from their internal
   counterparts by a small source classifier.
3. Every GAN pool is much closer to old training phishing than to current
   PhishTank phishing.

The August 11 snapshot is now diagnostic and must not be reused as an unbiased
final test for a future corrected model.

## Data examined

- 100,000 sampled training-benign URLs;
- 100,000 sampled training-phishing URLs;
- 50,000 internal-test URLs per class;
- all 250 external Top-Million legitimate proxies;
- all 250 external PhishTank verified-online URLs;
- 100,000 samples from each GAN and each quality-selected SeqGAN pool; and
- all existing real-only and 10% augmented external evaluations.

## Cause 1: a reversed path/root shortcut

The strongest substantive problem is class-conditional URL structure:

| Group | Root/empty path | Mean path length | Query present | Mean URL length |
|---|---:|---:|---:|---:|
| Training benign | 3.43% | 37.43 | 18.07% | 63.44 |
| Training phishing | 89.04% | 3.34 | 1.79% | 26.43 |
| External legitimate | **100.00%** | 1.00 | 0.00% | 23.65 |
| External phishing | 66.80% | 8.07 | 10.40% | 45.50 |

The frozen training data strongly associates deep paths and queries with
benign URLs and root-like URLs with phishing. The external acquisition process
reverses that relationship: every Top-Million legitimate proxy is a root URL,
while current PhishTank URLs are longer and more structurally varied. Models
therefore learn a source shortcut that works on the same-source internal test
but reverses externally.

A lexical logistic diagnostic reaches ROC-AUC 0.963 on the internal test but
only 0.346 externally, below random ranking. This is direct evidence that
old-data lexical relationships reverse rather than merely become noisier.

## Cause 2: inconsistent acquisition/canonicalization format

Almost no sampled training URL retains a scheme marker:

| Group | HTTPS | Contains `://` |
|---|---:|---:|
| Training benign | 0.00% | 0.01% |
| Training phishing | 0.00% | 0.01% |
| External legitimate | 100.00% | 100.00% |
| External phishing | 82.00% | 100.00% |

Thus `https`, double-slash, colon, and slash count are out-of-support external
features. A five-fold source classifier distinguishes internal from external
URLs with ROC-AUC 1.000 for both labels.

However, a no-retraining ablation shows this is not a scheme-only problem.
Stripping schemes fails to restore ROC-AUC. Converting root URLs to the old
training style makes false-positive rates approach 100%, because the deeper
root/path label shortcut remains. The solution is consistent canonicalization
followed by retraining on structurally balanced data—not post-hoc editing of
test URLs.

## Cause 3: current PhishTank structure is underrepresented

Compared with sampled training phishing, external PhishTank URLs have:

- mean URL length 45.50 versus 26.43;
- mean path length 8.07 versus 3.34;
- mean query length 4.44 versus 0.54;
- mean digit ratio 0.076 versus 0.036;
- mean entropy 4.215 versus 3.705;
- suspicious-token presence 12.0% versus 0.85%; and
- length above the 127-character payload limit 2.4% versus 0.21%.

These are source and possibly temporal differences. Because PhishTank and the
original master have different, undocumented acquisition histories, the
current evidence cannot cleanly separate time drift from source drift. It does
establish that the current malicious distribution is underrepresented.

## GAN attribution

Mean lexical Kolmogorov–Smirnov distance across the 28 features:

| Synthetic pool | Distance from training phishing | Distance from external phishing |
|---|---:|---:|
| Gumbel-LSGAN | 0.033 | 0.338 |
| SeqGAN | **0.020** | 0.333 |
| RelGAN | 0.059 | 0.312 |
| Realism-filtered SeqGAN | 0.032 | 0.335 |
| Filtered + challenging SeqGAN | 0.092 | **0.278** |

The challenging pool moves somewhat toward current phishing, but remains much
closer to the old training distribution. The GANs learned what they were given;
more epochs on the same data would reinforce rather than solve the mismatch.

Across the original four classifier families, external macro ROC-AUC is 0.473
for real-only, 0.459 for random 10%, 0.455 for realism-filtered 10%, and 0.432
for filtered-plus-challenging 10%. Individual CNN/LSTM changes are mixed, but
augmentation does not consistently repair the shift and generally worsens the
four-model average.

## Why internal metrics stayed high

Internal test versus training mean KS distance is only 0.0034 for benign and
0.0077 for phishing. Both came from the same master source and share nearly
identical class-specific artifacts. Hostname-disjoint splitting prevents exact
host leakage, but it cannot prevent source or acquisition-style shortcuts.

Consequently, the internal test measures same-source generalization well but
does not estimate cross-source performance.

## Required correction

1. Apply one deterministic URL canonicalization policy before any split.
2. Construct one source-labelled development CSV containing both root and deep
   URLs in both classes.
3. Balance or stratify scheme, root/path, length, and source properties so they
   cannot act as label proxies.
4. Use hostname-, source-, and time-aware validation/test partitions and report
   source-macro metrics.
5. Retrain real-only models first; do not use GAN data until cross-source
   real-only ROC-AUC and FPR are acceptable.
6. Retrain GANs only on the corrected malicious training view, then require
   similarity to multiple malicious development sources—not only the old
   training centroid.
7. Collect a later untouched snapshot after every rule and threshold is frozen.

Machine-readable evidence and graphs are in
`artifacts/results/distribution_shift/`.
