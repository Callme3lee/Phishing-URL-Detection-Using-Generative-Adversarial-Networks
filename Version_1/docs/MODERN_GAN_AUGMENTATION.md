# Modern URL GAN Adaptation and Augmentation

## Outcome

Modern, quality-filtered LSGAN augmentation improves the modern temporal-test
mean over the matched modern real-only condition for all requested metrics. The
main gains are recall and F1 while precision remains approximately 99% and the
false-positive rate remains below 1%.

This result repairs the specific distribution-shift failure found in the 31 July
stress test. It does not turn that same stress test into independent proof: its
failure motivated this design, so a newly collected future snapshot is still
required for the final unbiased claim.

## Data and leakage control

The modern malicious corpus combines:

- OpenPhish public-feed history from 1 May through 1 August 2026 (185
  twice-daily snapshots in the selected window); and
- verified-online PhishTank URLs submitted in the same window.

Every canonical exact hostname found in any row of the three previous project
CSVs was removed. Every hostname in the frozen 31 July evaluation was also
removed. URLs were deduplicated across snapshots and sources. Hostnames were
grouped by their earliest observation before temporal splitting:

| Split | Time role | Phishing URLs |
|---|---|---:|
| Train | Before 25 July | 54,126 |
| Validation | 25–28 July | 2,359 |
| Temporal test | 29 July–1 August | 2,619 |

The corpus contains 45,464 unique modern malicious hostnames. A separate Tranco
snapshot supplied 50,000 unfamiliar legitimate hard-negative domain roots after
the same prior/frozen hostname exclusions. Source hashes, commit identifiers and
all counts are in `artifacts/modern_training_data/encoded/metadata.json`.

## GAN training

The existing autoregressive generator was first adapted for three supervised
epochs on all 54,126 modern training URLs. Temporal-validation perplexity fell
from 8.26 to 7.07. That checkpoint initialized two adversarial models:

- LSGAN: three full 54,126-URL epochs, language weight 2.0, feature weight 0.5,
  diversity weight 0.1, final sampling temperature 0.8.
- WGAN-GP: one 10,000-URL comparison epoch with three critic steps and gradient
  penalty 10, sized for the 6 GB RTX 3060 laptop GPU.

On the initial strictly filtered samples:

| GAN | Strict acceptance | Uniqueness | Valid structure | Memorization | Bigram diversity | Frozen-CNN detection |
|---|---:|---:|---:|---:|---:|---:|
| Modern LSGAN | 78.1% | 100% | 100% | 0% | 4.15% | 93.60% |
| Modern WGAN-GP | 84.7% | 100% | 100% | 0% | 3.57% | 84.65% |

LSGAN was selected because its accepted output was more diverse and more
consistently malicious-looking to the frozen classifier. A larger 5,000-sample
LSGAN generation produced 4,378 strictly accepted URLs (87.56%). The strict
filter removed invalid syntax, duplicates, every exact real-URL memorization,
and every collision with 2,049,204 prior/frozen hostname hashes.

## Controlled classifier experiment

The shared real training base has 19,701 modern phishing and 19,701 unfamiliar
benign URLs. Adding all 4,378 accepted LSGAN URLs makes the augmented training
set exactly 10% generated. Both conditions initialize from the same older CNN,
fine-tune with the same hyperparameters over seeds 42/43/44, and use a validation
threshold that maximizes F1 subject to false-positive rate <= 1%.

### Temporally held-out result

Mean over three seeds on 2,619 later phishing plus 2,619 disjoint benign URLs:

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Modern real-only | 82.56% | 99.00% | 65.79% | 79.04% | 96.88% | 0.662% |
| Modern + 10% LSGAN | **84.78%** | **99.05%** | **70.24%** | **82.19%** | **97.44%** | 0.675% |
| Augmented minus real-only | **+2.22 pp** | +0.04 pp | **+4.45 pp** | **+3.15 pp** | **+0.56 pp** | +0.013 pp |

Paired seed 95% confidence intervals exclude zero for accuracy (+1.09 to +3.35
pp), recall (+2.09 to +6.82 pp), F1 (+1.47 to +4.83 pp), and ROC-AUC (+0.21 to
+0.91 pp). Precision is effectively unchanged and its interval crosses zero.

The graph is `artifacts/modern_training_data/summary/comparison.png`; exact
values are in `summary.json` and `summary.csv` beside it.

### Frozen 31 July regression check

The already-selected checkpoints and thresholds were also applied to the
excluded 510-example July snapshot. No threshold was changed. Mean real-only
versus augmented results were 83.86% vs 85.49% accuracy, 67.71% vs 70.98%
recall, 80.71% vs 83.01% F1, and 97.72% vs 98.16% ROC-AUC. Precision was 100%
and false-positive rate was 0% for both conditions across all seeds.

This is strong regression evidence that modern hard-negative adaptation fixes
the earlier 99–100% false-positive collapse. It is not an independent final
test because the earlier failure influenced the repair. Results and a separate
graph are under `artifacts/modern_training_data/summary_frozen_regression/`.

## Reproduction

Prepare the timestamped corpus after obtaining the official source snapshots:

```bash
PYTHONPATH=src python3 -m phishing_gans.prepare_modern_training --help
```

Adapt and adversarially train the generators:

```bash
PYTHONPATH=src python3 -m phishing_gans.pretrain_generator --help
PYTHONPATH=src python3 -m phishing_gans.finetune_autoregressive_gan --help
```

Generate and strictly filter the LSGAN corpus:

```bash
PYTHONPATH=src python3 -m phishing_gans.generate_corpus --help
PYTHONPATH=src python3 -m phishing_gans.filter_corpus --help
```

Prepare the balanced classifier experiment and train it:

```bash
PYTHONPATH=src python3 -m phishing_gans.prepare_modern_classifier --help
PYTHONPATH=src python3 -m phishing_gans.train_classifier --help
```

All neural training and inference paths call `require_cuda()` and do not fall
back to CPU.

## Scientific conclusion

The supported claim is now: **under a timestamped, hostname-separated temporal
development protocol, adding 10% quality-filtered modern LSGAN phishing URLs to
a modern balanced real corpus improves mean accuracy, recall, F1 and ROC-AUC
over modern real-only fine-tuning while maintaining approximately 99% precision
and below-1% false-positive rate.** A future untouched collection is required
to confirm this prospectively.
