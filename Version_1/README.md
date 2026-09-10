# GPU URL Classification and GAN Project

This repository builds a reproducible malicious/phishing URL experiment using:

1. A compact GPU character-level CNN classifier.
2. A compact GPU character-level LSTM classifier.
3. CPU Random Forest and Linear SVM classifiers using lexical URL features.
4. A character-level Least-Squares GAN (LSGAN).
5. A character-level Wasserstein GAN with gradient penalty (WGAN-GP).

The current dataset is `Data sets/urls_dataset2M.csv`, containing the labels
`benign` and `malicious`. Before describing all positive examples as *phishing*
in the final report, verify that the dataset source defines `malicious` that way.

## Why this classifier?

The character CNN learns URL patterns directly, avoids manual feature engineering,
and is small enough for a laptop RTX 3060. It also uses the same fixed-length
character representation that the GANs will use.

## Requirements

- Python 3.11 or newer
- An NVIDIA GPU with a working driver
- A CUDA-enabled PyTorch installation

Confirm GPU access before training:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Training intentionally refuses to fall back to CPU.

## Step 1: Prepare the 2M dataset

From the repository root:

```bash
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans.prepare_data
```

The preparation step:

- maps `benign` to 0 and `malicious` to 1;
- removes exact duplicate URL strings;
- builds the character vocabulary from training data only;
- pads/truncates URLs to 128 characters;
- creates an 80/10/10 split using a stable hash of each hostname; and
- stores compact memory-mapped NumPy arrays under `artifacts/data_2m`.

Grouping by hostname prevents the same hostname from appearing in multiple
splits. This is stricter than an ordinary random row split.

For a quick preparation test:

```bash
python3 -m phishing_gans.prepare_data \
  --max-rows 10000 \
  --output-dir /tmp/url-data-smoke
```

## Step 2: Train the GPU character classifiers

```bash
python3 -m phishing_gans.train_classifier
```

Train the LSTM on the identical encoded data and splits:

```bash
python3 -m phishing_gans.train_classifier \
  --model lstm \
  --epochs 5 \
  --batch-size 512
```

Laptop-friendly defaults are five epochs, batch size 1024, automatic mixed
precision, and four data-loader workers. If GPU memory is insufficient:

```bash
python3 -m phishing_gans.train_classifier --batch-size 512
```

Outputs are saved under `artifacts/classifier`:

- `best.pt`: best validation-F1 checkpoint;
- `metrics.json`: validation history and untouched real-URL test results.

The decision threshold is selected on validation data only. Each test report
includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate,
and the confusion matrix.

## Step 3: Train Random Forest and Linear SVM

Extract the 28 URL-only lexical features once:

```bash
python3 -m phishing_gans.prepare_lexical_features
```

Train Random Forest on all training rows:

```bash
python3 -m phishing_gans.train_traditional \
  --model rf \
  --rf-trees 150 \
  --rf-max-depth 24
```

Train the scalable linear SVM:

```bash
python3 -m phishing_gans.train_traditional --model svm
```

The Random Forest and Linear SVM are scikit-learn CPU reference baselines. On
this machine, full RF training takes about 104 seconds with 16 CPU threads and
SVM training takes about 17 seconds. A nonlinear/kernel SVM is intentionally
not used because its time and memory scaling are unsuitable for 2.06 million
training samples.

The lexical features include URL, hostname, path, query and fragment lengths;
character-type counts and ratios; punctuation counts; subdomain and TLD
lengths; IP-hostname and HTTPS flags; character entropy; suspicious keywords;
and the longest digit run. They require only the URL string—no webpage fetching.

## Baseline results

All results below use the same untouched hostname-separated test set of 252,430
real URLs. Thresholds were selected using validation data only.

| Model | Representation | F1 | Precision | Recall | ROC-AUC | False-positive rate |
|---|---|---:|---:|---:|---:|---:|
| Character CNN | learned characters | 97.05% | 97.52% | 96.59% | 99.58% | 3.26% |
| Character LSTM | learned character sequence | 97.17% | 97.32% | 97.02% | 99.61% | 3.55% |
| Random Forest | 28 lexical features | 96.08% | 97.27% | 94.92% | 99.03% | 3.53% |
| Linear SVM | 28 standardized lexical features | 94.36% | 97.49% | 91.42% | 96.27% | 3.13% |

The Random Forest's most important features were slash count, path length,
special-character count, total URL length, letter count, character entropy,
hostname length, letter ratio, digit count, and longest digit run.

Exact thresholds, confusion matrices, histories, and unrounded values are in:

- `artifacts/classifier/metrics.json` (CNN)
- `artifacts/classifier_lstm/metrics.json` (LSTM)
- `artifacts/classifier_rf/metrics.json` (Random Forest)
- `artifacts/classifier_svm/metrics.json` (Linear SVM)

## GAN experiments

The implemented LSGAN and WGAN-GP pilot commands, architecture decisions,
quality metrics, and negative mode-collapse findings are documented in
`docs/GAN_EXPERIMENTS.md`. Neither pilot's samples are approved for classifier
augmentation.

The subsequent 50K-scale corpus filtering and controlled real-only versus
oversampling versus GAN-augmentation results are documented in
`docs/AUGMENTATION_EXPERIMENTS.md`.

## Reproduce the external evaluation

Prepare the leakage-screened external test arrays once:

```bash
python3 -m phishing_gans.prepare_external
```

Prepare `new_data_urls.csv`, whose source labels use `1=benign` and
`0=malicious`, with the explicit reversed mapping:

```bash
python3 -m phishing_gans.prepare_external \
  --external "Data sets/new_data_urls.csv" \
  --output-dir artifacts/external_new_data_urls \
  --label-column status \
  --benign-label 1 \
  --malicious-label 0
```

Evaluate a frozen checkpoint using its original validation-selected threshold:

```bash
python3 -m phishing_gans.evaluate_checkpoint \
  --checkpoint artifacts/classifier/best.pt \
  --data-dir artifacts/external_phishing_test \
  --output artifacts/classifier/external_metrics.json
```

The same command can evaluate the LSTM, oversampling, LSGAN, and WGAN-GP
checkpoints by changing `--checkpoint` and `--output`. Do not select a new
threshold on the external test set, because that would leak test information.

The two-source external comparison figure can be reproduced with:

```bash
python3 -m phishing_gans.plot_external_comparison
```

## Retrained multi-source GAN and final results

The retrained autoregressive LSGAN and WGAN-GP use the malicious training
portions of the 2M dataset and both added sources. Exact duplicates are removed,
and added-source URLs are separated by a stable hostname hash into 80% training,
10% validation, and 10% held-out data.

Build the GAN-only corpus with:

```bash
python3 -m phishing_gans.prepare_multisource_gan_data
```

The consolidated project results—including Accuracy, Precision, Recall, F1,
ROC-AUC for CNN/LSTM/Random Forest/linear SVM and uniqueness for both retrained
GANs—are generated by:

```bash
python3 -m phishing_gans.project_results
```

This writes `artifacts/final_project_results.json`. The selected corpus
checkpoints are epoch 2 for both models; selection was based on diagnostic
validity, uniqueness, memorization, bigram diversity, and malicious-style
classifier scores rather than automatically using the final epoch.

## Controlled 20% / 30% / 40% augmentation matrix

Create the nine unique GAN-percentage CSVs and their shared real-only reference:

```bash
python3 -m phishing_gans.create_augmented_datasets
```

Encode character tensors and lexical features while retaining the original
fixed validation and test sets:

```bash
python3 -m phishing_gans.prepare_augmented_experiments
```

Run or resume all CNN, LSTM, RF, and linear-SVM conditions:

```bash
python3 -m phishing_gans.run_augmented_experiments
```

Aggregate the metrics and create the F1 dose-response graph:

```bash
python3 -m phishing_gans.summarize_augmented_experiments
python3 -m phishing_gans.plot_augmented_results
```

The complete interpretation is in `docs/AUGMENTED_PERCENTAGE_RESULTS.md`, and
machine-readable values are in `artifacts/augmented_experiments/summary.json`
and `summary.csv`.

## GAN-unseen external evaluation

The exact hostname-held-out 10% portions of both added external sources can be
reconstructed with `phishing_gans.prepare_external_heldout`. Evaluate every
frozen checkpoint without threshold retuning and aggregate the results with:

```bash
python3 -m phishing_gans.evaluate_augmented_external
python3 -m phishing_gans.summarize_external_heldout
python3 -m phishing_gans.plot_external_heldout_results
```

The interpretation and warning about misleading F1 under threshold collapse are
in `docs/UNSEEN_EXTERNAL_RESULTS.md`. Complete values are saved as
`artifacts/augmented_experiments/external_heldout_summary.json` and `.csv`.

## Source-balanced quality-filtered follow-up

The next robustness experiment uses balanced real samples from all three sources,
encoded-collision screening, quality-filtered LSGAN samples at 10%, 20%, and 30%,
three training seeds, and matching-source validation calibration. Prepare and
summarize it with:

```bash
python3 -m phishing_gans.prepare_robust_augmentation
python3 -m phishing_gans.add_robust_lsgan_doses
python3 -m phishing_gans.summarize_robust_augmentation
python3 -m phishing_gans.summarize_robust_external --calibrated
```

The complete protocol and interpretation are in
`docs/ROBUST_AUGMENTATION_RESULTS.md`. Quality-filtered LSGAN beats real-only
balanced-test F1 for CNN and SVM, while real-only remains best for source-macro
F1. GAN augmentation gives the best source-macro ROC-AUC for CNN and SVM.

## Augmented pretraining and real-only fine-tuning

The completed improvement initializes a CNN from each quality-LSGAN
checkpoint, then fine-tunes it for two low-learning-rate epochs on the shared
real-only corpus:

```bash
python3 -m phishing_gans.run_augmented_finetuning
```

The design and verified results are documented in
`docs/AUGMENTED_FINETUNING.md`. The 20%-then-real CNN achieves the best measured
balanced F1 and source-macro F1; the 30%-then-real CNN achieves the best
source-macro ROC-AUC. This experiment requires CUDA and does not fall back to
CPU.

## Later-collected external stress test

A frozen 31 July 2026 snapshot combines 255 OpenPhish URLs with 255 Tranco
domain-root URLs after removing every canonical exact hostname found anywhere
in the three prior project datasets. All real-only and 20%-then-real CNN seeds
were evaluated on GPU with their original thresholds. Both model families fail
under the new source/domain shift: real-only averages 45.23% accuracy and 64.78%
ROC-AUC, while augmented pretraining averages 44.71% accuracy and 55.88%
ROC-AUC. False-positive rates are approximately 99–100%.

The strict protocol, graph, limitations, and next actions are in
`docs/LATER_COLLECTED_EVALUATION.md`. The frozen inputs and machine-readable
results are under `artifacts/later_collected_2026-07-31/`.

## Modern GAN adaptation and hard-negative repair

The follow-up corpus combines timestamped May–August 2026 OpenPhish history and
verified-online PhishTank URLs while excluding every prior-project and frozen
test hostname. The selected modern LSGAN produced 4,378 strictly accepted URLs
from 5,000 generations. In a paired three-seed temporal experiment, exact-10%
LSGAN augmentation improves mean accuracy from 82.56% to 84.78%, recall from
65.79% to 70.24%, F1 from 79.04% to 82.19%, and ROC-AUC from 96.88% to 97.44%,
while precision remains approximately 99% and FPR remains below 1%.

The complete methodology, confidence intervals, GPU commands, limitations and
frozen-snapshot regression check are in `docs/MODERN_GAN_AUGMENTATION.md`.
Machine-readable outputs and graphs are under
`artifacts/modern_training_data/`.

## Post-training snapshot smoke test

A strict 1 August 2026 PhishStats snapshot yielded only six genuinely
post-cutoff unseen-host phishing URLs after excluding every old, modern,
generated and previously evaluated hostname. Paired with six unused benign
controls, modern + 10% LSGAN improves three-seed mean accuracy from 75.00% to
80.56%, recall from 50.00% to 61.11%, and F1 from 66.67% to 75.56%, with 100%
precision and zero false positives. The sample is far too small for significance;
the correct interpretation is a favorable smoke test only. See
`docs/POST_TRAINING_SNAPSHOT.md` and
`artifacts/post_training_snapshot_2026-08-01_0904Z/`.

## Prospective 300-URL evaluation

The next independent evaluation is now protected by an append-only accumulator.
It starts with six genuinely post-training unseen-host phishing URLs and targets
300 before any further evaluation. The exclusion index covers 2,149,042 old,
modern, generated and previously tested hostnames. Run the collection entry
point once per day:

```bash
./scripts/collect_prospective_snapshot.sh
```

Daily collection is installed in the user crontab for 14:15 local time; logs
append to `artifacts/prospective_evaluation/collector.log`.

The freezer refuses to create the evaluation below target or overwrite a frozen
set. Protocol, current status and final freeze command are documented in
`docs/PROSPECTIVE_EVALUATION.md`; live state is in
`artifacts/prospective_evaluation/status.json`.

## Immediate 600-URL external dataset

An immediate independently collected evaluation set is frozen under
`artifacts/immediate_external_evaluation_2026-08-01/frozen_300/`. It contains
300 exact-hostname-unseen phishing URLs from current PhishTank/PhishStats feeds
and 300 unused Tranco controls. All 600 hostnames are unique and overlap none of
the 2,149,042 prior/training/generated/evaluation hostname exclusions. The six
frozen modern checkpoints were evaluated once with no test-set threshold
recalibration. Across three paired seeds, 10% modern LSGAN augmentation improved
mean accuracy from 72.67% to 75.78%, recall from 45.78% to 52.00%, F1 from
62.28% to 68.05%, and ROC-AUC from 96.43% to 97.29%; precision increased from
99.09% to 99.18%, while FPR stayed at 0.44%. See
`docs/IMMEDIATE_EXTERNAL_300.md` and the graph at
`artifacts/immediate_external_evaluation_2026-08-01/summary/comparison.png`.

Repeated-run metrics can be aggregated reproducibly with:

```bash
python3 -m phishing_gans.aggregate_experiments --help
```

It accepts one `--condition NAME FILE...` group per training condition and
writes means, sample standard deviations, individual run values, and two-sided
95% Student-t confidence intervals. The completed three-seed output is
`artifacts/multiseed_10pct_summary.json`.

In some restricted container environments, NVIDIA device nodes may be hidden.
Normal host-terminal commands can still use the GPU when `nvidia-smi` and
`torch.cuda.is_available()` both succeed.
