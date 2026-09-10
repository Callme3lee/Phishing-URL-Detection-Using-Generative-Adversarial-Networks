# Version 2 — Single-Dataset Phishing URL GAN Study

## Objective

Version 2 measures whether GAN-generated malicious URLs improve phishing URL
classification when every real-only and augmented classifier is built from one
immutable real dataset and one immutable hostname-separated split.

## One-dataset rule

The only real dataset permitted for development is:

`data/raw/urls_dataset2M.csv`

Its frozen SHA-256 is:

`f934406ec02c317b6eccab8c4f4cf7ac080e8981ececa25fbca863d79239defd`

The file has 2,568,184 data rows with columns `url,label`. Version 2 normalizes
labels to `0 = benign` and `1 = malicious`.

“One dataset” means one master source and one split manifest, with two valid
role-specific views:

- CNN, LSTM, Random Forest, and linear SVM use every labelled training row.
- GANs use only label-1 rows from the **same training split**. GANs must not see
  validation or test rows because their output influences augmented training.

Generated URLs are not a second real dataset. They are an experimental addition
to the unchanged real training rows.

## Selected models

Real-only baselines:

1. Character CNN
2. Character LSTM
3. Random Forest on lexical URL features
4. Linear SVM on the same standardized lexical features

An isolated fifth **hybrid CNN** now fuses the character-CNN representation
with the same 28 standardized lexical features. It preserves every original
baseline and experimental split. See
[`docs/HYBRID_CNN_EXPERIMENT.md`](docs/HYBRID_CNN_EXPERIMENT.md).

URL generators:

1. **Gumbel-LSGAN** — selected as the stable, manageable adversarial generator.
2. **SeqGAN** — selected as the text-specific discrete-sequence GAN comparison.

A compact **RelGAN** candidate has now been trained, full-corpus filtered, and
screened at 10% augmentation with all four classifiers. It failed the seed-42
promotion rule and was not advanced to multi-seed confirmation. See
[`docs/RELGAN_MODEL.md`](docs/RELGAN_MODEL.md).

The reasoning is documented in [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

## Required experiment order

```text
prepare master dataset
        ↓
train four real-only baselines
        ↓
train shared malicious-URL language model
        ↓
fine-tune Gumbel-LSGAN and SeqGAN
        ↓
generate, measure, and rejection-filter URLs
        ↓
build paired augmented conditions from the unchanged real train split
        ↓
retrain classifiers with seeds 42/43/44
        ↓
summarize paired changes and confidence intervals
        ↓
open a separately frozen external dataset once
```

## Commands

Run commands from `Version_2`:

```bash
export PYTHONPATH=src
python3 -m phishing_gans_v2.prepare_data --config configs/project.json
python3 -m phishing_gans_v2.verify_data --config configs/project.json
python3 -m phishing_gans_v2.prepare_features --config configs/project.json

# Real-only controls (CNN/LSTM require CUDA; RF/SVM use scikit-learn CPU)
python3 -m phishing_gans_v2.train_baselines --config configs/project.json
python3 -m phishing_gans_v2.summarize_baselines

# Both generators require CUDA and share the same malicious train view
python3 -m phishing_gans_v2.pretrain_generator --config configs/project.json
python3 -m phishing_gans_v2.train_generators --model gumbel_lsgan
python3 -m phishing_gans_v2.train_generators --model seqgan
python3 -m phishing_gans_v2.generate_corpus --model gumbel_lsgan
python3 -m phishing_gans_v2.generate_corpus --model seqgan
python3 -m phishing_gans_v2.summarize_generators

# Materialize nested dose manifests and train one paired condition
python3 -m phishing_gans_v2.build_augmented_sets --model gumbel_lsgan
python3 -m phishing_gans_v2.build_augmented_sets --model seqgan
python3 -m phishing_gans_v2.train_baselines \
  --augmentation-manifest data/augmented/gumbel_lsgan/generated_10_percent.json
python3 -m phishing_gans_v2.summarize_augmentation

# Completed low-data protocol (commands skip existing artifacts when resumed)
python3 -m phishing_gans_v2.build_low_data_experiment
python3 -m phishing_gans_v2.run_low_data_matrix
python3 -m phishing_gans_v2.run_low_data_confirmation
python3 -m phishing_gans_v2.summarize_low_data

# RelGAN architecture test and CUDA training
python3 -m unittest discover -s tests -p 'test_relgan.py' -v
python3 -m phishing_gans_v2.train_relgan
python3 -m phishing_gans_v2.summarize_relgan
python3 -m phishing_gans_v2.generate_corpus --model relgan \
  --batch-size 64 --commit-interval 20000 --max-gpu-temperature 84
python3 -m phishing_gans_v2.build_augmented_sets --model relgan
python3 -m phishing_gans_v2.run_experiment_matrix \
  --generators relgan --doses 10 --models cnn,lstm,random_forest,linear_svm \
  --seeds 42 --include-augmentation
python3 -m phishing_gans_v2.summarize_relgan_augmentation

# Directly challenge frozen real-only classifiers with 100k URLs from each GAN
python3 -m phishing_gans_v2.evaluate_generated_challenge --sample-count 100000

# Controlled synthetic quality-selection experiment
python3 -m phishing_gans_v2.build_quality_experiment
python3 -m phishing_gans_v2.train_baselines \
  --augmentation-manifest data/augmented/seqgan_quality/realism_filtered_10_percent.json
python3 -m phishing_gans_v2.train_baselines \
  --augmentation-manifest data/augmented/seqgan_quality/realism_challenging_10_percent.json
python3 -m phishing_gans_v2.summarize_quality_experiment
python3 -m phishing_gans_v2.plot_quality_experiment

# Isolated real-only character/lexical hybrid (CUDA)
python3 -m phishing_gans_v2.train_hybrid_cnn --seeds 42,43,44
python3 -m phishing_gans_v2.summarize_hybrid_cnn
python3 -m phishing_gans_v2.evaluate_hybrid_external

# Diagnostic attribution of external distribution shift
python3 -m phishing_gans_v2.analyze_distribution_shift
python3 -m phishing_gans_v2.evaluate_external_canonicalization_ablation

# Freeze and evaluate a new PhishTank / local Top-Million snapshot
python3 -m phishing_gans_v2.collect_phishtank_topmillion \
  --phishtank-gzip data/external/2026-08-11/raw/phishtank_online_valid.csv.gz
python3 -m phishing_gans_v2.evaluate_phishtank_topmillion
```

The first command is a one-time freeze operation and now refuses to overwrite
the prepared split. Use the verification command for normal reruns.

Run only one heavy laptop workload at a time. The commands are deliberately
separate so interrupted work resumes at the next artifact instead of rerunning
the whole study.

Neural training commands will refuse to run without CUDA. Traditional RF/SVM
training remains on CPU because scikit-learn provides the simplest reproducible
implementation.

## Results

[`docs/RESULTS.md`](docs/RESULTS.md) is the authoritative human-readable results
record. It distinguishes completed results from planned runs and links every
claim to machine-readable JSON/CSV artifacts.

The direct answer to whether generated URLs evade the original classifiers is
in [`docs/GENERATED_CHALLENGE_EVALUATION.md`](docs/GENERATED_CHALLENGE_EVALUATION.md).

The controlled test of random versus realism-filtered and challenging synthetic
selection is in [`docs/QUALITY_SELECTION_EXPERIMENT.md`](docs/QUALITY_SELECTION_EXPERIMENT.md).

The later prospective test on PhishTank and local Top-Million URLs is in
[`docs/PHISHTANK_TOPMILLION_EVALUATION.md`](docs/PHISHTANK_TOPMILLION_EVALUATION.md).

The feature-, source-, GAN-, and canonicalization-level root-cause analysis is
in [`docs/DISTRIBUTION_SHIFT_ROOT_CAUSE.md`](docs/DISTRIBUTION_SHIFT_ROOT_CAUSE.md).

For a single professor-facing narrative covering the dataset, implementation,
model reasoning, verified metrics, current conclusion, and next iteration, read
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Historical project

All earlier work is preserved at `../Version_1/`. Version 1 metrics must never
be presented as Version 2 results.
