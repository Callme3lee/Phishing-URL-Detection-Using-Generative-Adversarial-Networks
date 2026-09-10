# Version 3 — Scheme-Neutral URL Classification and GAN Study

Version 3 keeps `../URLs_New.csv` as the single frozen dataset for every original real-only baseline. A separately frozen, clearly labelled shift-control experiment adds overlap-excluded real sources to test a hybrid classifier; it does not change or replace the baseline split.

## Frozen data contract

- SHA-256: `18067bad500580a7def376c5a4438c9886adda5eb739fc209fdda4c6309966a7`
- Labels: `legitimate = 0`, `phishing = 1`
- Only HTTP(S) URLs are retained.
- URL schemes and user information are removed before encoding or feature extraction.
- Canonical URLs and 128-character model inputs are deduplicated.
- Ambiguous labels are quarantined.
- Hostnames, not individual rows, are deterministically assigned to train/validation/test.
- GANs may train only on phishing rows from the frozen training split.
- Version 2 contributes no training rows. Its database is consulted only to quarantine exact cross-version label disagreements.

## Current stage

The five original real-only baselines are complete for seeds 42/43/44. The strongest shift-control experiment is a character CNN + 35 PSL-aware lexical features + source-balanced conditional CORAL. A working Manifest V3 research extension now runs the frozen HYB-MS +10% three-seed ensemble locally through ONNX Runtime Web. It is an implementation proof, not deployment-ready: the later deployment-confirmation experiment failed and interactive cross-device browser validation remains outstanding.

See [docs/BROWSER_EXTENSION.md](docs/BROWSER_EXTENSION.md), [docs/DATASET.md](docs/DATASET.md), [docs/RESULTS.md](docs/RESULTS.md), [docs/EXTERNAL_REAL_ONLY_EVALUATION.md](docs/EXTERNAL_REAL_ONLY_EVALUATION.md), [docs/BENIGN_COVERAGE_DATASET.md](docs/BENIGN_COVERAGE_DATASET.md), [docs/LEGITIMATE_COVERAGE_EXPERIMENT.md](docs/LEGITIMATE_COVERAGE_EXPERIMENT.md), [docs/URLS_NEW_STRUCTURE_BALANCING.md](docs/URLS_NEW_STRUCTURE_BALANCING.md), [docs/HYBRID_CNN_CORAL_EXPERIMENT.md](docs/HYBRID_CNN_CORAL_EXPERIMENT.md), [docs/CALIBRATED_HYBRID_ENSEMBLE.md](docs/CALIBRATED_HYBRID_ENSEMBLE.md), [docs/PRE_GAN_EVALUATION.md](docs/PRE_GAN_EVALUATION.md), and [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

## Reproduce

Run from `Version_3`:

```bash
export PYTHONPATH=src

# One-time materialization. It refuses to overwrite an existing freeze.
python3 -m phishing_gans_v3.prepare_data

# Normal verification and feature preparation.
python3 -m phishing_gans_v3.verify_data
python3 -m phishing_gans_v3.prepare_features
python3 -m phishing_gans_v3.audit_shortcuts

# Current real-only baselines.
python3 -m phishing_gans_v3.train_traditional --seeds 42,43,44
python3 -m phishing_gans_v3.train_neural --models cnn,lstm,rnn --seeds 42,43,44
python3 -m phishing_gans_v3.summarize_baselines

# Unchanged baseline architectures on the frozen 363,692-row multi-source set.
python3 -m phishing_gans_v3.prepare_multisource_baselines
python3 -m phishing_gans_v3.train_traditional --data-dir data/experiments/hybrid_cnn_coral_v1/processed --feature-dir data/experiments/multisource_baselines_v1/features --output-dir artifacts/experiments/multisource_baselines_v1 --condition multisource_real --seeds 42,43,44
python3 -m phishing_gans_v3.train_neural --data-dir data/experiments/hybrid_cnn_coral_v1/processed --output-dir artifacts/experiments/multisource_baselines_v1 --condition multisource_real --models cnn,lstm,rnn --seeds 42,43,44
python3 -m phishing_gans_v3.evaluate_external --dataset data/external/2026-08-12-large-frozen/external_urls.csv --collection-report data/external/2026-08-12-large-frozen/collection_report.json --checkpoint-dir artifacts/experiments/multisource_baselines_v1 --output-dir artifacts/results/multisource_baselines_v1/external_large_2026-08-12
python3 -m phishing_gans_v3.summarize_multisource_baselines

# Frozen external collection/evaluation (do not overwrite or tune on it).
python3 -m phishing_gans_v3.collect_external_snapshot
python3 -m phishing_gans_v3.evaluate_external
python3 -m phishing_gans_v3.analyze_external_shift

# Fixed hybrid shift-control experiment (CUDA required for training/evaluation).
python3 -m phishing_gans_v3.prepare_hybrid_domain_data
python3 -m phishing_gans_v3.train_hybrid_domain
python3 -m phishing_gans_v3.compare_hybrid_external

# Post-model calibration and temporal-confirmation guard.
python3 -m phishing_gans_v3.collect_hybrid_calibration
python3 -m phishing_gans_v3.calibrate_hybrid_threshold
python3 -m phishing_gans_v3.check_confirmation_readiness

# Complete artifact and scientific-readiness audit before GAN implementation.
python3 -m phishing_gans_v3.evaluate_pre_gan_readiness

# Re-audit the already frozen legitimate development pool and rebuild its
# deterministic, Tranco-domain-grouped manifests. This does not retrain models.
python3 -m phishing_gans_v3.audit_benign_coverage

# Defensive multi-source hard-example GAN (CUDA required).
# This is isolated from the earlier SeqGAN, URL-GAN, and RelGAN artifacts.
python3 -m phishing_gans_v3.prepare_ms_hardgan
python3 -m phishing_gans_v3.train_ms_hardgan
python3 -m phishing_gans_v3.generate_ms_hardgan

# Tests.
python3 -m unittest discover -s tests -v

# Frozen browser implementation and portability gate (CPU is sufficient).
python3 -m phishing_gans_v3.validate_browser_bundle
```

Do not pass `--force` to the preparation command unless deliberately creating a new frozen experiment version.

The MS-HardGAN methodology and its explicit zero-day limitation are documented in `docs/MS_HARDGAN.md`.
