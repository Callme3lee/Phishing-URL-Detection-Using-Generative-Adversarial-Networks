# Project Status

| Stage | Status | Evidence |
|---|---|---|
| Version 1 archive | Complete | `../Version_1/ARCHIVE.md` |
| Master CSV copied and hashed | Complete | `configs/project.json` |
| Version 2 structure and protocol | Complete | `README.md`, `docs/` |
| Full data preparation | Complete and verified | `data/processed/preparation_report.json`, `verification_report.json` |
| Four real-only baselines | Complete for seeds 42/43/44 | `artifacts/baselines/`, `artifacts/results/baselines/` |
| Isolated hybrid CNN | Complete for seeds 42/43/44; best internal real-only accuracy/recall/F1/ROC-AUC, external shift remains | `docs/HYBRID_CNN_EXPERIMENT.md`, `artifacts/hybrid_cnn/`, `artifacts/results/hybrid_cnn/` |
| Shared generator pretraining | Complete | `artifacts/generators/pretrained/best.pt`, `history.json` |
| Gumbel-LSGAN | Complete | `artifacts/generators/gumbel_lsgan/`, `artifacts/generated/gumbel_lsgan/` |
| SeqGAN | Complete | `artifacts/generators/seqgan/`, `artifacts/generated/seqgan/` |
| Generated-corpus verification | Complete | `artifacts/results/generators/` |
| Nested augmented dataset manifests | Complete | `data/augmented/` |
| Gumbel-LSGAN 10% confirmation | Complete for all classifiers and seeds 42/43/44 | `artifacts/augmentation/`, `artifacts/results/augmentation/` |
| SeqGAN 10% confirmation | Complete for all classifiers and seeds 42/43/44 | `artifacts/augmentation/`, `artifacts/results/augmentation/` |
| Original-GAN 20/30/40% dose screening | Complete for Gumbel-LSGAN and SeqGAN, all classifiers, seed 42 | `artifacts/augmentation/`, `artifacts/results/augmentation/` |
| Organized comparison graphs | Complete | `artifacts/results/augmentation/dose_response/`, `delta_heatmaps/` |
| 10%-real low-data subset | Frozen and verified (210,434 complete hostname-grouped rows) | `data/low_data/real_10_percent/experiment.json` |
| Low-data weighted augmentation screen | Seed 42 complete for CNN/LSTM at 1%, 2%, 5%, 10%, and 15% | `docs/LOW_DATA_EXPERIMENT.md`, `artifacts/low_data/real_10_percent/results/` |
| Low-data confirmation | Complete for CNN-2% and LSTM-10%, paired seeds 42/43/44; no reliable improvement established | `docs/LOW_DATA_EXPERIMENT.md`, `artifacts/low_data/real_10_percent/results/` |
| RelGAN full corpus | Complete: 2,000,000 raw, 1,989,755 strictly accepted | `artifacts/generated/relgan/`, `artifacts/results/generators/` |
| RelGAN nested manifests | Complete for 10/20/30/40% | `data/augmented/relgan/` |
| RelGAN augmentation screen | 10%, seed 42 complete for all four classifiers; failed promotion rule, so no seed-43/44 confirmation | `artifacts/augmentation/relgan/`, `artifacts/results/augmentation/relgan_screen/` |
| External evaluation | Complete on frozen 500-URL hostname-disjoint snapshot | `artifacts/results/external/2026-08-09/` |
| Frozen real-only classifier challenge | Complete: 100,000 accepted URLs per GAN, all four classifiers and seeds 42/43/44 | `docs/GENERATED_CHALLENGE_EVALUATION.md`, `artifacts/results/generated_challenge/` |
| Controlled quality-selection experiment | Complete: four conditions, all classifiers/seeds, common and validation thresholds, disjoint 20k synthetic challenge | `docs/QUALITY_SELECTION_EXPERIMENT.md`, `artifacts/results/quality_experiment/` |
| Later PhishTank / Top-Million evaluation | Complete: frozen balanced 500-URL snapshot, all four conditions/classifiers/seeds, all requested metrics | `docs/PHISHTANK_TOPMILLION_EVALUATION.md`, `artifacts/results/external/2026-08-11_phishtank_topmillion/` |
| Distribution-shift root-cause analysis | Complete: real-data source/path shortcut is primary; GANs reproduce and can amplify it | `docs/DISTRIBUTION_SHIFT_ROOT_CAUSE.md`, `artifacts/results/distribution_shift/` |
