# Project result graphs

This folder brings the project's result figures together for supervisor and examiner discussions. The original files remain in their experiment folders so report links and reproducibility are not affected.

## Folder contents

- `00_Supervisor_Selection/` contains 19 clearly numbered figures covering the main research story.
- `Complete_Archive/` contains all 189 result images found in Versions 1--3, the final evaluations, the report, and user-evaluation folders. When a PDF counterpart exists, it is included as well.
- The directory structure below `Complete_Archive/` preserves every original source path. This prevents unrelated graphs named `comparison.png` from being confused with one another.

The complete archive contains 189 PNG figures and 185 PDF figures or result documents. Screenshots, logos, decorative images, and browser-interface previews are excluded.

## Recommended discussion order

| No. | Figure | Main purpose |
|---:|---|---|
| 1 | `01_v3_internal_baselines.png` | Compare the Version 3 real-only classifiers on the controlled internal evaluation. |
| 2 | `02_gan_quality_comparison.png` | Compare the quality and diversity measurements of the investigated generators. |
| 3 | `03_seqgan_internal_metrics.png` | Show the earlier SeqGAN augmentation behavior. |
| 4 | `04_ms_hardgan_internal_metrics.pdf` | Show the controlled MS-HardGAN-C augmentation results. |
| 5 | `05_ms_hardgan_challenge_recall.pdf` | Show behavior on the synthetic challenge evaluation. |
| 6 | `06_matched_gan_hybrid_comparison.png` | Compare SeqGAN and MS-HardGAN-C under the matched hybrid experiment. |
| 7 | `07_external_classifier_comparison.png` | Explain why the hybrid was selected instead of presenting it without baseline evidence. |
| 8 | `08_final_30k_all_classifiers.png` | Compare classifiers on the frozen low-prevalence 30K evaluation. |
| 9 | `09_real_only_vs_gan10_external.png` | Present the main controlled comparison. Grey dashed squares are Real-only; blue solid circles are GAN +10%. FNR and FPR are separated into different panels. |
| 10 | `10_threshold_tradeoff.png` | Explain that reducing false positives and false negatives requires an operating-point choice. |
| 11 | `11_extension_fpr_vs_fnr.png` | Show the extension-specific FPR/FNR threshold frontier. |
| 12 | `12_model_score_distributions.png` | Show how legitimate and phishing score overlap limits threshold-only improvement. |
| 13 | `13_model_vs_extension.png` | Separate native-model results from extension-policy results. |
| 14 | `14_user_false_reports.png` | Summarise participant-reported false warnings by user and category. |
| 15 | `15_training_url_length_distribution.png` | Describe the URL-length composition of the classifier training data. |
| 16 | `16_complete_2m_model_comparison.png` | Show the controlled evaluation against the Version 1 2M dataset. |
| 17 | `17_online_3pct_model_comparison.png` | Show the later online low-prevalence evaluation. |
| 18 | `18_version1_summary.png` | Summarise the main Version 1 result. |
| 19 | `19_version2_baselines.png` | Summarise the Version 2 baseline comparison. |

For a short presentation, use figures **1, 2, 6, 7, 9, 10, 13, and 14**. Figures 15--19 provide supporting evidence when questions arise about dataset composition or earlier project versions.

## Interpretation note

Do not compare graph heights across different datasets as though they came from one test. Versions 1, 2, and 3 and the internal, external, 2M, 30K, and online evaluations use different samples or protocols. Comparisons are strongest when the models were evaluated on the same frozen rows, with the same labels, metrics, and threshold-selection procedure. The report explains the relevant protocol for each final claim.
