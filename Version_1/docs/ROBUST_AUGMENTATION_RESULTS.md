# Source-balanced, quality-filtered augmentation results

## What changed

The earlier experiment sampled real data only from the 2M source. This follow-up
uses real URLs from all three available sources and filters generated URLs before
augmentation. It tests whether the GAN signal survives a harder, leakage-aware,
multi-source comparison.

The real-only training set contains 90,000 URLs: 15,000 benign and 15,000
malicious URLs from each source. Validation contains 18,000 URLs and the balanced
test set contains 30,000 URLs. Hostnames are separated before sampling. Exact
encoded duplicates are excluded across sources and splits, including collisions
caused when distinct long URLs are truncated to the same 128-character tensor.

The quality filter ranks accepted generated URLs using character-language
likelihood, robust distance from the real malicious length distribution, and
within-URL character diversity. LSGAN doses of 10%, 20%, and 30% of the final
training set were compared with random and quality-filtered 30% combined-GAN
conditions. CNN, LSTM, Random Forest, and linear SVM were trained with seeds 42,
43, and 44. Neural models used the RTX 3060 GPU.

## Balanced held-out test results

The table gives the best mean-F1 condition for each classifier across three
seeds. Change is paired against the corresponding real-only run.

| Model | Best condition | Mean F1 | Change from real-only | Mean ROC-AUC |
|---|---|---:|---:|---:|
| CNN | Quality LSGAN 30% | **94.8105%** | **+0.1493 pp** | 98.2202% |
| LSTM | Quality LSGAN 10% | **94.4964%** | +0.0151 pp | 98.0872% |
| Random Forest | Real only | **93.5821%** | 0.0000 pp | 97.7073% |
| Linear SVM | Quality LSGAN 20% | **86.0279%** | **+0.7311 pp** | 91.9975% |

CNN improved in all three seeds. Its paired mean F1 gain was +0.1493 percentage
points, with a three-run 95% t interval from +0.0518 to +0.2467 points. The LSTM
gain is too small to treat as meaningful. SVM is deterministic under this setup,
so its repeated values verify reproducibility but do not estimate initialization
variance. Random Forest remains strongest without generated data.

## Hostname-held-out source results

Thresholds were calibrated only on validation hostnames from the matching source
and then frozen for the full source test bucket. These test sets are hostname
unseen, but the collection sources are now represented in training; they must not
be described as completely unseen-source evaluation.

| Model | Best source-macro F1 | Mean macro F1 | Best source-macro ROC-AUC | Mean macro ROC-AUC |
|---|---|---:|---|---:|
| CNN | Real only | **95.0526%** | Quality LSGAN 30% | **98.2027%** |
| LSTM | Real only | **94.3239%** | Random combined GAN 30% | **97.9897%** |
| Random Forest | Real only | **92.0866%** | Real only | **96.9485%** |
| Linear SVM | Real only | **86.0072%** | Quality LSGAN 10% | **89.3073%** |

Augmentation therefore improves ranking robustness for CNN, LSTM, and SVM, but
does not beat real-only source-macro F1 with one global training condition. It
helps `new_data_urls.csv` more than `URL dataset.csv`; increasing the synthetic
dose improves some new-data results while gradually reducing performance on the
other source.

## Main conclusion

The useful augmentation strategy is not “add as many unique GAN URLs as
possible.” The improved strategy is:

1. train with source-balanced real data;
2. remove encoded as well as raw-string collisions;
3. select realistic, diverse LSGAN samples;
4. tune the synthetic dose per classifier; and
5. calibrate thresholds on deployment-like validation data.

Under this protocol, quality-filtered LSGAN augmentation beats real-only F1 for
CNN and SVM on the controlled balanced test and beats real-only source-macro
ROC-AUC for both models. Real-only remains the safest choice when a single model
must maximize source-macro F1.

## Two-stage CNN improvement

A completed follow-up initializes the CNN from the 10%, 20%, or 30%
quality-LSGAN checkpoint and fine-tunes it for two epochs on the shared real-only
corpus at learning rate `1e-4`. Across seeds 42, 43, and 44, the 20%-then-real
schedule gives the best balanced F1 of 94.8442% (+0.1830 percentage points) and
the best source-macro F1 of 95.0915% (+0.0389 points). The 30%-then-real schedule
gives the best source-macro ROC-AUC of 98.2076% (+0.2380 points).

The balanced F1 gain is repeatable across all three seeds and its paired 95%
t interval excludes zero. The smaller source-macro F1 interval includes zero,
so additional seeds or another unseen collection source are still required.
Complete details are in `docs/AUGMENTED_FINETUNING.md`.

## Reproducible files

- Dataset builder: `src/phishing_gans/prepare_robust_augmentation.py`
- Lower-dose builder: `src/phishing_gans/add_robust_lsgan_doses.py`
- Multi-seed runner support: `src/phishing_gans/run_augmented_experiments.py`
- Source calibration: `src/phishing_gans/evaluate_robust_calibrated.py`
- Summary generators: `src/phishing_gans/summarize_robust_augmentation.py` and
  `src/phishing_gans/summarize_robust_external.py`
- Machine results: `artifacts/robust_augmentation/three_seed_summary.json` and
  `artifacts/robust_augmentation/calibrated_external_three_seed_summary.json`
- Figures: `artifacts/robust_augmentation/three_seed_f1.png` and
  `artifacts/robust_augmentation/calibrated_external_three_seed.png`
