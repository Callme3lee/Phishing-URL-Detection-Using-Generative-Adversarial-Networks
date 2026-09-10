# Controlled Synthetic Quality-Selection Experiment

## Research question

Does selecting realistic and classifier-challenging synthetic phishing URLs
produce a different result from randomly selecting the same number of URLs?

This experiment does not redesign the pipeline. It keeps the real data, frozen
splits, four classifier families, seeds 42/43/44, training code, synthetic dose,
and evaluation data unchanged.

## Four conditions

| Condition | Real rows | Synthetic rows | Synthetic source |
|---|---:|---:|---|
| Real-only | 2,055,821 | 0 | None |
| Existing random 10% | 2,055,821 | 228,425 | SeqGAN |
| Realism-filtered random 10% | 2,055,821 | 228,425 | SeqGAN |
| Realism-filtered + challenging 10% | 2,055,821 | 228,425 | SeqGAN |

Using SeqGAN exclusively in every augmented condition prevents generator
identity from becoming a hidden confound. The realism-filtered and challenging
conditions share the same realism-eligible source population and diversity cap;
only random versus difficulty-aware selection changes.

## Frozen selection rule

The rule was frozen before any new augmented model was evaluated. Its SHA-256
is `bfd43c48da8f7ea866c6eb2c3cf7a06ff73196f2d466dd44d08f9268635b78a7`.

Realism requires:

- membership in the existing strictly accepted SeqGAN pool;
- a TLD-like hostname;
- maximum absolute lexical-feature z-score no greater than 5, using only real
  malicious training statistics; and
- no upper structural outlier beyond the real-malicious 99.5th percentile for
  URL/hostname length, subdomains, double slashes, double dots, slashes,
  percent signs, or at signs.

The combined realism pass rate is 94.50%, leaving 1,200,643 non-reserved
eligible training candidates.

Difficulty is calculated from all 12 frozen real-only checkpoints. Raw scores
are never combined because CNN/LSTM probabilities, RF probabilities, and SVM
decision scores have different scales. Each checkpoint score is converted to
an empirical percentile within the eligible non-reserved pool. The frozen
ranking combines mean low-score difficulty (50%), evasion-vote agreement (30%),
and normalized boundary proximity (20%).

Both new pools represent all 64 MiniBatchKMeans lexical clusters, and no cluster
contributes more than 3%. The filtered-random pool has 10.39% majority evasion;
the challenging pool has 37.83%. Both remain exactly 228,425 unique rows.

## Untouched synthetic challenge

A 100,000-row tail outside the existing random 10% prefix was reserved before
augmentation selection. No URL in this reserve appears in any of the three
augmentation conditions. A diverse 20,000-row challenging subset was frozen for
evaluation only. It contains intended-phishing examples only, so it supports
recall and FNR—not accuracy, precision, FPR, or ROC-AUC.

## Threshold protocols

The primary comparison follows the requested common-threshold rule: every
condition uses the matching real-only model/seed threshold. This holds the
numeric decision boundary constant but can expose score-calibration drift after
retraining. Therefore, the report also retains the established secondary
protocol in which every trained condition uses its own validation-selected
threshold under the same validation-FPR constraint. ROC-AUC is independent of
both threshold choices.

## Primary results: common real-only thresholds

Values below are means across seeds. Deltas are percentage-point changes from
the existing random SeqGAN 10% condition.

| Model | Selection | Recall Δ | F1 Δ | ROC-AUC Δ | FPR Δ | Challenge recall Δ |
|---|---|---:|---:|---:|---:|---:|
| CNN | Filtered | +0.52 | +0.14 | +0.00 | +0.38 | +2.88 |
| CNN | Challenging | +0.66 | +0.05 | -0.07 | +0.81 | **+11.05** |
| LSTM | Filtered | +0.63 | +0.17 | -0.01 | +0.45 | +3.64 |
| LSTM | Challenging | +0.63 | +0.09 | -0.05 | +0.66 | **+12.01** |
| Random Forest | Filtered | +0.18 | +0.01 | +0.01 | +0.24 | +2.18 |
| Random Forest | Challenging | +0.82 | -0.06 | -0.07 | +1.35 | **+9.23** |
| Linear SVM | Filtered | +1.58 | +0.82 | +0.10 | +0.17 | +4.72 |
| Linear SVM | Challenging | +1.64 | +0.72 | +0.12 | +0.53 | **+6.02** |

The challenging pool consistently improves held-out synthetic recall. The
paired improvement versus random augmentation is +11.05 points for CNN (95% CI
+3.83 to +18.27) and +9.23 points for RF (+9.09 to +9.38). The LSTM mean is
+12.01 points, but its three-seed interval (-0.92 to +24.95) crosses zero due to
seed variability. SVM is deterministic on these inputs, so its identical-seed
result is not three independent stochastic replications.

The real-test effect is not a free improvement. Challenging selection increases
recall but also increases FPR and lowers precision for every model. CNN and RF
ROC-AUC fall by about 0.07 points. RF challenging selection raises FPR by 1.35
points and lowers F1 by 0.06 points relative to random augmentation.

## Secondary results: condition validation thresholds

When every condition uses its validation-selected threshold to keep FPR near
1%, the mean real-test F1 scores are:

| Model | Real-only | Random 10% | Filtered 10% | Challenging 10% |
|---|---:|---:|---:|---:|
| CNN | **96.721%** | 96.620% | 96.577% | 96.342% |
| LSTM | **96.809%** | 96.808% | 96.687% | 96.470% |
| Random Forest | **95.510%** | 95.462% | 95.433% | 95.237% |
| Linear SVM | **93.648%** | 92.948% | 93.519% | 92.630% |

This confirms that much of the common-threshold recall/F1 gain comes from a
shift toward predicting more URLs as phishing. The real-only model remains best
by validation-threshold F1 for every classifier.

## Conclusion

Synthetic sample quality and difficulty **do matter**, but the effect depends
on what is measured:

- realism filtering generally improves common-threshold performance over naive
  random augmentation;
- classifier-challenging selection strongly improves robustness to a disjoint
  generated challenge;
- the robustness gain comes with higher real-test false positives, lower
  precision, and usually lower ROC-AUC; and
- under the established validation-FPR threshold policy, neither selected pool
  beats the real-only F1 baseline.

Therefore this experiment supports a robustness trade-off, not the claim that
quality-selected GAN augmentation is universally better. For the current
browser-extension goal, the real-only LSTM remains the safest primary detector.
The challenging model is useful as an experimental robustness model, not a
replacement production model.

## Reproducibility

```bash
export PYTHONPATH=src
python3 -m phishing_gans_v2.build_quality_experiment
python3 -m phishing_gans_v2.train_baselines \
  --augmentation-manifest data/augmented/seqgan_quality/realism_filtered_10_percent.json
python3 -m phishing_gans_v2.train_baselines \
  --augmentation-manifest data/augmented/seqgan_quality/realism_challenging_10_percent.json
python3 -m phishing_gans_v2.summarize_quality_experiment
python3 -m phishing_gans_v2.plot_quality_experiment
```

Machine-readable results and all PNG/PDF figures are in
`artifacts/results/quality_experiment/`.
