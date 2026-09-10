# Frozen-Classifier Challenge Evaluation

## Question

Do the real-only classifiers recognize URLs produced by the three GANs, and are
the URLs that evade them plausible enough to be useful augmentation candidates?

This is different from retraining a classifier with synthetic data and then
measuring accuracy on the unchanged real test set. A reduction in that accuracy
can reflect class-prior distortion, synthetic noise, or distribution mismatch;
by itself it does not show that the original classifier was fooled by generated
URLs.

## Protocol

- Take the first 100,000 strictly accepted, unique URLs from each frozen
  Gumbel-LSGAN, SeqGAN, and RelGAN pool (300,000 generated URLs total).
- Do not retrain or fine-tune a classifier.
- Evaluate every URL with the real-only CNN, LSTM, Random Forest, and linear SVM
  checkpoints for seeds 42, 43, and 44.
- Reuse each checkpoint's validation-selected decision threshold unchanged.
- Treat a generated URL as *intended phishing*. Report detection rate (recall)
  and false-negative/evasion rate. Accuracy, precision, FPR, and ROC-AUC are not
  defined for this one-class challenge set.
- Also measure consensus evasion, URL/TLD-like structure, lexical inlier status,
  and standardized lexical nearest-neighbor distance to real malicious training
  URLs. No test labels or test performance are used to select samples.

All 100,000 selected URLs per GAN are unique because they come from the strictly
accepted, deduplicated pools, which also have zero exact overlap with real data.

## Direct classifier results

Values are means across seeds 42, 43, and 44. Detection and evasion sum to 100%.

| Generator | Classifier | Detection/recall | Evasion/FNR |
|---|---|---:|---:|
| Gumbel-LSGAN | CNN | 93.68% | 6.32% |
| Gumbel-LSGAN | LSTM | 93.47% | 6.54% |
| Gumbel-LSGAN | Random Forest | 92.53% | 7.47% |
| Gumbel-LSGAN | Linear SVM | 90.69% | 9.31% |
| SeqGAN | CNN | 90.54% | 9.46% |
| SeqGAN | LSTM | 90.45% | 9.55% |
| SeqGAN | Random Forest | 88.87% | 11.13% |
| SeqGAN | Linear SVM | 87.06% | 12.94% |
| RelGAN | CNN | 88.90% | 11.10% |
| RelGAN | LSTM | 88.63% | 11.37% |
| RelGAN | Random Forest | 87.31% | 12.69% |
| RelGAN | Linear SVM | 84.66% | 15.34% |

The real-only classifiers therefore do **not** simply fail on GAN output: they
detect 84.66%–93.68% of it, depending on generator and classifier. However,
6.32%–15.34% is assigned the legitimate label, so the generated pools contain a
meaningful challenge subset. RelGAN is hardest and Gumbel-LSGAN is easiest.

## Consensus and realism checks

Each URL receives 12 votes (four classifier families times three seeds).
Majority evasion means at least seven of the 12 checkpoints predicted
legitimate. “Plausibly hard” additionally requires the URL to be TLD-like and
within five standard deviations of every real-malicious lexical feature.

| Generator | Majority evasion | Unanimous evasion | TLD-like | Lexical inlier | Plausibly hard | Mean lexical NN distance |
|---|---:|---:|---:|---:|---:|---:|
| Gumbel-LSGAN | 7.38% | 3.80% | 99.74% | 96.15% | 4.73% | 0.237 |
| SeqGAN | 10.97% | 5.64% | 99.75% | 94.69% | 6.89% | 0.371 |
| RelGAN | 12.35% | 5.76% | 94.21% | 93.82% | 7.64% | 1.435 |

The real-to-real reference mean nearest-neighbor distance is 0.265. This
distance is a diagnostic in standardized 28-feature space, not a semantic
similarity score. Gumbel-LSGAN is closest to the real lexical reference, SeqGAN
is moderately farther away, and RelGAN is much farther away.

RelGAN's higher evasion must therefore be interpreted cautiously. Its largest
overall standardized shift is `double_slash_count` (+0.91), and its hard subset
has an especially large shift on that feature (+6.67). Some of its apparent
challenge is likely caused by abnormal URL structure rather than realistic new
phishing behavior. SeqGAN offers a more balanced result: more challenge than
Gumbel-LSGAN while remaining much closer to real phishing than RelGAN.

## Relation to the reference paper

The referenced GAN evaluation paper compares classifiers trained on original
data with classifiers trained on original-plus-generated data and interprets a
lower accuracy on real test URLs as stronger GAN “fooling.” That experiment
measures the downstream effect of adding synthetic rows, but it does not
directly test whether a frozen original classifier misclassifies the generated
URLs. It can conflate generator evasion with augmentation-induced distribution
shift.

Version 2 retains the existing augmentation comparisons but adds this direct,
frozen-classifier challenge test. The two results are compatible: naive
augmentation can slightly degrade real-test metrics even though most generated
URLs are easy for the original models and a smaller subset genuinely crosses
their decision boundaries.

## Conclusion

Lower accuracy after augmentation does **not** mean that classifiers are
failing to classify the GAN. The direct experiment shows partial evasion, not
total failure. RelGAN yields the most evasive samples but also the largest
realism mismatch. The best next experiment is therefore to use the
*plausibly-hard* subset as a predeclared synthetic selection rule and evaluate
it against the same frozen real-only controls—without changing doses or dataset
sizes merely to obtain a favorable result.

Machine-readable evidence and figures are in
`artifacts/results/generated_challenge/`. The experiment is reproducible with:

```bash
export PYTHONPATH=src
python3 -m phishing_gans_v2.evaluate_generated_challenge --sample-count 100000
```
