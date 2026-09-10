# Calibrated Hybrid Ensemble

## The project in plain language

Phishing websites often use deceptive URLs: misspelled brands, unusual subdomains, long paths, random characters, fake login terms, IP addresses, or misleading query parameters. This project tries to detect those attacks using only the URL text. It does not download or render the webpage.

Each URL has a label:

- `0`: legitimate;
- `1`: phishing.

A classifier reads examples with known labels and learns to return a phishing score. Metrics describe different kinds of correctness:

- **Accuracy:** fraction of all decisions that are correct.
- **Precision:** when the model says phishing, how often it is right.
- **Recall:** how many real phishing URLs it catches.
- **F1:** a balance of precision and recall.
- **ROC-AUC:** how well scores rank phishing above legitimate URLs across every possible threshold.
- **FPR:** legitimate URLs incorrectly blocked as phishing.
- **FNR:** phishing URLs incorrectly allowed.

For a browser extension, FPR is especially important. A detector that blocks nearly every safe website is unusable even if its phishing recall is high.

## What went wrong originally

The first CNN, LSTM, RNN, Random Forest, and SVM models performed very well on the internal test split. On outside data, however, the neural models classified approximately 99% of legitimate domains as phishing. They had learned dataset/source patterns that did not transfer well.

Internal structure balancing did not solve the problem. Adding a small amount of outside legitimate coverage helped, but external FPR remained approximately 86–88% for the best LSTM condition.

## What the hybrid changed

The hybrid model combines two views of the same URL:

1. A character CNN learns useful character sequences automatically.
2. A lexical branch receives 35 numeric URL properties, including PSL-aware component lengths, character ratios, entropy, path depth, query count, IP/punycode flags, and suspicious-token ratios.

Source-balanced sampling prevents the large URLs_New groups from completely overwhelming smaller outside sources. Class-conditional CORAL encourages legitimate representations from different sources to resemble one another, and does the same separately for phishing sources.

Three models were trained with seeds 42, 43, and 44. For the deployment candidate, their three phishing scores are averaged. Averaging makes the decision less dependent on one random training run.

## Why calibration was needed

The model produces a continuous score, but the browser must make a yes/no decision. The threshold controls the trade-off:

- lower threshold: catch more phishing, but block more legitimate sites;
- higher threshold: block fewer legitimate sites, but miss more phishing.

The earlier thresholds either produced too many false alarms or missed too much phishing. A dedicated calibration set was therefore frozen after all model checkpoints were fixed.

## Calibration data contract

The frozen calibration dataset contains 400 rows:

| Class | Rows | Source |
|---|---:|---|
| Legitimate proxy | 200 | Tranco root domains |
| Phishing | 200 | PhishTank verified-online feed |

All 400 URLs, hostnames, and 128-character encodings are unique. Measured overlap is zero against:

- URLs_New and its frozen splits;
- hybrid training data;
- hybrid domain validation;
- legitimate coverage development data;
- both previous external datasets; and
- earlier Version 2 external identities.

The calibration set never retrains the model. It is used only to choose a threshold.

The newly downloaded OpenPhish and Tranco files were byte-identical to the archived development files. PhishTank had advanced only slightly. Therefore, this is honestly labelled a post-model, identity-disjoint calibration set—not a genuinely later temporal snapshot.

## Frozen threshold

The fixed policy was registered before scoring external labels:

> Select the ensemble threshold with maximum calibration F1, subject to calibration FPR being no more than 1%.

The resulting frozen threshold is:

`0.9879557490348816`

Calibration resubstitution metrics are:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9425 | 0.9890 | 0.8950 | 0.9396 | 0.9745 | 0.9769 | 0.0100 | 0.1050 |

These values are optimistic because the same labels selected the threshold. They are not presented as an unbiased test result.

## Exploratory outside results

The threshold was then applied unchanged to the older external sets:

| External set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-Domain / PhishTank, 500 rows | 0.9260 | 0.9908 | 0.8600 | 0.9208 | 0.9746 | 0.9821 | 0.0080 | 0.1400 |
| Tranco / OpenPhish, 400 rows | 0.8875 | 0.9755 | 0.7950 | 0.8760 | 0.9324 | 0.9567 | 0.0200 | 0.2050 |

This operating point is much more practical than either earlier hybrid threshold. External FPR is now 0.8–2.0%, compared with approximately 99% for the original LSTM. The cost is lower phishing recall: 79.5–86.0%, rather than roughly 91–97% at the less conservative threshold.

These results are still exploratory because both old external sets informed the hybrid design. They validate that calibration behaves sensibly; they do not replace a new confirmation test.

## What remains

A confirmation-readiness guard now prevents us from calling unchanged feeds “new.” At present it reports:

- OpenPhish changed: no;
- Tranco changed: no;
- at least 100 PhishTank URLs verified after calibration: no;
- ready for temporal confirmation: no.

When the feeds advance, download them into a new dated directory, rerun the readiness check, freeze a hostname-disjoint confirmation set, and evaluate the threshold exactly once. Do not change the model, features, ensemble, or threshold after seeing those labels.

## Reproduce

From `Version_3`:

```bash
export PYTHONPATH=src
python -m phishing_gans_v3.collect_hybrid_calibration
python -m phishing_gans_v3.calibrate_hybrid_threshold
python -m phishing_gans_v3.check_confirmation_readiness
```

The collection and calibration commands refuse to overwrite their frozen outputs.
