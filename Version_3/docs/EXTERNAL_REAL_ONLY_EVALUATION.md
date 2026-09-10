# External Real-Only Evaluation — 2026-08-11

## Frozen protocol

The external snapshot contains 500 rows:

- 250 newly sampled legitimate root-domain proxies from the local Top-Domain ranked list.
- 250 PhishTank URLs that were verified and online when the official hourly feed was downloaded.
- Selected PhishTank verification times range from 2026-08-07 23:32 UTC to 2026-08-11 12:12 UTC.
- All 500 normalized URLs, 128-token encodings, and hostnames are unique.
- There is zero URL, encoded-input, or hostname overlap with the Version 3 master and zero overlap with previous external snapshots.

The models and their thresholds were frozen before this dataset was evaluated. No external threshold calibration, retraining, or result-driven sample selection was performed.

## Results

Mean ± sample standard deviation across seeds 42/43/44:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.4900 ± 0.0020 | 0.4949 ± 0.0010 | **0.9667 ± 0.0061** | **0.6546 ± 0.0014** | 0.6580 ± 0.0256 | 0.9867 ± 0.0083 | **0.0333 ± 0.0061** |
| LSTM | 0.4860 ± 0.0035 | 0.4928 ± 0.0017 | 0.9587 ± 0.0092 | 0.6510 ± 0.0019 | **0.8431 ± 0.0421** | 0.9867 ± 0.0140 | 0.0413 ± 0.0092 |
| RNN | 0.4807 ± 0.0012 | 0.4901 ± 0.0006 | 0.9600 ± 0.0000 | 0.6489 ± 0.0005 | 0.7821 ± 0.0157 | 0.9987 ± 0.0023 | 0.0400 ± 0.0000 |
| Random Forest | 0.3080 ± 0.0020 | 0.3657 ± 0.0014 | 0.5227 ± 0.0023 | 0.4303 ± 0.0017 | 0.2741 ± 0.0021 | 0.9067 ± 0.0023 | 0.4773 ± 0.0023 |
| Linear SVM | 0.3700 ± 0.0000 | 0.3184 ± 0.0000 | 0.2280 ± 0.0000 | 0.2657 ± 0.0000 | 0.2849 ± 0.0000 | **0.4880 ± 0.0000** | 0.7720 ± 0.0000 |

## Interpretation

The external experiment fails deployment requirements. The neural classifiers recognize most current PhishTank URLs, but almost every unseen Top-Domain root is falsely labelled phishing. Their high recall therefore does not compensate for an unusable 98.7–99.9% false-positive rate.

LSTM retains the best external ranking ability at ROC-AUC 0.8431. Its approximately 98.7% FPR at the frozen internal threshold shows threshold and score-distribution shift: useful relative ranking remains, but internal score calibration does not transfer.

RF and SVM have ROC-AUC below 0.5, indicating that important lexical relationships reverse between the internal and external sources. This is not fixed by changing a threshold alone.

## Structural explanation

| Source/class | URL length | Path length | Host dots | Estimated subdomains |
|---|---:|---:|---:|---:|
| Internal train legitimate | 50.69 | 28.18 | 2.29 | 1.29 |
| Internal train phishing | 55.74 | 26.12 | 1.58 | 0.58 |
| External Top-Domain legitimate | 15.61 | 1.00 | 1.14 | 0.14 |
| External PhishTank phishing | 34.90 | 6.49 | 1.87 | 0.87 |

The external legitimate roots are far shorter and structurally different from the legitimate training URLs. The hostname-dot relationship also reverses: training legitimate URLs contain more dots than training phishing URLs, whereas the external phishing URLs contain more dots than the Top-Domain roots. This supports the earlier warning that Version 3 still contains source/structure signals after scheme removal.

## What this result does and does not prove

It proves that the current real-only models do not transfer safely to previously unseen root-domain traffic at their frozen operating thresholds. It does not establish population-wide accuracy because Top-Domain membership is only a legitimacy proxy, benign examples are root URLs, the dataset is balanced, and PhishTank is a single phishing source.

The next defensible action is to collect labelled legitimate URLs containing both root and deep-link traffic from an independent source, then redesign the real training distribution to cover those structures. The current external set must remain frozen and must not become training or threshold-tuning data.
