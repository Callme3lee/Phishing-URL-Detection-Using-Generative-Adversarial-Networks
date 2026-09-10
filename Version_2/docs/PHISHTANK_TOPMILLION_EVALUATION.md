# PhishTank / Top-Million Prospective Evaluation

## Purpose

This is a small, evaluation-only test of whether the already-trained Version 2
classifiers generalize to a different, later-collected source. No model was
retrained and no threshold was adjusted from these results.

The snapshot was frozen on 2026-08-11 and contains:

- 250 verified-online phishing URLs from the official PhishTank feed; and
- 250 root URLs sampled reproducibly from ranks 1–1,000,000 of the local
  `top10milliondomains.csv` file as legitimate proxies.

All 500 URLs, 128-token encodings, and canonical hostnames are unique. They
have zero exact, encoded, or hostname overlap with the Version 2 master dataset
and zero URL/hostname overlap with the earlier external snapshot. The frozen
CSV SHA-256 is
`8c0727149427aed39a3cc9afb13af032266a9c57edc4e8c4a6b5bc1871665705`.

## Evaluation contract

Every CNN, LSTM, Random Forest, and linear SVM checkpoint for seeds 42, 43,
and 44 was tested in four existing conditions:

1. real-only;
2. random SeqGAN 10%;
3. realism-filtered SeqGAN 10%; and
4. realism-filtered plus classifier-challenging SeqGAN 10%.

The primary metrics below use each checkpoint's previously selected internal
validation threshold. Values are means across the three seeds. FPR is the
fraction of legitimate URLs incorrectly flagged; FNR is the fraction of
phishing URLs missed.

## Results

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | Real only | 0.588 | 0.728 | 0.351 | 0.461 | 0.563 | 0.175 | 0.649 |
| CNN | Random 10% | 0.615 | 0.773 | 0.353 | 0.478 | 0.576 | 0.123 | 0.647 |
| CNN | Realism-filtered 10% | **0.627** | **0.844** | 0.325 | 0.466 | **0.580** | **0.071** | 0.675 |
| CNN | Filtered + challenging 10% | 0.572 | 0.696 | **0.377** | 0.465 | 0.562 | 0.233 | **0.623** |
| LSTM | Real only | 0.495 | 0.497 | 0.471 | 0.482 | 0.536 | 0.480 | 0.529 |
| LSTM | Random 10% | 0.469 | 0.477 | **0.636** | **0.539** | 0.510 | 0.697 | **0.364** |
| LSTM | Realism-filtered 10% | 0.458 | 0.463 | 0.525 | 0.492 | 0.504 | 0.609 | 0.475 |
| LSTM | Filtered + challenging 10% | **0.569** | **0.702** | 0.256 | 0.362 | **0.548** | **0.117** | 0.744 |
| Random Forest | Real only | **0.541** | 0.972 | **0.084** | **0.155** | **0.425** | 0.003 | **0.916** |
| Random Forest | Random 10% | 0.531 | 0.925 | 0.069 | 0.129 | 0.398 | 0.007 | 0.931 |
| Random Forest | Realism-filtered 10% | 0.518 | **1.000** | 0.036 | 0.069 | 0.392 | **0.000** | 0.964 |
| Random Forest | Filtered + challenging 10% | 0.506 | **1.000** | 0.012 | 0.024 | 0.314 | **0.000** | 0.988 |
| Linear SVM | Real only | 0.444 | 0.470 | 0.888 | 0.615 | **0.370** | 1.000 | 0.112 |
| Linear SVM | Random 10% | 0.456 | 0.477 | 0.912 | 0.626 | 0.351 | 1.000 | 0.088 |
| Linear SVM | Realism-filtered 10% | **0.464** | **0.481** | **0.928** | **0.634** | 0.345 | 1.000 | **0.072** |
| Linear SVM | Filtered + challenging 10% | 0.434 | 0.465 | 0.868 | 0.605 | 0.303 | 1.000 | 0.132 |

## Interpretation

No condition is reliable enough on this snapshot. The best overall ranking
signal is CNN realism-filtered 10% (ROC-AUC 0.580), but it still misses 67.5%
of the current PhishTank URLs. The challenging CNN condition increases recall
to 37.7%, but also increases FPR to 23.3%.

The SVM's apparent recall and F1 gains are not useful: every SVM condition has
FPR 1.000, meaning every legitimate proxy was flagged. Random Forest has the
opposite failure mode: it almost never flags legitimate URLs, but misses
91.6–98.8% of phishing URLs. LSTM random 10% raises recall to 63.6%, but its
69.7% FPR makes the trade-off unacceptable.

CNN augmentation shows small favorable changes for some metrics, but only
three seeds and 500 source-specific rows were tested. The changes are mixed
across metrics and do not establish robust superiority over real-only.

## Limitations

- Top-Million membership is a legitimacy proxy, not a manual safety audit.
- Legitimate root URLs are structurally easier and different from arbitrary
  benign deep links.
- PhishTank's verified-online feed and a popularity list represent different
  acquisition processes; source artifacts may influence the scores.
- This balanced dataset measures discrimination under a controlled class mix,
  not browser-extension precision at real-world phishing prevalence.
- The snapshot must remain evaluation-only. Using it for retraining or
  threshold tuning would invalidate its prospective role.

Machine-readable results, per-URL predictions, mean/SD tables, and graphs are
in `artifacts/results/external/2026-08-11_phishtank_topmillion/`.
