# PhishTank / Top-Million External Evaluation

This prospectively frozen snapshot contains 250 PhishTank verified-online phishing URLs and 250 Top-Million legitimate root proxies.

Every URL, 128-token encoding, and hostname is unique and has zero overlap with the Version 2 master data or the prior external snapshot.

The primary results use each checkpoint's frozen internal-validation threshold. No model, threshold, or selection rule was fitted to this snapshot.

## CNN

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real only | 0.588000 | 0.728070 | 0.350667 | 0.460583 | 0.562896 | 0.174667 | 0.649333 |
| Random 10% | 0.615333 | 0.772948 | 0.353333 | 0.478458 | 0.576389 | 0.122667 | 0.646667 |
| Realism-filtered 10% | 0.627333 | 0.843939 | 0.325333 | 0.465673 | 0.580349 | 0.070667 | 0.674667 |
| Filtered + challenging 10% | 0.572000 | 0.695891 | 0.377333 | 0.464526 | 0.562381 | 0.233333 | 0.622667 |

## LSTM

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real only | 0.495333 | 0.497466 | 0.470667 | 0.482461 | 0.535787 | 0.480000 | 0.529333 |
| Random 10% | 0.469333 | 0.477256 | 0.636000 | 0.539081 | 0.510160 | 0.697333 | 0.364000 |
| Realism-filtered 10% | 0.458000 | 0.463431 | 0.525333 | 0.492104 | 0.503616 | 0.609333 | 0.474667 |
| Filtered + challenging 10% | 0.569333 | 0.702424 | 0.256000 | 0.361919 | 0.547869 | 0.117333 | 0.744000 |

## Random Forest

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real only | 0.540667 | 0.972222 | 0.084000 | 0.154538 | 0.424997 | 0.002667 | 0.916000 |
| Random 10% | 0.531333 | 0.924837 | 0.069333 | 0.128712 | 0.397616 | 0.006667 | 0.930667 |
| Realism-filtered 10% | 0.518000 | 1.000000 | 0.036000 | 0.069421 | 0.392224 | 0.000000 | 0.964000 |
| Filtered + challenging 10% | 0.506000 | 1.000000 | 0.012000 | 0.023695 | 0.313904 | 0.000000 | 0.988000 |

## Linear SVM

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real only | 0.444000 | 0.470339 | 0.888000 | 0.614958 | 0.370256 | 1.000000 | 0.112000 |
| Random 10% | 0.456000 | 0.476987 | 0.912000 | 0.626374 | 0.350688 | 1.000000 | 0.088000 |
| Realism-filtered 10% | 0.464000 | 0.481328 | 0.928000 | 0.633880 | 0.344832 | 1.000000 | 0.072000 |
| Filtered + challenging 10% | 0.434000 | 0.464668 | 0.868000 | 0.605300 | 0.302944 | 1.000000 | 0.132000 |

## Limitations

- Top-Million membership is a legitimate proxy, not a manual safety audit.
- Root-domain legitimate URLs are easier than arbitrary benign deep links.
- This balanced 500-row source-specific snapshot is not a population-wide prevalence or deployment estimate.
- Three identical SVM runs reflect deterministic convergence, not independent stochastic replications.
