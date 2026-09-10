# External Snapshot Evaluation

Dataset: **500 URLs** (250 legitimate, 250 phishing).

Frozen internal-validation thresholds were used; no retraining or external tuning was performed.

| Model family | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Real-only LSTM | 0.391333 | 0.326943 | 0.366667 | 0.336195 | 0.367083 |
| SeqGAN 10% LSTM | 0.358667 | 0.336936 | 0.473333 | 0.386076 | 0.303227 |

## Per-seed results

| Model family | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Real-only LSTM | 42 | 0.336000 | 0.215278 | 0.124000 | 0.157360 | 0.225096 |
| Real-only LSTM | 43 | 0.624000 | 0.589080 | 0.820000 | 0.685619 | 0.726672 |
| Real-only LSTM | 44 | 0.214000 | 0.176471 | 0.156000 | 0.165605 | 0.149480 |
| SeqGAN 10% LSTM | 42 | 0.266000 | 0.145455 | 0.096000 | 0.115663 | 0.173320 |
| SeqGAN 10% LSTM | 43 | 0.446000 | 0.468384 | 0.800000 | 0.590842 | 0.387712 |
| SeqGAN 10% LSTM | 44 | 0.364000 | 0.396970 | 0.524000 | 0.451724 | 0.348648 |

Values in the first table are means across seeds 42, 43, and 44; error bars in the graph are one standard deviation.

Caution: Tranco root URLs are legitimate proxies and are structurally easier than arbitrary benign deep links. This is a small temporal snapshot, not a population-wide estimate.
