# GAN-unseen external evaluation

## Protocol

The augmented classifiers were evaluated without retraining or threshold
adjustment on hostname-held-out portions of both added sources. The held-out
hostnames were assigned to the 10% test bucket before GAN training and were not
used to pretrain/fine-tune either GAN.

| Source | Benign | Malicious | Total |
|---|---:|---:|---:|
| `URL dataset.csv` | 38,990 | 10,319 | 49,309 |
| `new_data_urls.csv` | 22,893 | 32,570 | 55,463 |

Encoded malicious duplicates were removed using the same cross-source rule as
the GAN training-corpus builder. Every classifier uses its decision threshold
selected on the original validation set.

## Best results on `new_data_urls`

| Model | Best condition | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| CNN | WGAN-GP 40% | 75.92% | 75.17% | 88.08% | **81.12%** | **86.36%** |
| LSTM | WGAN-GP 40% | 75.37% | 75.52% | 85.91% | **80.38%** | **84.48%** |
| RF | Combined 40% | 76.54% | 75.75% | 88.32% | **81.55%** | **86.33%** |
| SVM | WGAN-GP 40% | 77.04% | 79.00% | 82.94% | **80.92%** | **80.98%** |

Augmentation transfers strongly to this source. WGAN-GP 40% is best for CNN,
LSTM, and SVM, while the combined 40% corpus is best for RF.

## Results on `URL dataset`

The best trustworthy ranking performance is real-only for every model:

| Model | Real-only Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN | 79.99% | 96.14% | 4.58% | 8.75% | **79.27%** |
| LSTM | 83.15% | 72.28% | 31.61% | **43.99%** | **85.05%** |
| RF | 84.75% | 88.84% | 31.00% | **45.96%** | **77.99%** |
| SVM | 73.50% | 29.19% | 18.66% | 22.77% | **51.47%** |

Some augmented models report F1 near 34.6%, but this is not an improvement. For
example, CNN WGAN-GP/combined 40% predicts almost every row as malicious:
approximately 20.92% accuracy, 20.92% precision, and 99.96% recall. Its ROC-AUC
falls to 46.69% or 40.57%. RF/SVM augmented ROC-AUC can fall far below 50%.
This is threshold and ranking collapse under source shift.

## Source-balanced conclusion

Using an unweighted average across the two held-out sources, real-only has the
highest ROC-AUC for every classifier:

| Model | Best source-macro ROC condition | Source-macro F1 | Source-macro ROC-AUC |
|---|---|---:|---:|
| CNN | Real only | 30.33% | **71.09%** |
| LSTM | Real only | 55.50% | **79.48%** |
| RF | Real only | 52.91% | **72.88%** |
| SVM | Real only | 46.06% | **64.26%** |

Therefore the GANs add useful patterns for `new_data_urls`, but the benefit does
not generalize consistently across sources. Real-only remains the safest overall
modeling choice, with real-only LSTM giving the strongest source-balanced
ROC-AUC. A production system would require multi-source real training data and
separate calibration data from the deployment domain.
