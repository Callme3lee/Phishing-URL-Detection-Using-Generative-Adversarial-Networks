# Retrained multi-source results

## Data and protocol

The GAN training corpus contains only malicious URLs and preserves hostname
separation:

| Split | URLs |
|---|---:|
| Training | 1,535,986 |
| Validation | 189,679 |
| Held out | 186,857 |

It combines the original 2M training partitions with the leakage-screened added
sources. A further 15,493 exact encoded duplicates shared by the two added
sources were removed.

The shared autoregressive generator was pretrained on 200,000 randomly selected
training URLs for five epochs. Validation perplexity improved from 10.11 to
7.15. LSGAN and WGAN-GP were then independently fine-tuned for three epochs on
the same 200,000-row subset and seed.

Epoch 2 was selected for both models after comparing 1,000 generated URLs from
every checkpoint. LSGAN epoch 3 was rejected because its structural validity
fell to 49.2%, demonstrating partial collapse.

## Four classifier baselines

All values use the untouched 252,430-URL hostname-separated real test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN | 96.6561% | 97.5210% | 96.5923% | 97.0544% | 99.5803% |
| LSTM | **96.7781%** | 97.3184% | **97.0243%** | **97.1711%** | **99.6071%** |
| Random Forest | 95.5873% | 97.2725% | 94.9246% | 96.0842% | 99.0269% |
| Linear SVM | 93.7618% | **97.4861%** | 91.4196% | 94.3554% | 96.2669% |

The LSTM is the strongest overall baseline by Accuracy, Recall, F1, and ROC-AUC.
The SVM has the highest Precision but substantially lower Recall.

## Retrained GAN uniqueness

Each selected checkpoint generated 50,000 URLs at temperature 0.8 with the same
generation seed.

| GAN | Raw unique | Raw uniqueness | Strictly accepted | Acceptance | Filtered uniqueness |
|---|---:|---:|---:|---:|---:|
| LSTM-LSGAN | 48,571 | 97.1420% | 46,233 | 92.4660% | 100.0000% |
| LSTM-WGAN-GP | **49,940** | **99.8800%** | **47,431** | **94.8620%** | 100.0000% |

WGAN-GP is the stronger retrained generator by raw uniqueness and strict-filter
acceptance. Both final accepted corpora are fully unique because duplicates,
training memorization, malformed URLs, invalid hosts/TLDs, repeated-character
artifacts, and non-ASCII outputs are removed.

Machine-readable results are written to
`artifacts/final_project_results.json` by
`src/phishing_gans/project_results.py`.
