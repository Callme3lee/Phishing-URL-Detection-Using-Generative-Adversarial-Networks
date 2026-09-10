# Fresh 10K pre-feedback-database evaluation

## Result

The frozen +10% GAN ensemble improved accuracy, precision, recall, F1, PR-AUC, FPR and FNR relative to the frozen real-only ensemble. ROC-AUC decreased slightly from 93.11% to 92.98%. Because the registered rule required ROC-AUC to increase, the strict rule was not fully met.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 97.35% | 73.93% | 72.60% | 73.26% | 93.11% | 78.35% | 1.35% | 27.40% |
| +10% GAN | 97.75% | 78.95% | 75.00% | 76.92% | 92.98% | 80.91% | 1.05% | 25.00% |

The GAN model reduced false alerts from 128 to 100 and missed phishing URLs from 137 to 125. It uniquely corrected 84 decisions and uniquely worsened 44 decisions (exact paired p=0.0005158).

Neither condition passed the complete practical gate because real-only recall was 72.60% and GAN recall was 75.00%, below the registered 80% minimum. Both conditions remained within the 2% FPR ceiling.

The paired-bootstrap F1 delta was +3.66% with a 95% interval of +1.75% to +5.72%. The FPR delta was -0.29% with a 95% interval of -0.49% to -0.11%.

## Cohort

- 9,500 legitimate Tranco root proxies and 500 current phishing URLs.
- Phishing sources: 210 OpenPhish and 290 PhishTank verified-online URLs.
- Zero URL, hostname or 128-token identity overlap with training, development, calibration and earlier external cohorts.
- Dataset SHA-256: `6801eb222c49779fb431f4c45e98e85ac984ae8a6bbd65f1c5c29abdc52cb812`.
- No checkpoint, threshold or selection rule was changed after the cohort was frozen.

This is useful pre-feedback evidence, but Tranco roots remain a legitimate proxy and underrepresent long legitimate paths, queries and redirects.
