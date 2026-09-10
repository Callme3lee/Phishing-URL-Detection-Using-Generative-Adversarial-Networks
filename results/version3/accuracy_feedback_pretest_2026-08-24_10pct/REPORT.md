# Fresh 10K pre-feedback-database evaluation

## Result

The frozen +10% GAN ensemble improved accuracy, precision, recall, F1, PR-AUC, FPR and FNR relative to the frozen real-only ensemble. The registered augmentation rule passed because recall, F1 and ROC-AUC increased while FPR decreased.

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 97.20% | 86.66% | 85.10% | 85.87% | 97.03% | 91.34% | 1.46% | 14.90% |
| +10% GAN | 97.84% | 89.92% | 88.30% | 89.10% | 97.45% | 93.51% | 1.10% | 11.70% |

The GAN model reduced false alerts from 131 to 99 and missed phishing URLs from 149 to 117. It uniquely corrected 106 decisions and uniquely worsened 42 decisions (exact paired p=1.446e-07).

Practical gate: Both conditions passed: recall was at least 80% and FPR was at most 2%.

The paired-bootstrap F1 delta was +3.23% with a 95% interval of +2.10% to +4.45%. The FPR delta was -0.36% with a 95% interval of -0.59% to -0.16%.

## Cohort

- 9,000 legitimate and 1,000 phishing URLs.
- Source composition: 1,000 PhishTank verified-online, 9,000 Tranco current root proxy.
- Zero URL, hostname or 128-token identity overlap with training, development, calibration and earlier external cohorts.
- Dataset SHA-256: `cec7f6b43b3ebcbd12e345833489f603071459a551e997da749390e69d1a4ffe`.
- No checkpoint, threshold or selection rule was changed after the cohort was frozen.

This is useful pre-feedback evidence, but Tranco roots remain a legitimate proxy and underrepresent long legitimate paths, queries and redirects.
