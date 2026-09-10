# Fair HYB-MS Natural-Sampling GAN Experiment

Every condition contains the exact same 363,692 real URLs. Each stored row is visited once per epoch with seeded shuffling. The augmented conditions add the already frozen nested GAN pools; no 65/35 source-weighted sampler is used.

Metrics are mean ± sample standard deviation across seeds 42/43/44. Thresholds were frozen from validation/domain-validation and were not tuned on either external set.

## Internal frozen test

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 0.9356 ± 0.0043 | 0.9999 ± 0.0001 | 0.6930 ± 0.0206 | 0.8185 ± 0.0143 | 0.9978 ± 0.0000 | 0.9951 ± 0.0001 | 0.0000 ± 0.0000 | 0.3070 ± 0.0206 |
| GAN 10% | 0.9364 ± 0.0078 | 0.9999 ± 0.0001 | 0.6966 ± 0.0372 | 0.8208 ± 0.0259 | 0.9971 ± 0.0004 | 0.9938 ± 0.0008 | 0.0000 ± 0.0000 | 0.3034 ± 0.0372 |
| GAN 20% | 0.9320 ± 0.0110 | 0.9998 ± 0.0001 | 0.6758 ± 0.0523 | 0.8057 ± 0.0368 | 0.9969 ± 0.0001 | 0.9933 ± 0.0004 | 0.0000 ± 0.0000 | 0.3242 ± 0.0523 |
| GAN 30% | 0.9277 ± 0.0149 | 0.9999 ± 0.0001 | 0.6552 ± 0.0711 | 0.7902 ± 0.0511 | 0.9967 ± 0.0004 | 0.9928 ± 0.0006 | 0.0000 ± 0.0000 | 0.3448 ± 0.0711 |
| GAN 40% | 0.9266 ± 0.0221 | 0.9997 ± 0.0003 | 0.6500 ± 0.1059 | 0.7845 ± 0.0782 | 0.9965 ± 0.0003 | 0.9924 ± 0.0003 | 0.0001 ± 0.0001 | 0.3500 ± 0.1059 |

## Frozen external 10K

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 0.6577 ± 0.0136 | 0.9883 ± 0.0037 | 0.3191 ± 0.0268 | 0.4820 ± 0.0306 | 0.9351 ± 0.0081 | 0.9487 ± 0.0089 | 0.0037 ± 0.0010 | 0.6809 ± 0.0268 |
| GAN 10% | 0.6452 ± 0.0325 | 0.9873 ± 0.0029 | 0.2941 ± 0.0650 | 0.4506 ± 0.0775 | 0.9077 ± 0.0195 | 0.9233 ± 0.0182 | 0.0037 ± 0.0005 | 0.7059 ± 0.0650 |
| GAN 20% | 0.6239 ± 0.0395 | 0.9879 ± 0.0019 | 0.2510 ± 0.0802 | 0.3960 ± 0.0986 | 0.8943 ± 0.0177 | 0.9110 ± 0.0149 | 0.0031 ± 0.0013 | 0.7490 ± 0.0802 |
| GAN 30% | 0.6247 ± 0.0360 | 0.9776 ± 0.0167 | 0.2561 ± 0.0786 | 0.4014 ± 0.0937 | 0.8764 ± 0.0102 | 0.8864 ± 0.0065 | 0.0067 ± 0.0068 | 0.7439 ± 0.0786 |
| GAN 40% | 0.6254 ± 0.0600 | 0.9541 ± 0.0295 | 0.2666 ± 0.1355 | 0.4033 ± 0.1590 | 0.8469 ± 0.0210 | 0.8618 ± 0.0132 | 0.0158 ± 0.0156 | 0.7334 ± 0.1355 |

## Imbalanced external 30K

| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 0.8043 ± 0.0071 | 0.9847 ± 0.0099 | 0.3547 ± 0.0214 | 0.5213 ± 0.0239 | 0.9399 ± 0.0102 | 0.9172 ± 0.0191 | 0.0023 ± 0.0014 | 0.6453 ± 0.0214 |
| GAN 10% | 0.8009 ± 0.0200 | 0.9851 ± 0.0049 | 0.3428 ± 0.0657 | 0.5063 ± 0.0738 | 0.9201 ± 0.0091 | 0.8867 ± 0.0173 | 0.0021 ± 0.0004 | 0.6572 ± 0.0657 |
| GAN 20% | 0.7884 ± 0.0197 | 0.9816 ± 0.0062 | 0.3024 ± 0.0690 | 0.4594 ± 0.0779 | 0.9045 ± 0.0137 | 0.8624 ± 0.0179 | 0.0026 ± 0.0015 | 0.6976 ± 0.0690 |
| GAN 30% | 0.7865 ± 0.0255 | 0.9616 ± 0.0321 | 0.3046 ± 0.1016 | 0.4553 ± 0.1076 | 0.8896 ± 0.0104 | 0.8204 ± 0.0122 | 0.0064 ± 0.0072 | 0.6954 ± 0.1016 |
| GAN 40% | 0.7765 ± 0.0280 | 0.9171 ± 0.0619 | 0.2902 ± 0.1292 | 0.4271 ± 0.1360 | 0.8614 ± 0.0215 | 0.7818 ± 0.0250 | 0.0144 ± 0.0156 | 0.7098 ± 0.1292 |

## Interpretation

- GAN 10% gives a very small internal threshold-metric increase over real-only, but its internal ROC-AUC and PR-AUC decrease.
- On both frozen external sets, real-only is best among these natural-sampling conditions for accuracy, recall, F1, ROC-AUC, PR-AUC, and FNR.
- Increasing the GAN share causes a generally monotonic external ranking decline. On the 30K set, ROC-AUC falls from 0.9399 (real-only) to 0.8614 (GAN 40%).
- This means the current GAN pool does not replace real external coverage. Its synthetic phishing distribution increasingly pulls the model away from later real URLs.
- The experiment is fair with respect to stored real rows and sampling. It does not claim identical feature representations between fundamentally different baseline architectures.
