# Controlled GAN-percentage classifier results

## Protocol

Every condition uses the identical class-stratified subset of 69,300 unique real
training URLs: 29,334 benign and 39,966 malicious. Generated URLs comprise
exactly 20%, 30%, or 40% of the final training dataset. The generated subsets
are nested, so the 20% samples are contained in 30%, and 30% is contained in
40%.

All models use the original unchanged validation set of 254,549 real URLs and
test set of 252,430 real URLs. Decision thresholds are selected using validation
data only. CNN and LSTM use five epochs and seed 42; RF and linear SVM use their
original baseline hyperparameters and seed 42.

## Best F1 result for each model

| Model | Best condition | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 change from real-only |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | LSGAN 40% | 95.0129% | 96.7944% | 94.3814% | **95.5727%** | 98.9514% | **+0.1285 pp** |
| LSTM | Real only | 95.1095% | 97.1007% | 94.2390% | **95.6485%** | 98.8212% | 0.0000 pp |
| Random Forest | Real only | 95.2256% | 97.3245% | 94.2189% | **95.7465%** | 98.7719% | 0.0000 pp |
| Linear SVM | LSGAN 40% | 94.0249% | 98.1485% | 91.2446% | **94.5707%** | 96.2304% | **+0.2406 pp** |

## F1 dose response

Values in parentheses are percentage-point changes from the same model's
real-only reference.

| Model | Real only | LSGAN 20% | LSGAN 30% | LSGAN 40% |
|---|---:|---:|---:|---:|
| CNN | 95.4441% | 95.5102% (+0.0661) | 95.5676% (+0.1235) | **95.5727% (+0.1285)** |
| LSTM | **95.6485%** | 95.5220% (-0.1265) | 95.6146% (-0.0339) | 95.6099% (-0.0386) |
| RF | **95.7465%** | 95.7188% (-0.0276) | 95.7446% (-0.0019) | 95.6862% (-0.0603) |
| SVM | 94.3301% | 94.4601% (+0.1300) | 94.5234% (+0.1933) | **94.5707% (+0.2406)** |

| Model | Real only | WGAN-GP 20% | WGAN-GP 30% | WGAN-GP 40% |
|---|---:|---:|---:|---:|
| CNN | **95.4441%** | 95.3399% (-0.1043) | 95.4534% (+0.0093) | 95.2897% (-0.1545) |
| LSTM | **95.6485%** | 95.4039% (-0.2445) | 95.3847% (-0.2638) | 95.3423% (-0.3062) |
| RF | **95.7465%** | 95.6615% (-0.0850) | 95.5616% (-0.1849) | 95.4508% (-0.2957) |
| SVM | **94.3301%** | 94.1642% (-0.1659) | 94.0549% (-0.2752) | 93.9508% (-0.3793) |

| Model | Real only | Combined 20% | Combined 30% | Combined 40% |
|---|---:|---:|---:|---:|
| CNN | 95.4441% | 95.4395% (-0.0046) | **95.5160% (+0.0719)** | 95.4968% (+0.0527) |
| LSTM | **95.6485%** | 95.5295% (-0.1190) | 95.5301% (-0.1184) | 95.5803% (-0.0682) |
| RF | **95.7465%** | 95.7298% (-0.0167) | 95.6895% (-0.0570) | 95.6194% (-0.1271) |
| SVM | **94.3301%** | 94.3196% (-0.0105) | 94.2876% (-0.0425) | 94.2521% (-0.0780) |

## Conclusion

LSGAN shows the most useful dose response. It improves CNN and linear SVM F1,
with the largest measured gains at 40%. It approximately matches, but does not
improve, Random Forest and slightly reduces LSTM F1. WGAN-GP's superior URL
uniqueness does not translate into superior classifier augmentation: most
WGAN-GP conditions reduce F1, increasingly so at higher percentages. Combining
the GANs dilutes LSGAN's benefit and gives only a small CNN improvement at 30%.

These are single-seed differences below 0.25 percentage points. They are useful
for choosing the next experiment, but must not yet be described as statistically
significant. The recommended follow-up is a three-seed repeat of real-only,
LSGAN 30%, and LSGAN 40% for CNN and SVM.
