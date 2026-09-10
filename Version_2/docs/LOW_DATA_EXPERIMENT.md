# Low-Data Augmentation Experiment

## Frozen design

The experiment uses one immutable hostname-grouped subset containing 210,434
real training URLs (10.24% of the full training rows): 88,356 legitimate and
122,078 phishing. The slightly non-exact percentage is a consequence of keeping
all rows from each selected hostname together. The complete validation and test
sets remain unchanged.

SeqGAN augmentation is screened at 1%, 2%, 5%, 10%, and 15% of final training
size. The generated counts are 2,126, 4,295, 11,075, 23,382, and 37,135.
Generated rows receive sample
weight 0.5; real rows receive weight 1.0. Class balancing is calculated only
from the frozen real subset, so adding synthetic phishing URLs cannot reduce the
importance of real phishing examples.

## Seed-42 screening result

| Model | Condition | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| CNN | Real only | 0.953324 | 0.992672 | 0.925058 | 0.957673 | 0.993886 |
| CNN | SeqGAN 1% | 0.952983 | 0.992571 | 0.924551 | 0.957354 | 0.993892 |
| CNN | SeqGAN 2% | **0.953868** | 0.992246 | **0.926420** | **0.958204** | 0.994025 |
| CNN | SeqGAN 5% | 0.953249 | 0.991952 | 0.925607 | 0.957631 | 0.993860 |
| CNN | SeqGAN 10% | 0.953285 | 0.992304 | 0.925336 | 0.957651 | **0.994040** |
| CNN | SeqGAN 15% | 0.953392 | 0.992261 | 0.925565 | 0.957754 | 0.994007 |
| LSTM | Real only | 0.952777 | 0.991746 | 0.924968 | 0.957194 | 0.993486 |
| LSTM | SeqGAN 1% | 0.951516 | 0.991991 | 0.922508 | 0.955989 | 0.993502 |
| LSTM | SeqGAN 2% | 0.951500 | 0.991330 | 0.923106 | 0.956002 | 0.993307 |
| LSTM | SeqGAN 5% | 0.951809 | **0.992202** | 0.922828 | 0.956258 | 0.993047 |
| LSTM | SeqGAN 10% | **0.953007** | 0.991355 | **0.925746** | **0.957428** | **0.993552** |
| LSTM | SeqGAN 15% | 0.953400 | 0.992122 | 0.925711 | 0.957767 | 0.993150 |

CNN-2% and LSTM-10% improve accuracy, recall, F1, and ROC-AUC while slightly
reducing precision. CNN-15% has the same four-metric direction but smaller gains
than CNN-2%. LSTM-15% improves accuracy, precision, recall, and F1, but reduces
ROC-AUC. The 1% condition does not improve either model overall. These are
screening observations from one seed, not final evidence. The balanced promoted
conditions remain CNN-2% and LSTM-10%; the next valid step is to repeat them
with seeds 43 and 44 and calculate paired confidence intervals.

## Three-seed confirmation

The promoted conditions were compared with matched real-only controls using
the same seeds (42, 43, and 44). Values below are three-seed means. Change and
95% CI are paired augmented-minus-real-only differences in percentage points.

| Model | Metric | Real only | Augmented | Mean change (95% CI), pp |
|---|---|---:|---:|---:|
| CNN + SeqGAN 2% | Accuracy | 0.953188 | 0.953201 | +0.0013 (-0.1174, +0.1200) |
| CNN + SeqGAN 2% | Precision | 0.992537 | 0.992305 | -0.0232 (-0.0747, +0.0283) |
| CNN + SeqGAN 2% | Recall | 0.924945 | 0.925188 | +0.0243 (-0.2316, +0.2802) |
| CNN + SeqGAN 2% | F1 | 0.957550 | 0.957571 | +0.0022 (-0.1121, +0.1165) |
| CNN + SeqGAN 2% | ROC-AUC | 0.993799 | 0.993853 | +0.0053 (-0.0215, +0.0322) |
| LSTM + SeqGAN 10% | Accuracy | 0.953355 | 0.952071 | -0.1284 (-0.5460, +0.2893) |
| LSTM + SeqGAN 10% | Precision | 0.991892 | 0.991748 | -0.0144 (-0.1146, +0.0858) |
| LSTM + SeqGAN 10% | Recall | 0.925850 | 0.923719 | -0.2131 (-0.9459, +0.5198) |
| LSTM + SeqGAN 10% | F1 | 0.957734 | 0.956524 | -0.1209 (-0.5162, +0.2743) |
| LSTM + SeqGAN 10% | ROC-AUC | 0.993291 | 0.993265 | -0.0026 (-0.0315, +0.0262) |

CNN-2% is effectively tied with real-only: its average changes are tiny and
mixed by seed. LSTM-10% is worse on four of five mean metrics. Every interval
crosses zero, so this experiment does **not** establish a reliable augmentation
improvement. The correct result is that seed 42 supplied a promising screening
signal that did not reproduce consistently. No additional percentage should be
selected from the same screen after seeing these confirmation results; that
would turn the confirmation set into another tuning set.

Machine-readable evidence is in `artifacts/low_data/real_10_percent/results/summary.json`
and `confirmation.csv`. The professor-facing confidence-interval graph is
`paired_confirmation_ci.png` (with a PDF copy alongside it).
