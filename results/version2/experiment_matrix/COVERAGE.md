# Experiment Matrix Coverage

Protocol: screen every GAN/dose/classifier combination with seed 42, then confirm selected conditions with seeds 43 and 44.

Full optional matrix: **60/108** runs (55.6%).
Augmentation screening: **32/32** runs.

| Condition | Complete | Optional total |
|---|---:|---:|
| Real only | 12 | 12 |
| gumbel_lsgan 10% | 12 | 12 |
| gumbel_lsgan 20% | 4 | 12 |
| gumbel_lsgan 30% | 4 | 12 |
| gumbel_lsgan 40% | 4 | 12 |
| seqgan 10% | 12 | 12 |
| seqgan 20% | 4 | 12 |
| seqgan 30% | 4 | 12 |
| seqgan 40% | 4 | 12 |

All four classifiers at seed 42 complete a screen. Seeds 43 and 44 are confirmation runs, not required for candidates that already underperform in screening.
