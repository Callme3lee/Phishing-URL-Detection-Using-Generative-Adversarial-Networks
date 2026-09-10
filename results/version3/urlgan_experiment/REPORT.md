# URL-GAN Experiment

## Simple names

- `URL-GAN`: the complete new GAN.
- `2L-LSTM-G`: its two-layer LSTM generator.
- `Char-CNN-D`: its character-CNN discriminator.
- `GS`: Gumbel-Softmax relaxation.
- `SG`: the original compact SeqGAN used as the reference.

## Fixed training contract

All generators used the same 82,219 phishing URLs from the original frozen training split, the same 168-token character vocabulary, maximum length 128, batch size 64, seed 42, and a 500,000-sample final generation budget. No legitimate, validation, test, calibration, or external URL was used to train the generators.

## Results

| GAN | Accepted | Accept. | Validity | Uniqueness | Source AUC ↓ | Lex. SMD ↓ | Clust. cov. ↑ | Challenge ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SG | 349,687 | 0.6994 | 0.9817 | 0.9984 | 0.7357 | 0.1104 | 0.9400 | 0.1091 |
| URL-GAN | 239,368 | 0.4787 | 0.9815 | 0.9981 | 0.8817 | 0.3169 | 0.9000 | 0.0505 |
| RelGAN | 12,869 | 0.0257 | 0.8143 | 0.9939 | 0.9888 | 0.8338 | 0.8600 | 0.0009 |

Source AUC closer to 0.5 is better; lower lexical SMD is better. Challenge is neither automatically good nor bad—it measures how often most frozen real-only classifiers miss a generated phishing URL.

## Conclusion

URL-GAN is stable and usable, clearly better than RelGAN, but the original compact SeqGAN remains the best augmentation generator because it has higher filter yield and closer real-phishing distributions.
