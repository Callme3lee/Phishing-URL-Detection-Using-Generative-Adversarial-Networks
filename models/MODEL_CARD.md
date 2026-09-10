# HYB-MS + 10% MS-HardGAN-C model card

## Purpose

The final browser model is a defensive phishing-URL research classifier. It evaluates URL text locally and assigns a score used by a tiered browser policy. It is intended for academic demonstration and controlled user evaluation, not production certification.

## Composition

- Three independently trained seeds: 42, 43, and 44
- Character-level CNN representation
- 35 scheme-neutral, PSL-aware lexical features
- Hybrid feature fusion and multi-source/domain training controls
- Training condition containing 10% filtered MS-HardGAN-C phishing-style augmentation
- Ensemble score formed from the three frozen models
- Validation-selected thresholds; test data was not used for threshold tuning

GAN generation occurs only before classifier training. The browser does not generate URLs at runtime.

## Packaged ONNX models

| File | SHA-256 |
|---|---|
| `hyb_gan10_seed42.onnx` | `70da6879cb57d638b20bf8a3135ca73b2a9aa6203c5089ad93bdf73de5dcf914` |
| `hyb_gan10_seed43.onnx` | `254a41b49af7b0771fa85a6af3acbf1c3a01780c532eda093c5172524b748b87` |
| `hyb_gan10_seed44.onnx` | `f6e8205385918e11b4cfdd8a9b0a0335ec13422bff0bf5bd84a26f512789aac3` |

The browser preprocessing, scaler, vocabulary, Public Suffix List rules, and policy metadata are stored beside the models in `extension/models/`.

## Input and output

- Input: HTTP(S) URL text
- Encoded sequence length: 128 positions
- Output: continuous phishing score, then a separately defined policy tier
- Runtime: ONNX Runtime Web, locally inside the extension

The score is not a calibrated probability or guarantee of safety.

## Limitations

- URL text cannot observe page content, certificate state, account behaviour, or all DNS/reputation evidence.
- Distribution shift can increase false negatives or false positives.
- Long legitimate login and service URLs can resemble phishing URLs.
- Trusted platforms may host user-controlled content.
- GAN augmentation improved selected frozen evaluations but did not improve every model, ratio, dataset, or metric.
- The extension is asynchronous and is not a network firewall.

See the graduation report and `Version_3/docs/` for the controlled comparisons and deployment limitations.
