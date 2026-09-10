# Reproduction guide

## 1. Environment

Recommended environment:

- Linux
- Python 3.11 or newer
- NVIDIA GPU with a CUDA-enabled PyTorch build for neural and GAN training
- Node.js and npm for extension verification

Create a separate environment for each project version because the versions represent different experimental stages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Version_3/requirements.txt
```

Install the CUDA build of PyTorch appropriate for the local driver rather than silently falling back to CPU.

## 2. Restore datasets

Obtain the datasets from their permitted source or archival record. Place them at the paths in `data/artifacts_manifest.json`, then run:

```bash
python3 scripts/verify_artifacts.py
```

The Version 1 and Version 2 2M files are byte-identical. To avoid duplication, create a link after restoring the Version 2 copy:

```bash
ln Version_2/data/raw/urls_dataset2M.csv "Version_1/Data sets/urls_dataset2M.csv"
```

If hard links are unavailable, use a symbolic link or copy.

## 3. Version 1

```bash
cd Version_1
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans.prepare_data
python3 -m phishing_gans.prepare_lexical_features
python3 -m phishing_gans.train_classifier
python3 -m phishing_gans.train_traditional --model rf
python3 -m phishing_gans.train_traditional --model svm
```

Additional GAN, augmentation, and external-evaluation commands are documented in `Version_1/README.md` and `Version_1/docs/`.

## 4. Version 2

Version 2 must be prepared before Version 3 because Version 3's conflict audit consults the Version 2 processed database.

```bash
cd Version_2
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans_v2.prepare_data
python3 -m phishing_gans_v2.verify_data
python3 -m phishing_gans_v2.prepare_features
python3 -m phishing_gans_v2.train_baselines
python3 -m phishing_gans_v2.summarize_baselines
```

The full experiment matrix and generator commands are documented in `Version_2/README.md` and `Version_2/docs/`.

## 5. Version 3 baselines

```bash
cd Version_3
export PYTHONPATH="$PWD/src"
python3 -m phishing_gans_v3.prepare_data
python3 -m phishing_gans_v3.verify_data
python3 -m phishing_gans_v3.prepare_features
python3 -m phishing_gans_v3.audit_shortcuts
python3 -m phishing_gans_v3.train_traditional --seeds 42,43,44
python3 -m phishing_gans_v3.train_neural --models cnn,lstm,rnn --seeds 42,43,44
python3 -m phishing_gans_v3.summarize_baselines
```

## 6. Version 3 hybrid and GAN study

```bash
python3 -m phishing_gans_v3.prepare_hybrid_domain_data
python3 -m phishing_gans_v3.train_hybrid_domain
python3 -m phishing_gans_v3.prepare_ms_hardgan
python3 -m phishing_gans_v3.train_ms_hardgan
python3 -m phishing_gans_v3.generate_ms_hardgan
python3 -m phishing_gans_v3.prepare_ms_hardgan_augmentations
python3 -m phishing_gans_v3.train_ms_hardgan_augmentations
python3 -m phishing_gans_v3.summarize_ms_hardgan_augmentation
```

Use the frozen validation data for threshold selection. Do not tune thresholds or synthetic-selection rules on final test sets.

## 7. Tests

```bash
cd Version_3
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -v

cd ../extension
npm ci
npm test
```

Some Python tests require restored processed data. The browser extension itself is complete and includes its final ONNX models.

## 8. Expected evidence

Aggregated original results are preserved in `results/`. Reproduced values should be compared using the same dataset hashes, hostname-disjoint splits, seeds, thresholds, and metric definitions. GPU libraries can produce small numeric differences; material differences require an environment and data audit rather than manual adjustment.
