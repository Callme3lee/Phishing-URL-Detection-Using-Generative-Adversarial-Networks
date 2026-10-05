# Final Version — root-representation robustness experiment

This directory is isolated from `Version_3`. It evaluates whether the frozen
Version 3 HYB-MS + 10% MS-HardGAN-C training recipe can stop treating the
presence of `/` on a root-only URL as a label shortcut.

The authoritative source is the exact dataset used by the browser extension:
`Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/processed`.
The original 367,184 real rows and 40,410 filtered MS-HardGAN-C rows are
retained. For each root-only training row, one counterpart is added with the
opposite root representation (`host` versus `host/`) and the same label and
source group. Non-root URLs are unchanged. Validation, domain-validation, test,
and external evaluation data are never augmented.

Version 3 and browser extension v0.6.0 are controls and are not modified.

## Reproduction

From the repository root:

```bash
PYTHONPATH=Version_3/src python Final_Version/scripts/prepare_root_paired.py
python Final_Version/scripts/derive_pair_preserving_weights.py
PYTHONPATH=Version_3/src python Final_Version/scripts/train_pair_weighted.py \
  --config Version_3/configs/project.json \
  --data-root Final_Version/data/root_paired_hyb_ms_gan10_v1 \
  --output-root Final_Version/artifacts/root_paired_mass_preserved_v1 \
  --conditions gan_10pct --variants hyb_ms --seeds 42,43,44
PYTHONPATH=Version_3/src python Final_Version/scripts/evaluate_root_paired_candidate.py \
  --candidate-checkpoints Final_Version/artifacts/root_paired_mass_preserved_v1/gan_10pct/hyb_ms \
  --output Final_Version/artifacts/results/root_paired_mass_preserved_v1
python Final_Version/tests/test_experiment_contract.py
```

Training requires CUDA, matching the Version 3 training contract.

The primary refined candidate uses pair-mass-preserving sampling: each of the
two equivalent root representations has weight 0.5, so their combined training
influence remains 1.0. This avoids accidentally making root URLs more important
merely because their harmless representations were duplicated.
