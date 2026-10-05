# v0.7.0 — Root-Paired Research Model

- Created as a separate copy of the working v0.6.0 extension.
- Replaced the three Version 3 ONNX files with the pair-mass-preserved
  root-paired HYB-MS + 10% MS-HardGAN-C checkpoints for seeds 42, 43, and 44.
- Re-derived all browser tier boundaries from the same frozen calibration
  cohort and registered FPR-cap rules. Final test labels were not used.
- Kept the existing interface, Arabic localisation, reviewed-service policy,
  local feedback, privacy, deduplication, and exact-host override behaviour.
- Identified the extension clearly as `Phishing URL Guard - Root-Paired` so it
  cannot be confused with the unchanged v0.6.0 deployment control.

The candidate reduced equivalent root-path class flips from 60.35% to 1.59%,
but its controlled external recall was 1.48 percentage points lower. It remains
an isolated research candidate rather than a silent replacement for v0.6.0.

