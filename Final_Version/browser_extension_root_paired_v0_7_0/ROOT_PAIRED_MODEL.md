# Root-Paired Research Extension v0.7.0

This folder is an isolated copy of the working v0.6.0 extension. It uses the
pair-mass-preserved root-paired HYB-MS + 10% MS-HardGAN-C ensemble from
`Final_Version`. The original Version 3 extension and its model files are not
modified.

## What changed

- The three ONNX files were exported from root-paired seeds 42, 43, and 44.
- Equivalent empty-path and `/` root forms shared the original training mass.
- Browser tiers were re-derived on the same frozen 23,100-row calibration
  cohort with the same registered FPR-cap method.
- The extension name and version identify this copy as Root-Paired v0.7.0.

## Frozen tier boundaries

| Tier | Lower bound | Upper bound |
|---|---:|---:|
| Allow | -- | 0.8217372298 |
| Secondary review | 0.8217372298 | 0.9899072647 |
| Caution | 0.9899072647 | 0.9991914630 |
| Warning | 0.9991914630 | 0.9999473691 |
| Block | 0.9999473691 | -- |

The caution boundary is the primary detection threshold. Thresholds were not
selected from final test labels.

## Known trade-off

The paired class-flip rate fell from 60.35% to 1.59%. In the controlled
external comparison, recall was 1.48 percentage points lower than Version 3.
This extension is therefore a separate research candidate, not a silent
replacement for the original v0.6.0 control.

## Install

Disable the old extension before testing this one so two copies do not scan the
same navigation. Open `chrome://extensions`, enable Developer mode, choose
**Load unpacked**, and select this folder. Confirm that the displayed extension
is **Phishing URL Guard - Root-Paired**, version **0.7.0**.

