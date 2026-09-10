# Browser false-positive mitigation — v0.2

## Problem

The v0.1 extension replaced the requested page for both `warning` and `block` model tiers. Because the primary model uses URL text only, some legitimate login pages, long paths, query strings, punycode domains, and unusual hostnames received strong scores. The browser behavior made those model false positives feel like confirmed phishing decisions.

The extension also had no persistent, user-controlled way to prevent repeated warnings for a recognized legitimate hostname.

## Implemented solution

Version 0.2 keeps the existing ONNX models and all calibrated score thresholds unchanged. It changes only the deployment response:

| Model tier | v0.1 behavior | v0.2 behavior |
|---|---|---|
| Allow | Normal navigation | Normal navigation |
| Secondary review | Passive badge | Passive badge |
| Caution | Passive badge | Passive badge |
| Warning | Interstitial interruption | Passive high-risk badge/popup |
| Block | Interstitial interruption | Interstitial with explicit override |

The popup now says “risk pattern” and explicitly states that the score is not proof or a probability. Users can add or remove the exact current hostname from a local trusted-sites list. Trust does not automatically include subdomains.

The implementation is fail-open: an unavailable model/runtime records an error but does not redirect the user.

## Post-hoc burden diagnostic

This diagnostic used already evaluated data, so it measures expected behavior and is not a fresh validation claim.

| Cohort | Legitimate rows | v0.1 false interruptions | v0.2 false interruptions | Avoided |
|---|---:|---:|---:|---:|
| Primary 30K | 28,500 | 88 | 6 | 82 |
| Long-URL stress | 2,000 | 22 | 0 | 22 |

The stricter interstitial policy also interrupts fewer phishing URLs: 564 instead of 1,176 on the primary cohort and 92 instead of 147 on the stress cohort. Those URLs may still receive passive warning badges. Therefore v0.2 reduces disruptive false positives; it does not improve the classifier itself.

## Model-version boundary

The extension continues to use the previously exported and browser-calibrated “Updated HYB-MS +10% MS-HardGAN-C” bundle. It is not silently replaced with the final conservative real-only research reference because a model replacement requires a separate ONNX export, FP32 portability calibration, parity validation, and prospective browser-policy confirmation.

## Validation

- 12/12 preprocessing fixtures pass.
- Conservative policy unit tests pass.
- ONNX Runtime Web score parity passes with maximum fixture difference `2.384e-7`.
- Node-hosted WASM measurement after the change: approximately 212 ms initialization, 7.8 ms mean inference, and 18.3 ms p95 on this machine.
- JSON and JavaScript syntax validation pass.

Complete diagnostic: `artifacts/results/browser_v0_2_policy_diagnostic/browser_v0_2_policy_diagnostic.pdf`.
