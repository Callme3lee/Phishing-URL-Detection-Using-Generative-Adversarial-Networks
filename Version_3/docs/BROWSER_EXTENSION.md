# Browser extension research prototype

## Outcome and status

Version 3 now contains a self-contained Chrome/Chromium Manifest V3 prototype in `browser_extension/`. It runs the frozen **HYB-MS + 10% MS-HardGAN-C** three-seed ensemble locally with ONNX Runtime Web. It does not upload visited URLs.

This completes the implementation proof, not production certification. The extension remains a research prototype because the later deployment-confirmation experiment failed, live-browser resource behavior has not been measured across devices, and URL text alone cannot resolve every suspicious URL.

## What is deployed

- three hybrid CNN ONNX models, trained with seeds 42/43/44;
- character sequence input of length 128;
- the same 35 PSL-aware lexical features and robust scaler used in training;
- arithmetic-mean ensemble scoring;
- browser-specific FP32 numerical thresholds frozen from the same 23,100-row calibration cohort under the same FPR-cap rules;
- five actions: allow, secondary review, caution, warning, and block;
- two persistent navigation modes: Normal (strict-tier interstitial) and Passive (warnings only);
- explicit passive handling for valid IPv4 and IPv6 literal destinations;
- an explicit one-time override on the strict block page;
- a dismissible in-page caution notice for suspicious results that remain navigable;
- a safe text-only demonstration page that never visits its example addresses.

DNS is deliberately absent from this first bundle. The earlier DNS experiment remains useful research evidence, but adding network lookups would change privacy, permissions, availability and latency. It should be introduced later as an independent secondary signal, not silently mixed into this URL-only reproduction.

## Install it locally

1. Open `chrome://extensions` in Chrome or Chromium.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select the absolute `Version_3/browser_extension` directory.
5. Pin **Phishing URL Guard — Research Prototype**.
6. Open the extension and choose **Open safe demonstration** before testing ordinary browsing.

Do not intentionally browse to a live phishing URL for a demonstration. The bundled examples are reserved or illustrative strings and are scored without navigation.

## Behavior

On each top-level HTTP(S) navigation the lightweight service worker asks a local offscreen extension document to preprocess the URL and evaluate all three ONNX models. Keeping WebAssembly in the document context avoids Manifest V3 service-worker suspension/module-loading problems. The popup shows the resulting tier and score. Beginning with v0.2, only the strictest block tier receives an interstitial page with a **Go back** action and explicit overrides. Warning, caution, and review tiers remain passive. Users can trust or untrust one exact hostname locally; trust never automatically includes subdomains.

Version 0.3 adds a persistent mode selector in the popup. **Normal** retains the strict-block interstitial. **Passive** keeps the same scoring and visible risk tiers but never replaces the requested page. This is suitable for demonstrations and warning-only use; it does not change the classifier. Valid literal IP destinations are handled separately in both modes. Private/local addresses such as `192.168.1.1` receive a local-IP notice, while public addresses such as `8.8.8.8` receive an IP-review notice. Both remain passive because the URL-text model was designed around domain-name structure. This is not an IP allow-list: a public or local IP can still host malicious content.

Version 0.4 adds a dismissible notification inside navigable pages classified as `secondary_review`, `caution`, `warning`, or public-IP review. It also keeps a strict warning visible after the user explicitly chooses **Continue once**, and shows the same notice for a `block` result in Passive mode. Clearly allowed pages, exact-host trusted pages, and private/local IP pages do not receive this notification. The notice is advisory: it never modifies page content beyond its isolated overlay and never blocks interaction or navigation.

The score is a ranking output, not a probability that a site is safe or malicious. In particular, `0.99` must not be presented as “99% malicious.”

The conservative v0.2 policy does not alter the model or its calibrated thresholds. A post-hoc diagnostic reduced false interruptions from 88 to 6 on 28,500 primary legitimate URLs and from 22 to 0 on 2,000 deliberately long legitimate URLs. It also reduced phishing interstitials, while retaining passive warning badges. See `BROWSER_FALSE_POSITIVE_MITIGATION_V020.md`.

## Rebuild the frozen browser bundle

Run from `Version_3`:

```bash
export PYTHONPATH=src

python3 -m phishing_gans_v3.export_browser_bundle --overwrite
python3 -m phishing_gans_v3.validate_browser_bundle
python3 -m phishing_gans_v3.freeze_browser_policy
python3 -m phishing_gans_v3.validate_browser_bundle
```

The first validation materializes browser calibration scores. Freezing then applies the already declared maximum-recall/FPR-cap rule to those FP32 scores, without final-test tuning. The second validation is the final portability gate.

The checked-in extension already contains the exported models and frozen policy, so rebuilding is unnecessary for normal use.

## Run validation

Python/ONNX bundle validation:

```bash
export PYTHONPATH=src
python3 -m phishing_gans_v3.validate_browser_bundle
python3 -m pytest -q tests
```

JavaScript preprocessing and WebAssembly inference validation:

```bash
cd browser_extension
npm install
npm test
```

`npm install` is only required for the Node.js test harness. The extension runtime is already vendored and performs no CDN fetch.

## Current measured evidence

| Check | Result |
|---|---:|
| PyTorch-to-ONNX fixture maximum difference | `6.41e-7` |
| JavaScript preprocessing fixtures | 12/12 exact token matches; lexical error `<=1e-6` |
| ONNX Runtime Web fixture maximum score difference | `2.38e-7` |
| Full calibration rows checked | 23,100 |
| Browser calibration FPR | 0.500% |
| Browser calibration recall | 79.32% |
| Browser calibration F1 | 86.48% |
| Browser calibration ROC-AUC | 98.03% |
| Binary decisions changed by FP32 portability policy | 1/23,100 |
| Risk tiers changed | 2/23,100 |
| Node-hosted WebAssembly, repeated local runs | 167–197 ms initialization; 7–9 ms mean URL latency; 17–27 ms p95 |

The WebAssembly latency numbers use the same runtime as the extension but were measured in Node on this development machine. They are not a substitute for interactive Chrome measurements on several user devices.

Detailed machine-readable evidence is stored in:

- `browser_extension/tests/deployment_parity_report.json`;
- `browser_extension/tests/wasm_benchmark.json`;
- `browser_extension/tests/parity_fixtures.json`;
- `browser_extension/models/bundle_manifest.json`.

## Required work before claiming deployment readiness

1. Install and exercise the unpacked extension in interactive Chrome on multiple machines.
2. Measure cold start, warm inference, memory, CPU and battery impact during realistic browsing.
3. Run a new prospective low-prevalence evaluation collected after all policy decisions were frozen.
4. Record warnings per 1,000 benign navigations, not only balanced accuracy metrics.
5. Add privacy-preserving secondary evidence for the review tier and test fail-open behavior when it is unavailable.
6. Conduct security review, permission minimization, adversarial testing, accessibility checks and signed-store packaging.

Until those gates pass, describe it as a **working local research prototype and deployment candidate**, not deployment-ready software.
