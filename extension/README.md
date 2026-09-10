# Phishing URL Guard 0.6.0 — Explainable URL Signals

This release keeps the frozen v0.5.9 model and thresholds unchanged and adds diagnostic URL explanations. The extension popup and full warning page show up to four lexical signals measured from the same 35 inputs used by the hybrid model; the smaller in-page notice stays concise and directs users to the extension for details. The explanations are debugging evidence, not proof of phishing or exact neural-feature attribution. See `RELEASE_NOTES_v0.6.0.md`.

The popup, warning page, Activity page, Offline checker and in-page advisories support English and Arabic. Use the `العربية` or `English` button to switch; the choice is stored locally. Arabic uses right-to-left layout while URLs and hostnames remain left-to-right. The permanent allow action is visible directly on warning, caution and high-risk notices, clearly limited to the exact website name, and remains separate from one-time continuation. It can be reversed later from the popup. When local feedback consent is enabled, choosing allow also records one deduplicated incorrect-warning report locally.

This release moves trusted-service handling into a documented, versioned catalogue. Supported first-party pages show no caution/review banner, warning badge or blocking redirect. The original model score remains available in Technical details. A catalogue decision is a browser policy choice, not a newly learned prediction.

The model, its input features and tier boundaries match the preserved v0.5.7 files. The primary reported detection begins at the existing caution boundary. In Normal mode, High Risk and Very High Risk interrupt navigation, while Caution remains advisory-only. Query URLs also get a diagnostic score with values removed; that score never lowers warnings for unknown hosts. No report or historical user result was changed.

## Install

1. Export the full JSON from the old extension first.
2. Disable the old extension. Two active copies would both scan and show warnings.
3. Extract the release ZIP into a permanent folder.
4. Open the browser's Extensions page, enable Developer mode, select Load unpacked and choose the extracted folder containing manifest.json.
5. Confirm version 0.6.0 in both the Extensions page and the popup. Open Activity and choose whether to keep local detection history.
6. Refresh existing website tabs to load the new content script.

A separately loaded folder normally gets a separate extension ID and starts new local activity. Existing data stays with the old extension: export before removing it. An in-place upgrade retaining the old ID preserves older data in the detailed export. Historical results are not repaired or merged.

Close and reopen any tab still showing an older warning. An older active extension can continue to show its own banner, so leave only one Phishing URL Guard enabled.

## Catalogue design

The candidate source is the frozen Tranco XN6XN top-250 snapshot generated on 5 September 2026. Popularity never creates trust. The snapshot supplies rank and selection metadata; only entries manually added to `policy/site_policy.json` can make a page quiet. The catalogue contains 28 reviewed rule groups and 69 exact hostnames covering Google and Google Workspace, Facebook, Microsoft, YouTube, Apple/iCloud, Instagram, Twitter/X, LinkedIn, Amazon, GitHub, Bing, Wikipedia, WhatsApp, Yahoo, ChatGPT/OpenAI and Reddit.

At startup the extension verifies the shortlist hash, its 250 sequential ranks, every catalogue rule and its Tranco rank. It uses the included Public Suffix List to obtain the registered domain for popularity metadata. Infrastructure domains such as Cloudflare, Amazon Web Services, Google APIs, Gstatic, Akamai and Facebook CDN remain popularity-only entries.

A reviewed fixed-service rule works whether a query is present or absent and regardless of the raw model tier. A supported login URL longer than the model's 4096-character limit receives an explicit unavailable model score and its catalogue policy can still apply. User-created Google Workspace content follows the separate risk-preserving rule below.

Fixed Google Workspace service pages remain quiet, but user-created Docs, Forms, Sheets, Slides, Drawings, shared Drive files/folders and Google Sites do not receive blanket trust. For these pages, model Caution is softened to Review suggested; High Risk and Very High Risk are not downgraded and interrupt navigation in Normal mode. Links leaving those pages, shared ChatGPT pages/custom GPT paths, OAuth consent routes and known outbound redirects receive ordinary model handling where configured. HTTPS, exact hostname and path matching are required. Credentials, nonstandard ports, look-alike domains and arbitrary subdomains receive no exception. Redirect destinations and browser history changes are evaluated independently.

See `CATALOG_METHODOLOGY.md` and `SERVICE_RULE_SOURCES.md` for provenance, maintenance rules and limitations.

## Fixes

- Per-tab navigation tokens and current/pending URL checks reject outdated scan results before presentation or redirect.
- Serialized study updates prevent concurrent status reads, feedback and scans from overwriting measurements.
- Persistent salted URL identities deduplicate across tabs and worker restarts. Different queries remain distinct; fragments and HTTP/HTTPS variants of the same canonical model input do not.
- Feedback uses the same identity system. Repeated feedback updates one retained report; different query variants no longer collapse merely because their exported URLs are sanitized.
- Assessment IDs link feedback to automatic assessments. The background computes scores instead of trusting UI-supplied scores.
- Model block-risk decisions, advisory displays and actual navigation interruptions are counted separately. Passive mode never redirects. Continue once applies only to its warning tab and URL.
- Trust/mode changes refresh tab decisions. Popup scans check again that the page has not changed.
- Pending work cannot add scan measurements after collection is paused/reset. Inference responses have a timeout.
- Sensitive actions require the extension's own control pages; content scripts cannot directly read local study storage.

## Measurement fields

| Field | Meaning |
|---|---|
| uniqueAutomaticUrls | Unique canonical URLs first assessed automatically, not websites or visits |
| decisionCounts | Effective tier at the first unique assessment, not displayed warning counts |
| navigationInterrupts | Successful warning-page navigation requests; repeat visits may interrupt again |
| advisoryDisplays | Unique linked assessments with an advisory iframe-load acknowledgment, not proof a person read it |
| feedbackSubmissions | Successful submissions including repeated presses |
| feedbackReports | Latest retained report per reported URL identity, up to 250 records |
| assessmentId | Automatic-assessment link; null for feedback not yet linked |
| feedbackRecordsEvicted | Older feedback records removed by retention limit |
| untrackedAssessments | Assessments beyond the 25,000-identity capacity |
| servicePageDecisions | Unique automatic decisions made quiet by a recognised service rule |
| modelViewEvaluations | Successful full/normalized ensemble evaluations; each ensemble uses three seed models |

Schema 6 exports record the policy reason and raw model tier separately. Existing model-run fields count scoring requests; modelViewEvaluations counts the successfully scored URL views. Do not silently combine schemas or treat unreported URLs as correct predictions. User claims alone do not establish accuracy, F1, FPR or FNR. Many unique URLs may belong to only a few websites.

## Privacy and limits

Inference is local; there is no automatic upload. Collection is optional. Raw browsing URLs are not persisted in the study, but current-page results are held in transient session caches. Salted fingerprints stay local and are excluded from exports. Deliberately reported URLs have credentials, query and fragment removed, but paths can still contain personal information. Review exports before sharing.

The identity ledger stops adding entries at 25,000 rather than evicting old entries and recounting repeat URLs. The dashboard reports that limit. Feedback and intervention-detail histories retain the latest 250 entries; cumulative interruption totals remain. Export regularly. Reset erases the study and its archive.

This is an asynchronous research warning extension, not a network firewall. A page may begin loading before inference finishes. Classifier errors, long-query weaknesses and distribution shift remain. Low scores do not guarantee safety. Keep browser protection enabled; do not ask testers to visit live phishing pages.

## Tests

Run `npm ci` then `npm test` for preprocessing/ONNX parity, policy and reliability regression checks. `python tests/browser_smoke.py` checks model loading. `python tests/browser_navigation.py` checks actual banners, badges, page changes and a redirect in a disposable Chrome profile using locally fulfilled HTTP responses. It does not contact the test destinations or establish real-site classification metrics. These browser tests require Chrome and websocket-client. Source tests are not bundled in the install-only ZIP.

See RELEASE_NOTES_v0.6.0.md and PILOT_TESTER_GUIDE.md.
