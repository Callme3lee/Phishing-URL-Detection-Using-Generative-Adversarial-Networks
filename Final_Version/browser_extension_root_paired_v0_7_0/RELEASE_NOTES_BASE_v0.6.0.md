# v0.6.0 — Explainable URL signals

This release preserves the frozen HYB-MS + 10% GAN three-seed ensemble and the v0.5.9 thresholds. No classifier was retrained and no evaluation result was changed.

Suspicious results now show concise diagnostic signals derived from the same 35 lexical inputs supplied to the hybrid model. Continuous features are flagged only when they are substantially above the frozen training median relative to its interquartile range. Direct checks cover IP hosts, Punycode and explicit ports. The existing query-normalized second score can also identify when query values materially increase the score.

These signals are intended to help analyse false positives. They are not SHAP values, causal feature attributions, or proof that a website is phishing. When no individual lexical feature crosses the diagnostic threshold, the interface states that only the combined URL pattern was unusual.

Signal codes are saved with consented local assessment and feedback records. Raw URLs are still not stored for automatic checks, and nothing is uploaded automatically.

Google Workspace handling was refined after a controlled audit found that blanket trust for shared Docs and Drive content hid phishing URLs. Fixed Google service pages remain quiet. Ordinary collaborative Google document editor/viewer routes now keep the raw model tier but use a non-blocking capped notice when their query names are reviewed. Query-caused warnings with a low query-free diagnostic become Review suggested. Public Forms, Sites, Drive shares, published presentations, redirects, look-alikes, unknown parameters, and other high-risk shared content retain blocking in Normal mode. The original controlled audit recovered 24 phishing detections, reduced FNR from 21.33% to 13.33%, and did not change FPR (0.69%); this later usability correction requires a separately labelled evaluation and does not rewrite that result.

In-page notices now show only the result, hostname and risk score. Detailed lexical reasons remain available in the extension popup and full warning page, reducing clutter while preserving explainability when requested.
