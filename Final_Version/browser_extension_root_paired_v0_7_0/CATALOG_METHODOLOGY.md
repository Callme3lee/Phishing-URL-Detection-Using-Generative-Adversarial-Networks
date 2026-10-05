# Reviewed-site catalogue methodology

## Purpose

The URL model can assign high scores to long, unfamiliar login and service URLs. This catalogue reduces visible false warnings on explicitly reviewed first-party pages while preserving the model output for analysis. It does not alter model weights, thresholds or training data.

## Candidate selection

The project uses the first 250 entries of Tranco list XN6XN. Tranco generated this list on 5 September 2026 by aggregating rankings from 7 August to 5 September 2026. The stored CSV is an immutable research snapshot with SHA-256 `e1df51074244cb64c8ae2d5a23cb7f907f113dcd00cd70938f5e22b6a1cceea9`.

Popularity only identifies candidates. It never creates a quiet policy automatically. CDN, cloud, analytics, advertising, DNS and other infrastructure domains remain untrusted even when highly ranked.

## Review requirements

A quiet entry must name an exact ASCII HTTPS hostname, its registered domain, its frozen Tranco rank, allowed paths, excluded paths when required, and a written review basis. Hostnames cannot overlap between rules. The registered-domain and rank relationship is checked during tests and again when the extension loads the catalogue.

The runtime rejects credentials, explicit ports, whitespace/control characters, Unicode hostnames and malformed or ambiguous paths before applying a rule. Host matching is exact. `google.com.attacker.example`, `evil.chatgpt.com` and visually similar internationalised names do not inherit a rule.

Known redirect routes and selected shared-content/authorization routes are excluded. Each destination after a redirect and each single-page application history change receives a new decision. A catalogue loading or integrity failure falls back to the ordinary model policy.

## Measurement

Schema 7 stores the raw model tier, visible decision tier, decision reason and catalogue rule separately. It starts a new local measurement period for the reviewed Google-document query correction while preserving an older export under the legacy record. `servicePageDecisions` counts unique automatic URLs made quiet by a reviewed rule. This value measures policy use, not correct classifications. Labelled evaluation must report raw-model and final browser-policy results separately.

Ordinary collaborative Google document editor/viewer routes keep the raw model tier but use a non-blocking capped notice when no query is present or every query name belongs to the reviewed list. If the query-free diagnostic is Allow or Review, the notice is reduced further to Review suggested. Public Forms, Sites, Drive shares, published presentations, redirect-like or unknown query names, and look-alike hosts remain under the normal risk-preserving policy.

## Updating

Create a new catalogue identifier and measurement schema for every rule or ranking change. Save the permanent Tranco list ID, generation date, source period, top-250 CSV and hashes. Review new candidates manually, add tests for allowed and excluded routes, repeat the Chrome navigation test, and start a new measurement period. Never rewrite an earlier frozen snapshot or merge results across policy versions.
