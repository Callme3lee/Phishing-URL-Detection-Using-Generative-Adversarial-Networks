# Legitimate Structural-Coverage Development Pool — 2026-08-11

## Why this pool exists

The frozen external evaluation showed two different facts:

- The CNN, LSTM, and RNN recalled 95.9–96.7% of the 250 current PhishTank URLs.
- The same models falsely classified 98.7–99.9% of the 250 unseen Top-Domain roots as phishing.

Therefore, the immediate deployment failure is dominated by missing legitimate structural coverage. The phishing result is encouraging, but it is not proof that phishing detection is solved: PhishTank is one source, the sample is small, and high recall is less meaningful while the models predict almost everything as phishing.

This dataset is a **development pool for repairing legitimate coverage**. It is not a replacement external test, and it has not yet been merged into the frozen Version 3 training set.

## Frozen pool

| Property | Value |
|---|---:|
| Rows | 3,800 |
| Root URLs | 950 |
| Shallow-path URLs | 950 |
| Deep-path URLs | 950 |
| Query URLs | 950 |
| Unique normalized URLs | 3,800 |
| Unique 128-token inputs | 3,800 |
| Unique hostnames | 1,197 |
| Unique Tranco domains | 1,178 |
| Dataset SHA-256 | `3c04291c2d396fd14e96b12cf9c9fa2533ca33ea98cc264b49f36da5c1d0bc56` |

The root URLs come from Tranco list `Q2XX4`, generated on 2026-08-10. Archived non-root URLs come from the Common Crawl `CC-MAIN-2026-30` URL index and were restricted to HTTP 200 HTML records. The pool accepts only the listed pay-level hostname or its `www` form.

The public Common Crawl index began disconnecting and rate-limiting during collection. Instead of repeatedly pressuring the service or changing the sample after inspecting model results, collection stopped at the largest fully cached, exactly balanced pilot: 950 rows in each structural category. The cached responses make this freeze auditable.

## Label meaning and limitations

Every row has label `0`, but its precise label semantics are **popular-domain legitimate proxy**. Tranco membership and an archived HTTP 200 response do not independently prove that every historical path is harmless. Popular sites can be compromised, blocklists are incomplete, and Common Crawl is not a security adjudication service.

Consequently:

- report these rows as proxy-labelled development data, not manually verified benign ground truth;
- never use them as the final external evaluation set;
- inspect a sample manually before promotion into a new training freeze;
- retain source, rank, crawl timestamp, and Common Crawl digest for every row;
- repeat blocklist screening immediately before any later promotion.

## Leakage and safety audit

The automated audit passed every check:

- no exact URL, 128-token input, or hostname overlap with the original Version 3 dataset;
- no exact URL or hostname overlap with the frozen 2026-08-11 external evaluation;
- no exact URL or hostname overlap with the downloaded 300-URL OpenPhish community snapshot;
- no duplicates inside the pool;
- all URLs reproduce their stored canonical URL and hostname;
- all Tranco domains remain within a single proposed split.

The original collection also screened against all Version 3 phishing rows and the current PhishTank verified-online feed. URLhaus was not treated as a phishing source: it focuses on malware distribution and its bulk API requires an Auth-Key. Absence from any feed must not be interpreted as proof of legitimacy.

## Grouped development manifests

The proposed split is deterministic from the Tranco pay-level domain using seed 42. This prevents the same site from appearing in multiple splits, even when it contributes several paths or both bare and `www` hostnames.

| Split | Rows | Tranco domains | Hostnames |
|---|---:|---:|---:|
| Train | 3,042 | 946 | 960 |
| Validation | 389 | 113 | 115 |
| Test | 369 | 119 | 122 |

These are development manifests only. The final experiment must retain the original Version 3 validation/test sets and must not use the frozen external snapshot for selection or threshold tuning.

## What should happen next

Run one pre-registered real-data coverage experiment:

1. Manually review a stratified sample from each structural category.
2. Freeze the accepted rows and their screening-feed hashes.
3. Add only the accepted **training-manifest** rows to the real training data; leave the original validation/test and external sets unchanged.
4. Retrain CNN, LSTM, RNN, RF, and SVM with the same seeds and hyperparameters.
5. Compare against the original real-only controls on the unchanged internal test.
6. Collect a later, independent legitimate/phishing snapshot and evaluate it once.

Separately, test phishing generalization on at least OpenPhish plus a later PhishTank snapshot. GAN work should resume only after this real-data structural shift is measured; synthetic phishing URLs cannot supply missing legitimate examples.

## Files and reproduction

- Frozen candidates: `data/development/benign_coverage_2026-08-11/benign_candidates.csv`
- Collection manifest: `data/development/benign_coverage_2026-08-11/collection_report.json`
- Verification: `data/development/benign_coverage_2026-08-11/audit/verification_report.json`
- Structural profile: `data/development/benign_coverage_2026-08-11/audit/profile.json`
- Grouped manifests: `data/development/benign_coverage_2026-08-11/audit/{train,validation,test}.csv`

Run the non-network audit from `Version_3`:

```bash
export PYTHONPATH=src
python3 -m phishing_gans_v3.audit_benign_coverage
```

The audit verifies the frozen dataset hash before generating any derived manifest.
