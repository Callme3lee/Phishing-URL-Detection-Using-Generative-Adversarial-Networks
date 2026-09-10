# Dataset card and access contract

The repository does not redistribute the full URL datasets. This avoids publishing participant browsing data, live malicious URLs, or third-party data without confirming redistribution rights. The code expects the following frozen inputs.

## Core datasets

| Dataset | Expected location | SHA-256 | Size | Role |
|---|---|---|---:|---|
| `urls_dataset2M.csv` | `Version_2/data/raw/urls_dataset2M.csv` and `Version_1/Data sets/urls_dataset2M.csv` | `f934406ec02c317b6eccab8c4f4cf7ac080e8981ececa25fbca863d79239defd` | 136,203,062 bytes | Version 1 and Version 2 real-data experiments |
| `URLs_New.csv` | repository root | `18067bad500580a7def376c5a4438c9886adda5eb739fc209fdda4c6309966a7` | 32,312,601 bytes | Version 3 frozen real-data source |

The two Version 1/2 paths refer to byte-identical copies. A local hard link or symbolic link may be used instead of storing the same file twice.

## Version 3 labels and cleaning

- Input rows: 450,176
- Input labels: 345,738 legitimate and 104,438 phishing
- Final unique model inputs after validation, canonicalisation, conflict handling, and deduplication: 447,331
- Frozen split: 358,619 train, 39,483 validation, and 49,229 test
- Label mapping: legitimate `0`, phishing `1`
- Split unit: canonical hostname
- Seeds used for final classifier comparisons: 42, 43, and 44

Schemes and user information are removed before Version 3 feature extraction. This prevents the model from exploiting a strong source-specific HTTPS shortcut. Full details are in `Version_3/docs/DATASET.md`.

## External evaluation sources

The project used timestamped or frozen samples derived from sources including PhishTank, OpenPhish, Tranco, and Common Crawl. These sources have different label meanings and usage conditions. Ranked domains and successful archived pages are treated as legitimate proxies, not independent proof that every URL is harmless.

External evaluation rows are not included in this public package. Their collection protocols, counts, limitations, and aggregate results remain documented in the version reports.

## Generated data

Generated URLs are synthetic phishing-style strings created for defensive research. They are labelled separately from real URLs and are filtered for validity, duplicate overlap, lexical plausibility, and diversity before controlled augmentation. Synthetic data is never used as external ground truth.

## Privacy and security

- Participant browsing exports are excluded.
- Row-level prediction tables are excluded.
- Query strings, credentials, and personal identifiers must not be published.
- Live phishing links must not be opened for demonstrations.
- Dataset access does not imply permission to redistribute its contents.

Use `artifacts_manifest.json` and `scripts/verify_artifacts.py` to verify locally obtained copies.
