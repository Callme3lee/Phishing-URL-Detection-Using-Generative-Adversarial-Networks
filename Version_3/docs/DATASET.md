# Version 3 Dataset

## Source and objective

The sole real dataset is `URLs_New.csv`: 450,176 input rows containing 345,738 legitimate and 104,438 phishing URLs. Version 3 does not mix it with the old 2M dataset. The purpose of preprocessing is to prevent models from exploiting acquisition and formatting artifacts while preserving useful URL text and structure.

## Why scheme removal is mandatory

Before cleaning, 99.99% of legitimate rows were HTTPS, compared with 6.21% of phishing rows. Predicting “HTTPS = legitimate” achieved 98.55% accuracy without learning phishing. Version 3 therefore removes schemes from character sequences and removes the HTTPS indicator from lexical features.

## Cleaning results

| Stage | Rows |
|---|---:|
| Input | 450,176 |
| Invalid URL rows | 11 |
| Unsupported scheme rows (`httpss`/FTP) | 43 |
| Version 2 exact label disagreements quarantined | 817 |
| Same-label duplicates after scheme-free canonicalization | 685 |
| Retained after canonical deduplication | 448,620 |
| Duplicate 128-character inputs removed | 1,284 |
| Cross-split encoded collisions removed | 5 rows across 2 collision groups |
| Final unique model inputs | **447,331** |

The quarantine CSV contains 2,845 records after canonical and encoded auditing. This is larger than the number of excluded model inputs because both sides of ambiguous encoded collisions are recorded for review.

## Frozen splits

| Split | Total | Legitimate | Phishing | Unique hostnames |
|---|---:|---:|---:|---:|
| Train | 358,619 | 276,400 | 82,219 | 133,208 |
| Validation | 39,483 | 29,535 | 9,948 | 16,540 |
| Test | 49,229 | 38,908 | 10,321 | 16,600 |

All hostname intersections between train, validation, and test are empty. Fractions are approximate by row because assignment is performed by hostname group, not by individual row.

## Remaining limitation

The scheme shortcut is eliminated, but a decision stump using only hostname dot count reaches 76.96% balanced accuracy on the test split. This may combine a genuine phishing property with collection-source bias. It has been retained and disclosed rather than automatically erased. Final model claims must therefore include genuinely independent, later-collected evaluation data.

Machine-readable evidence:

- `data/processed/preparation_report.json`
- `data/processed/verification_report.json`
- `artifacts/data_audit/shortcut_audit.json`
- `data/processed/quarantine.csv`
