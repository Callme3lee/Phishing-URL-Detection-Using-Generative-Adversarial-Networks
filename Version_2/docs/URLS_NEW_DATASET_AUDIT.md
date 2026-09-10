# `URLs_New.csv` Dataset Audit

## Decision

`URLs_New.csv` contains a good number of distinct URLs and hostnames, but it must **not** be used unchanged for training or final evaluation. Its URL-scheme distribution is an extremely strong label shortcut: 99.99% of legitimate rows use HTTPS, compared with only 6.21% of phishing rows.

The file has therefore been audited but has not been added to the Version 2 training pipeline.

## Dataset identity

- File: `URLs_New.csv`
- SHA-256: `18067bad500580a7def376c5a4438c9886adda5eb739fc209fdda4c6309966a7`
- Schema: `url,label`
- Total rows: 450,176
- Legitimate: 345,738 (76.80%)
- Phishing: 104,438 (23.20%)
- Valid after parsing: 450,165
- Invalid: 11

## Diversity findings

| Measurement | Result | Interpretation |
|---|---:|---|
| Raw unique URLs | 450,176 | Every raw string is distinct |
| Unique normalized URLs | 450,030 | 135 duplicate rows appear after canonicalization |
| Unique 128-character model inputs | 446,272 | 3,893 rows collapse to another encoded input |
| Unique hostnames | 166,984 | Broad hostname coverage |
| Legitimate hostnames | 111,310 | Good raw breadth, although popular hosts repeat |
| Phishing hostnames | 55,726 | Good phishing-host breadth |
| Cross-label hostnames | 52 | These hosts cover 10,720 rows and require group-aware splitting |
| Existing-master overlap, scheme-insensitive | 24,366 unique URLs | These are not new relative to the Version 2 master |
| Existing-master hostname overlap | 29,305 hostnames / 157,311 rows | A large subset shares hosts with existing data |

Phishing rows are not dominated by a single host: the largest phishing host contributes 0.94%, and the ten largest contribute 2.43%. Legitimate rows are more concentrated: the largest host contributes 3.66%, and the ten largest contribute 14.53%.

## Critical label shortcut

A deliberately trivial rule—predict legitimate for HTTPS and phishing otherwise—produces the following results on the valid rows:

| Metric | Result |
|---|---:|
| Accuracy | 98.55% |
| Phishing precision | 99.96% |
| Phishing recall | 93.79% |
| False-positive rate | 0.010% |
| TN / FP / FN / TP | 345,702 / 36 / 6,483 / 97,944 |

This performance requires no learned phishing pattern. It demonstrates that the labels are strongly coupled to how the URLs were collected or formatted. A high score from a CNN, LSTM, RF, or SVM trained directly on this file could therefore be misleading.

## Additional limitations

- Rows are sorted into two contiguous label blocks. A sequential split would be invalid.
- The CSV has no source column, so independent-source diversity cannot be measured.
- The CSV has no collection date, so recency and chronological diversity cannot be verified.
- There are 817 label disagreements among scheme-insensitive URLs that overlap the existing Version 2 master. These need individual investigation before merging.
- The class distribution is moderately imbalanced at 76.8% legitimate and 23.2% phishing.

## Safe preparation before use

1. Remove the 11 invalid rows.
2. Apply the same canonicalization as Version 2 and deduplicate at both normalized-URL and 128-character encoded-input levels.
3. Investigate the 817 conflicting labels and do not resolve them automatically.
4. Neutralize the scheme shortcut. The cleanest controlled option is to exclude scheme tokens from model input and lexical features, or create balanced HTTP/HTTPS strata whose proportions are not label-dependent.
5. Split by normalized hostname, keeping all rows from a hostname in exactly one split. Do not use a random row split.
6. Keep all URLs overlapping the current master out of any claimed external test set.
7. Recover source and collection-date metadata if possible. Without them, this file cannot prove temporal or independent-source generalization.
8. Re-run a shortcut audit before training; scheme-only and source-identification rules should no longer classify the labels accurately.

## Overall conclusion

The answer to “does it contain a good amount of different samples?” is **yes at the raw URL and hostname levels**. It provides 446,272 distinct model inputs and 166,984 hostnames, including 55,726 phishing hostnames. However, it is **not yet a clean, independent training or test dataset**. Its severe scheme-label artifact would inflate internal metrics and could worsen the distribution-shift problem already identified in Version 2.

The reproducible audit is implemented in `src/phishing_gans_v2/audit_new_dataset.py`; its machine-readable output is stored in `artifacts/results/new_dataset_audit/audit.json`.
