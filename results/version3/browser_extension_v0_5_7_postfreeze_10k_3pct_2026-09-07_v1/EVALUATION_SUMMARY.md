# Phishing URL Guard v0.5.7 — latest-snapshot evaluation

## Evaluation set

- Dataset: `data/evaluation/postfreeze_10k_3pct_2026-09-01/evaluation_urls.csv`
- Total URLs: 10,000
- Legitimate URLs: 9,700
- Phishing URLs: 300 (3%)
- Labels: Tranco legitimate-domain proxies and PhishTank verified-online phishing URLs
- Dataset SHA-256: `d5621a79f5b0257a2a4cdf2f56bed8550df328c4f13f6e535a91a78e4785db43`

The dataset was locked before inference and is identity-disjoint from earlier project URL files under the collection report's identity checks.

## Primary result

The primary phishing decision is **warning or block** at the unchanged threshold of `0.9953077435493469`.

| Metric | Result |
|---|---:|
| Accuracy | 98.91% |
| Precision | 85.77% |
| Recall | 76.33% |
| F1-score | 80.78% |
| ROC-AUC | 96.61% |
| PR-AUC | 87.10% |
| False-positive rate | 0.39% |
| False-negative rate | 23.67% |

Confusion matrix: TN = 9,662, FP = 38, FN = 71, TP = 229.

## Extension tiers

| Tier | Legitimate | Phishing | Total |
|---|---:|---:|---:|
| No warning | 8,872 | 16 | 8,888 |
| Review suggested | 708 | 26 | 734 |
| Use caution | 82 | 29 | 111 |
| High risk | 35 | 74 | 109 |
| Very high risk | 3 | 155 | 158 |

Normal mode pauses only the **very high risk** tier. At this strict interruption point, precision is 98.10%, recall is 51.67%, FPR is 0.03%, and FNR is 48.33%. The lower tiers still provide passive warning or review information without blocking the page.

The exact v0.5.7 reviewed-service policy was also checked against all 10,000 raw browser-form URLs. No row matched a reviewed quiet-list rule, so the catalogue did not change this snapshot's confusion matrix.

## Assessment

The extension is technically functioning and has a low false-positive rate at its primary warning threshold. Its precision, accuracy, ROC-AUC and PR-AUC are strong for this test. However, recall is 76.33%, leaving 71 of 300 phishing URLs below the warning threshold. Therefore, v0.5.7 is suitable as a **graduation-project research demonstration and user evaluation build**, but these results do not support claiming that it is a production-ready standalone phishing blocker.

The interface update does not alter these metrics. It improves usability and reduces the chance that a user permanently allows a suspicious hostname by mistake:

- permanent allow is offered only after a warning or review result;
- the exact hostname is displayed;
- the consequence of the permanent local exception is explained;
- confirmation remains disabled until the user acknowledges the hostname;
- visiting once remains a separate, safer option.
