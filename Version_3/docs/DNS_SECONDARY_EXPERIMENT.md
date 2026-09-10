# DNS secondary-verifier experiment

## Research question

Can independent DNS evidence recover phishing URLs that the frozen conservative HYB-MS +10% GAN policy assigns to `secondary_review`, without retraining or changing the primary URL model?

## Controlled design

- The primary model, its three checkpoints, score bands and `0.995319366` binary threshold remained frozen.
- A new development-only cohort contained 16,730 unique registered domains: 12,000 Tranco legitimate proxies and 4,730 PhishTank verified-online phishing URLs.
- All development domains were identity-disjoint from training, prior calibration/external sets and the protected 20K diagnostic cohort.
- Registered domains were deterministically split 60/20/20 into training, calibration and holdout partitions.
- Only `secondary_review` URLs received DNS lookups.
- No phishing page was opened. Collection queried only hostname A/AAAA/CNAME and registered-domain MX/NS records.
- A three-seed random-forest ensemble used 25 DNS-only count, TTL, NXDOMAIN, timeout and existence values. It did not receive URL characters, lexical features, primary scores or DNS response time.
- The secondary threshold `0.504088145` maximized combined calibration recall subject to total calibration FPR no greater than 0.75%.
- The model and threshold were locked before DNS evidence was collected for the 20K cohort.

## Development holdout

| System | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Primary only | 98.52% | 77.75% | 86.92% | 0.469% | 22.25% |
| Primary + DNS | 97.70% | 80.93% | 88.53% | 0.767% | 19.07% |

The locked branch recovered 30 additional phishing URLs and added 7 false positives on the 3,290-row identity-disjoint development holdout.

## Diagnostic 20K result

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary only | 97.55% | 94.83% | 79.85% | 86.70% | 97.52% | 92.88% | 0.483% | 20.15% |
| Primary + DNS | 97.69% | 92.30% | 83.90% | 87.90% | 97.56% | 93.14% | 0.778% | 16.10% |

Among 941 routed URLs, the DNS ensemble had ROC-AUC 87.60% and PR-AUC 69.81%. It escalated 134 rows: 81 phishing and 53 legitimate. The absolute recall gain was 4.05 points (95% Wilson interval 3.27–5.01), while FPR increased 0.294 points (95% Wilson interval 0.225–0.385).

## Interpretation

DNS provides a real independent gain: it reduces FNR by 4.05 points and improves F1 by 1.20 points without changing the primary URL model. It should not be treated as a hard-block signal by itself. At 1% phishing prevalence, projected precision falls from 62.53% for the primary to 52.14% for the combined binary policy because additional false alarms have greater weight.

The appropriate browser action is therefore:

- retain the frozen URL block/warning tiers;
- use positive DNS evidence to escalate `secondary_review` to an analyst/user review or stronger caution;
- do not silently block solely because DNS is absent or young-looking;
- cache DNS results and define a strict timeout/failure policy before latency testing.

## Limitations

- DNS is mutable. TTL, record absence and NXDOMAIN can change after collection.
- Record absence may reflect takedown timing, resolver filtering or temporary operational failure instead of an intrinsic phishing property.
- PhishTank and Tranco remain source proxies; the holdout is identity-disjoint but not temporally newer.
- The 20K labels were already viewed in the earlier threshold study, so this application is diagnostic rather than a fresh confirmation claim.
- Batched concurrent collection time is not browser-extension latency.

## Reproduction

```bash
PYTHONPATH=src python -m phishing_gans_v3.collect_dns_secondary_development
PYTHONPATH=src python -m phishing_gans_v3.tiered_url_inference \
  --input data/development/dns_secondary_v1/development_urls.csv \
  --output data/development/dns_secondary_v1/primary_scored.csv
PYTHONPATH=src python -m phishing_gans_v3.collect_dns_evidence \
  --input data/development/dns_secondary_v1/primary_scored.csv \
  --output data/development/dns_secondary_v1/dns_secondary_review.csv
PYTHONPATH=src python -m phishing_gans_v3.train_dns_secondary
PYTHONPATH=src python -m phishing_gans_v3.evaluate_dns_secondary
```

The next confirmatory action is to keep both models and thresholds frozen and evaluate a genuinely later low-prevalence cohort. DNS collection must occur at decision time or through a dated cache, without revising the rule after labels are seen.
