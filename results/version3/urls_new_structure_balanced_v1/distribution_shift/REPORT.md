# URLs_New-Only Structure Balancing — Distribution Shift

Structural total-variation distance (TVD) ranges from 0 to 1; lower is closer. Source-classifier ROC-AUC ranges from 0.5 (indistinguishable) to 1.0 (perfectly separable).

| External set | Class | Structure TVD original | Structure TVD balanced | Source AUC original | Source AUC balanced | Mean abs. SMD original | Mean abs. SMD balanced |
|---|---|---:|---:|---:|---:|---:|---:|
| topdomain_phishtank | legitimate | 0.8644 | 0.7500 | 0.9944 | 0.9907 | 0.7819 | 0.7733 |
| topdomain_phishtank | phishing | 0.6244 | 0.6244 | 0.9067 | 0.9067 | 0.3208 | 0.3208 |
| tranco_openphish | legitimate | 0.8644 | 0.7500 | 0.9972 | 0.9952 | 0.8040 | 0.7954 |
| tranco_openphish | phishing | 0.7344 | 0.7344 | 0.9342 | 0.9342 | 0.3476 | 0.3476 |

Balancing changes only legitimate training proportions. All phishing training rows are retained, so phishing-side shift should remain essentially unchanged.

A lower structural distance does not guarantee low external FPR because hostname vocabulary, collection source, TLDs, and other lexical relationships can remain separable.
