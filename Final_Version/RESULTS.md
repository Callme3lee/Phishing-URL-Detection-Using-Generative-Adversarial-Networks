# Final Version experiment result

## Decision

The pair-mass-preserved candidate is the valid Final Version research model.
It fixes most of the root-slash instability while retaining the same training
identities, model architecture, GAN pool, seeds, calibration cohort, threshold
rule, and evaluation sets used for the Version 3 extension model.

It is **not automatically promoted into extension v0.6.0**. The working
extension remains the deployment control because the candidate trades a small
amount of recall for a very large robustness improvement.

## Data contract

| Item | Value |
|---|---:|
| Original real rows | 367,184 |
| Original MS-HardGAN-C rows | 40,410 |
| Original stored rows | 407,594 |
| Root rows given an equivalent counterpart | 55,262 |
| Stored rows after pairing | 462,856 |
| Effective weighted training mass | 407,594 |
| Effective GAN exposure | 10% |
| Seeds | 42, 43, 44 |

Both representations of every root-only training row are stored. Each receives
sampling factor 0.5, so the pair retains total mass 1.0. This prevents the
augmentation from making root URLs more important merely because they have two
equivalent spellings.

## Final controlled comparison

Thresholds for both ensembles were selected independently on the same frozen
23,100-row calibration cohort by maximizing recall subject to FPR <= 1%.

| Dataset / metric | Version 3 / v0.6.0 | Final candidate | Change |
|---|---:|---:|---:|
| Internal accuracy | 97.201% | 96.720% | -0.481 pp |
| Internal precision | 99.833% | 99.863% | +0.030 pp |
| Internal recall | 86.936% | 84.636% | -2.300 pp |
| Internal F1 | 92.939% | 91.621% | -1.318 pp |
| Internal FPR | 0.039% | 0.031% | -0.008 pp |
| Internal FNR | 13.064% | 15.364% | +2.300 pp |
| External accuracy | 84.063% | 83.487% | -0.576 pp |
| External precision | 97.299% | 97.248% | -0.051 pp |
| External recall | 61.032% | 59.553% | -1.479 pp |
| External F1 | 75.012% | 73.870% | -1.142 pp |
| External ROC-AUC | 91.504% | 91.522% | +0.018 pp |
| External PR-AUC | 90.935% | 90.776% | -0.159 pp |
| External FPR | 1.092% | 1.086% | -0.006 pp |
| External FNR | 38.968% | 40.447% | +1.479 pp |
| Root slash class-flip rate | 60.348% | 1.595% | -58.753 pp |
| Root mean absolute score drift | 0.513 | 0.043 | -91.7% |

The candidate reduces class flips by approximately 97.36% relative to Version
3. External ROC-AUC is effectively preserved and FPR is slightly lower, but
external recall falls by 1.48 percentage points. This is a genuine trade-off,
not an across-the-board improvement.

## Rejected intermediate candidate

The first candidate gave every stored row equal sampling weight after pairing.
That unintentionally increased the total influence of root identities. It
reduced the flip rate to 1.58%, but external FNR increased to 41.12% and FPR to
1.35%. It is retained only as experiment evidence. The pair-mass-preserved
candidate is methodologically cleaner and performs better.

## Recommendation

Keep extension v0.6.0 unchanged for the current supervised demonstration unless
the supervisor prioritizes representation robustness over the 1.48-point
external recall difference. The Final candidate is suitable as evidence that
the slash shortcut can be corrected without a major architecture change. A
production replacement would require a fresh, genuinely independent evaluation
and an explicit acceptance criterion for the robustness-versus-recall trade-off.
