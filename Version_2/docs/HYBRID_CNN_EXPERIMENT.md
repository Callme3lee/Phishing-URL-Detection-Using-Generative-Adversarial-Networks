# Hybrid Character/Lexical CNN Experiment

## Controlled addition

The hybrid CNN was added as a fifth real-only classifier. Nothing in the
existing experiment was replaced or retrained. It uses the same 2,055,821 real
training rows, frozen hostname-separated validation/test sets, seeds 42/43/44,
eight-epoch maximum, patience-2 early stopping, class weighting, and threshold
rule requiring validation FPR at or below 1%.

The model has two branches:

- the existing character embedding and parallel CNN kernels over each
  128-token URL; and
- a small MLP over the existing 28 lexical/structural features.

The two representations are concatenated and classified by a 128-unit fusion
layer. Lexical mean and standard deviation were fitted on the real training
split only. No generated URLs were used.

## Frozen internal test results

Values are mean ± sample standard deviation across seeds 42/43/44.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Character CNN | 0.963503 ± 0.000657 | 0.992540 ± 0.000276 | 0.943149 ± 0.001230 | 0.967214 ± 0.000611 | 0.996119 ± 0.000059 | 0.009428 ± 0.000357 | 0.056851 ± 0.001230 |
| Character LSTM | 0.964454 ± 0.002135 | **0.992730 ± 0.000406** | 0.944645 ± 0.003560 | 0.968089 ± 0.001973 | 0.996460 ± 0.000083 | **0.009200 ± 0.000502** | 0.055355 ± 0.003560 |
| Random Forest | 0.950569 ± 0.000068 | 0.991746 ± 0.000118 | 0.921068 ± 0.000065 | 0.955101 ± 0.000060 | 0.987961 ± 0.000008 | 0.010195 ± 0.000147 | 0.078932 ± 0.000065 |
| Linear SVM | 0.931315 ± 0.000000 | 0.991858 ± 0.000000 | 0.886951 ± 0.000000 | 0.936476 ± 0.000000 | 0.963661 ± 0.000000 | 0.009683 ± 0.000000 | 0.113049 ± 0.000000 |
| Hybrid CNN | **0.966536 ± 0.000170** | 0.992233 ± 0.000157 | **0.948802 ± 0.000161** | **0.970032 ± 0.000153** | **0.996694 ± 0.000066** | 0.009877 ± 0.000200 | **0.051198 ± 0.000161** |

Relative to the previous best-F1 LSTM, the hybrid gains 0.208 percentage
points accuracy, 0.416 points recall, 0.194 points F1, and 0.023 points ROC-AUC.
It loses 0.050 points precision and raises FPR by 0.068 points. Thus the hybrid
is the strongest internal real-only model overall, but the LSTM remains slightly
better when minimizing false positives is the dominant objective.

With only three paired seeds, the 95% paired intervals for hybrid-minus-LSTM
accuracy, recall, and F1 cross zero. The ROC-AUC change is +0.000234 with a
paired 95% interval of [+0.000077, +0.000391], the only requested improvement
whose interval excludes zero. The other favorable means should therefore be
described as promising and consistent in direction, not definitive proof.

Best validation epochs were 5, 6, and 6. The checkpoint is approximately
567 KB and test inference averaged roughly 86,000 URLs/second on the RTX 3060
at batch size 1024.

## Frozen external validity check

After checkpoint selection, the hybrid was evaluated once on the existing
evaluation-only 500-URL PhishTank/Top-Million snapshot. Its unchanged internal
thresholds produced:

| Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.461 ± 0.082 | 0.488 ± 0.116 | 0.413 ± 0.153 | 0.426 ± 0.061 | 0.415 ± 0.088 | 0.491 ± 0.298 | 0.587 ± 0.153 |

The hybrid therefore improves the unchanged internal test but does not solve
the already-observed temporal/source distribution shift. It must not be tuned
on this external snapshot.

Machine-readable results, checkpoints, and graphs are in
`artifacts/hybrid_cnn/` and `artifacts/results/hybrid_cnn/`.
