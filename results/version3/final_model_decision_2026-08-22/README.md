# Final frozen-model decision

## Decision

**Retain weighted HYB-MS real-only as the conservative final reference.**

The +10% model clearly won the 30K/5% primary confirmation, but its long-legitimate-URL FPR was 1.45% versus 1.05% for real-only. The strict predeclared rule required no FPR increase on that stress set, so it did not pass the complete two-part selection gate.

This does not erase the GAN result: on the primary set, +10% improved F1 from 79.06% to 81.27%, reduced FPR from 1.26% to 1.01%, and produced a significant paired overall gain (p = 0.0000743). It remains the best ordinary low-prevalence candidate, while real-only is retained as the conservative unconditional reference.

Neither model is declared deployment-ready. No checkpoint, threshold, or dataset was changed after scoring began.
