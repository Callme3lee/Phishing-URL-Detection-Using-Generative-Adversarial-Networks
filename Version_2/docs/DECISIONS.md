# Version 2 Decision Log

## 2026-08-11 — Distribution-shift attribution

- Established that the shift precedes augmentation because every real-only
  family also fails externally.
- Found a reversed source shortcut: 89.0% of training phishes but only 3.4% of
  training benign URLs are root-like, whereas every external legitimate proxy
  is a root URL and current PhishTank URLs contain more paths and queries.
- Found complete acquisition-format separation (`://` in 100% external versus
  approximately 0.01% training), but a no-retraining canonicalization ablation
  showed scheme removal alone cannot repair performance.
- Measured each GAN as much closer to old training phishing than to PhishTank;
  classified augmentation as a secondary amplifier, not the initiating cause.
- Reclassified the August 11 snapshot as diagnostic for future work; a corrected
  model requires a new final prospective snapshot.

## 2026-08-11 — Isolated hybrid character/lexical CNN

- Added a fifth classifier without modifying any original model, checkpoint,
  dataset row, split, seed, or threshold rule.
- Fused the existing character CNN with a 28-feature lexical MLP; fitted
  feature normalization on real training rows only.
- Trained real-only checkpoints on CUDA for seeds 42/43/44 with thermal
  protection and validation-only checkpoint selection.
- The hybrid became the best internal real-only model by mean accuracy, recall,
  F1, ROC-AUC, and FNR; LSTM retained slightly better precision and FPR.
- A one-time evaluation on the already-frozen PhishTank/Top-Million snapshot
  remained poor, so the hybrid does not resolve external source shift and the
  snapshot remains prohibited for tuning.

## 2026-08-11 — Prospective PhishTank / Top-Million evaluation

- Froze 250 verified-online PhishTank URLs and 250 root-URL legitimate proxies
  from the local Top-Million CSV after model training and selection.
- Excluded exact, encoded, and hostname overlap with the master data and URL or
  hostname overlap with the previous external snapshot.
- Evaluated all four existing conditions, four classifier families, and seeds
  42/43/44 without retraining or external threshold calibration.
- Found severe source-shift failure modes: SVM flags every legitimate URL,
  Random Forest misses most phishing URLs, and no CNN/LSTM condition provides
  an acceptable recall/FPR balance.
- Kept this snapshot evaluation-only and did not use its results to select or
  tune a deployment model.

## 2026-08-10 — Controlled quality-selection experiment

- Kept SeqGAN as the common synthetic source so generator identity could not
  confound random, realism-filtered, and challenging 10% conditions.
- Froze exactly 228,425 synthetic rows per augmented condition and retained all
  2,055,821 real training rows.
- Calibrated every frozen classifier score to an empirical percentile before
  combining difficulty evidence; raw heterogeneous scores were never averaged.
- Limited every synthetic pool to at most 3% from any of 64 lexical clusters.
- Froze a disjoint 20,000-URL evaluation-only challenge before training.
- Evaluated both common real-only thresholds and the existing per-condition
  validation-threshold protocol. The first exposes recall/FPR trade-offs; the
  second preserves the established approximately 1% FPR operating policy.
- Found a strong held-out synthetic robustness gain from challenging selection,
  but no universal real-test advantage. Kept the real-only LSTM as the safest
  browser-extension model.

## 2026-08-10 — Direct generated-URL challenge evaluation

- Kept the existing real-only and 10/20/30/40% augmentation experiments
  unchanged; stopped treating an accuracy increase as a required outcome.
- Evaluated 100,000 accepted unique URLs per GAN against every frozen real-only
  classifier checkpoint using its unchanged validation-selected threshold.
- Reported recall/detection and FNR/evasion because accuracy, precision, FPR,
  and ROC-AUC are undefined on an intended-phishing-only challenge set.
- Added consensus and lexical realism checks so malformed outliers are not
  mistaken for high-quality evasive phishing.
- Found partial evasion rather than classifier failure. RelGAN was hardest but
  also had the largest lexical mismatch; future selection must require both
  challenge and plausibility.

## 2026-08-09 — Project reset

- Archived all previous visible project content under `Version_1`.
- Chose `urls_dataset2M.csv` as the only Version 2 real development source.
- Copied the source into Version 2 and verified identical SHA-256.
- Prohibited Version 1 artifacts and other CSVs from Version 2 model fitting.
- Defined hostname-grouped deterministic splits shared by every condition.
- Selected CNN, LSTM, Random Forest, and linear SVM as real-only baselines.
- Selected Gumbel-LSGAN and SeqGAN as the two primary URL generators.
- At reset time, staged a compact RelGAN architecture and config as future
  work, initially disconnected from training and excluded from results.
- Defined 10% accepted generated URLs as the primary augmentation dose; 20–40%
  are exploratory follow-ups.

## 2026-08-10 — Exploratory RelGAN training

- Activated the staged RelGAN as an exploratory third generator after explicit
  approval to train it.
- Used CUDA and the same seeded 200,000-row malicious view; bounded steps
  consumed 105,600 distinct training examples.
- Completed three MLE warm-up and three adversarial epochs.
- Selected adversarial epoch 3 by the unchanged checkpoint-quality formula.
- Kept RelGAN out of augmented classifier training pending full-corpus
  generation, strict filtering, and an approved screen-then-confirm protocol.

## 2026-08-10 — RelGAN corpus and augmentation screen

- Generated two million raw RelGAN URLs with resumable 20,000-row RNG
  checkpoints and automatic GPU thermal guarding.
- Strictly accepted 1,989,755 unique, valid, non-memorized URLs.
- Built hashed nested 10%, 20%, 30%, and 40% augmentation manifests.
- Screened the predeclared 10% condition with CNN, LSTM, Random Forest, and
  linear SVM at seed 42.
- Promoted no classifier: none improved accuracy, precision, recall, F1, and
  ROC-AUC together. Therefore seeds 43/44 and higher doses were not run.

## 2026-08-09 — Master split frozen

- Verified the master source SHA-256 before reading any rows.
- Retained 2,563,926 normalized exact-unique URLs from 2,568,184 input rows.
- Removed 1,642 duplicate 128-token encodings, leaving 2,562,284 model-unique rows.
- Froze 2,055,821 train, 254,298 validation, and 252,165 test rows.
- Restricted GAN fitting to the 1,186,594 malicious rows indexed inside the
  frozen training split; the implementation may cap each epoch's sampled view
  for laptop-GPU tractability, but cannot draw from another real dataset.
- Made prepared data immutable by default. Rebuilding now requires an explicit
  `--force` flag and must reproduce the recorded manifest hashes.
