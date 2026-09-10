# Prospective Evaluation Accumulator

## Objective

Accumulate 300 genuinely post-training, exact-hostname-unseen phishing URLs
without training, calibrating, selecting or repeatedly evaluating against them.
At the target, freeze the first 300 candidates, pair them with 300 unused benign
controls and evaluate the already-frozen checkpoints once.

## Current status

- Target: 300 phishing hostnames.
- Current: 6.
- Remaining: 294.
- Ready to freeze: no.
- Evaluated: no.

The live machine-readable state is
`artifacts/prospective_evaluation/status.json`. The candidate ledger is
`candidates.csv` beside it. The initial six candidates are the genuinely
post-cutoff examples from the 1 August smoke test.

## Leakage and integrity controls

The immutable exclusion index contains 2,149,042 BLAKE2b-64 hashes of canonical
hostnames from:

- all three original project datasets;
- the 31 July frozen evaluation;
- every modern malicious train, validation and temporal-test URL;
- all 50,000 modern benign hard negatives; and
- all 4,378 accepted generated LSGAN training URLs.

The index and the SHA-256 hashes of every input are recorded in
`excluded_hostname_hashes.metadata.json`. The collector additionally rejects
duplicate URLs and hostnames already present in the prospective ledger. It
stores immutable raw API responses and response headers under timestamped
folders. It reads URL strings only and never visits them.

Repeated import of the same six examples was tested and added zero rows. The
one-time freezer was also tested below the target and correctly refused to
create an evaluation directory.

## Collection command

Run once per day:

```bash
./scripts/collect_prospective_snapshot.sh
```

A user cron entry is installed to run this command automatically every day at
14:15 Africa/Tripoli local time. Output is appended to
`artifacts/prospective_evaluation/collector.log`. The installed source file is
`scripts/prospective_evaluation.cron`; installing it again is safe only after
checking that it will not replace unrelated user cron entries.

Each run makes five read-only requests of 100 records to the official PhishStats
API, safely below the documented anonymous daily quota. The API documentation
states that the source refresh cycle is approximately 90 minutes, so running
more frequently provides little value.

The command updates `status.json`. Do not run any classifier evaluation while
`ready_to_freeze` is false.

## One-time freeze

When the status reaches at least 300, download a fresh Tranco list and run:

```bash
PYTHONPATH=src python3 -m phishing_gans.freeze_prospective_evaluation \
  --candidates artifacts/prospective_evaluation/candidates.csv \
  --exclusions artifacts/prospective_evaluation/excluded_hostname_hashes.npy \
  --tranco-zip artifacts/prospective_evaluation/final_tranco_top_1m.csv.zip \
  --training-data-dir artifacts/data_2m \
  --output-dir artifacts/prospective_evaluation/frozen_target_300 \
  --target 300
```

The freezer uses the first 300 candidates in the deterministic ledger order,
refuses to run below target, and refuses to overwrite an existing frozen output.
Only after that succeeds should the six frozen real-only/LSGAN checkpoints be
evaluated. No threshold adjustment is permitted.

## Files

- `src/phishing_gans/build_prospective_exclusions.py`: exclusion-index builder.
- `src/phishing_gans/collect_prospective_snapshot.py`: append-only collector.
- `src/phishing_gans/freeze_prospective_evaluation.py`: target-enforcing freezer.
- `scripts/collect_prospective_snapshot.sh`: simple daily entry point.
