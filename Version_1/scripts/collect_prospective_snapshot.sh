#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PROJECT_PYTHON:-python3}"

env PYTHONPATH=src "$python_bin" -m phishing_gans.collect_prospective_snapshot \
  --root artifacts/prospective_evaluation \
  --exclusions artifacts/prospective_evaluation/excluded_hostname_hashes.npy \
  --cutoff 2026-08-01T07:22:49+00:00 \
  --target 300 \
  --pages 5 \
  --page-size 100
