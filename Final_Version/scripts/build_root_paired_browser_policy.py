from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phishing_gans_v3.analyze_threshold_sets import recall_threshold


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = ROOT / "Final_Version/artifacts/results/root_paired_mass_preserved_v1/candidate_scores.npz"
DEFAULT_CHECKPOINTS = ROOT / "Final_Version/artifacts/root_paired_mass_preserved_v1/gan_10pct/hyb_ms"
DEFAULT_OUTPUT = ROOT / "Final_Version/artifacts/deployment_candidates/root_paired_hyb_ms_gan10_v1/policy.json"
CALIBRATION = ROOT / "Version_3/data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze browser thresholds for the root-paired HYB-MS ensemble.")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stored = np.load(args.scores)
    labels = stored["calibration_labels"]
    scores = stored["calibration_scores"]
    caps = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
    names = ("Ultra-strict", "Strict", "Conservative", "Balanced", "Recall-first", "High-recall")
    points = {name: {"maximum_fpr": cap, **recall_threshold(labels, scores, cap)} for name, cap in zip(names, caps)}

    checkpoints = []
    for seed in (42, 43, 44):
        checkpoint = args.checkpoints / f"seed_{seed}/checkpoint.pt"
        checkpoints.append({"seed": seed, "path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint)})

    high_recall = points["High-recall"]["threshold"]
    balanced = points["Balanced"]["threshold"]
    conservative = points["Conservative"]["threshold"]
    strict = points["Strict"]["threshold"]
    payload = {
        "policy_version": "root_paired_hyb_ms_gan10_v1",
        "project_version": "Final Version",
        "model": "Root-paired HYB-MS +10% MS-HardGAN-C, pair-mass-preserved three-seed ensemble",
        "status": "ISOLATED RESEARCH EXTENSION; NOT A SILENT REPLACEMENT FOR VERSION 3",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": checkpoints,
        "ensemble_rule": "arithmetic mean of FP32 sigmoid probabilities from seeds 42/43/44",
        "input_contract": {
            "canonicalization": "Version 3 scheme-free canonical URL processing",
            "lexical_features": "35 PSL-aware lexical values using checkpoint-stored robust scaling",
            "maximum_characters": 128,
            "url_field": "url",
        },
        "selection_evidence": {
            "calibration_dataset": str(CALIBRATION.relative_to(ROOT)),
            "calibration_dataset_sha256": sha256(CALIBRATION),
            "calibration_rows": int(labels.size),
            "calibration_legitimate": int((labels == 0).sum()),
            "calibration_phishing": int((labels == 1).sum()),
            "threshold_method": "maximize recall subject to each registered calibration FPR cap; tie-break by F1 then lower FPR",
            "score_artifact": str(args.scores.relative_to(ROOT)),
            "score_artifact_sha256": sha256(args.scores),
            "operating_points": points,
            "final_test_tuning": False,
        },
        "primary_binary_rule": {
            "threshold": balanced,
            "action": "scores at or above this value are caution-or-higher detections",
            "calibration_fpr_cap": 0.01,
        },
        "tiers": [
            {"name": "allow", "lower_inclusive": None, "upper_exclusive": high_recall, "action": "allow normally"},
            {"name": "secondary_review", "lower_inclusive": high_recall, "upper_exclusive": balanced, "action": "show non-blocking review notice"},
            {"name": "caution", "lower_inclusive": balanced, "upper_exclusive": conservative, "action": "show passive caution"},
            {"name": "warning", "lower_inclusive": conservative, "upper_exclusive": strict, "action": "pause navigation with user override"},
            {"name": "block", "lower_inclusive": strict, "upper_exclusive": None, "action": "pause navigation with explicit user override"},
        ],
        "guardrails": [
            "Thresholds were derived from the same frozen calibration cohort and registered FPR-cap method, not from final test labels.",
            "Scores are ranking outputs, not guaranteed probabilities or percentages of trust.",
            "The candidate greatly reduces equivalent root-path decision drift but trades approximately 1.48 percentage points of external recall.",
            "This separate extension must not overwrite or be reported as the Version 3 v0.6.0 deployment control.",
            "The URL model does not include DNS, reputation, TLS, redirect-chain, or page-content evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "thresholds": points}, indent=2))


if __name__ == "__main__":
    main()
