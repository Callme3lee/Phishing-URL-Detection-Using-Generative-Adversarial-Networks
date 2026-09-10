from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json


ANALYSIS = Path("artifacts/results/threshold_sets_v2/threshold_analysis.json")
CALIBRATION = Path("data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv")
CHECKPOINT_ROOT = Path("artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms")
OUTPUT = Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1")


def main() -> None:
    policy_path = OUTPUT / "policy.json"
    if policy_path.exists():
        raise FileExistsError(f"Policy is already frozen: {policy_path}")
    analysis = load_json(ANALYSIS)
    points = {row["name"]: row for row in analysis["operating_points"]}
    required = ("Ultra-strict", "Strict", "Conservative", "Balanced", "High-recall")
    if any(name not in points for name in required):
        raise RuntimeError("Threshold analysis does not contain all required operating points")
    conservative = points["Conservative"]
    thresholds = {
        "allow_upper_exclusive": float(points["High-recall"]["single_stage_threshold"]),
        "secondary_review_upper_exclusive": float(points["Balanced"]["single_stage_threshold"]),
        "caution_upper_exclusive": float(conservative["single_stage_threshold"]),
        "warning_upper_exclusive": float(points["Strict"]["single_stage_threshold"]),
    }
    ordered = list(thresholds.values())
    if ordered != sorted(ordered):
        raise RuntimeError(f"Tier thresholds are not strictly ordered: {thresholds}")
    checkpoints=[]
    for seed in (42,43,44):
        path=CHECKPOINT_ROOT/f"seed_{seed}/checkpoint.pt"
        checkpoints.append({"seed":seed,"path":str(path),"sha256":sha256_file(path)})
    payload={
        "policy_version":"conservative_hyb_gan10_v1",
        "project_version":3,
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"OFFLINE RESEARCH DEPLOYMENT CANDIDATE; NOT DEPLOYMENT-READY",
        "model":"Updated HYB-MS +10% MS-HardGAN-C, three-seed mean-score ensemble",
        "ensemble_rule":"arithmetic mean of FP32 sigmoid probabilities from seeds 42/43/44",
        "primary_binary_rule":{"threshold":float(conservative["single_stage_threshold"]),
                               "action":"scores at or above this value are warning-or-block"},
        "tiers":[
            {"name":"allow","lower_inclusive":None,"upper_exclusive":thresholds["allow_upper_exclusive"],"action":"allow normally"},
            {"name":"secondary_review","lower_inclusive":thresholds["allow_upper_exclusive"],"upper_exclusive":thresholds["secondary_review_upper_exclusive"],"action":"request independent reputation/DNS/content evidence; do not block from URL score alone"},
            {"name":"caution","lower_inclusive":thresholds["secondary_review_upper_exclusive"],"upper_exclusive":thresholds["caution_upper_exclusive"],"action":"show passive caution; independent evidence may escalate"},
            {"name":"warning","lower_inclusive":thresholds["caution_upper_exclusive"],"upper_exclusive":thresholds["warning_upper_exclusive"],"action":"show strong phishing warning"},
            {"name":"block","lower_inclusive":thresholds["warning_upper_exclusive"],"upper_exclusive":None,"action":"interstitial block with explicit user override"},
        ],
        "selection_evidence":{
            "calibration_fpr_cap":float(conservative["calibration_fpr_cap"]),
            "identity_disjoint_20k_diagnostic_metrics":conservative["single_stage"],
            "threshold_analysis":str(ANALYSIS),"threshold_analysis_sha256":sha256_file(ANALYSIS),
            "calibration_dataset":str(CALIBRATION),"calibration_dataset_sha256":sha256_file(CALIBRATION),
        },
        "checkpoints":checkpoints,
        "input_contract":{"url_field":"url","canonicalization":"existing Version 3 scheme-free canonical URL processing","maximum_characters":128,
                          "lexical_features":"35 PSL-aware lexical values using checkpoint-stored robust scaling"},
        "guardrails":[
            "Scores are ranking outputs, not guaranteed probabilities or percentages of trust.",
            "The 20K cohort informed the diagnostic policy comparison and is not a fresh confirmation set.",
            "The secondary_review tier does not yet contain DNS, reputation, TLS, redirect, or page-content implementation.",
            "Do not change thresholds after future confirmation labels are viewed.",
        ],
        "next_confirmation":"After independent secondary evidence is implemented and calibrated on new development data, evaluate this frozen policy once on a genuinely later low-prevalence cohort.",
    }
    OUTPUT.mkdir(parents=True)
    write_json(policy_path,payload)
    write_json(OUTPUT/"manifest.json",{"policy":"policy.json","policy_sha256":sha256_file(policy_path),
        "checkpoint_hashes":{str(row["seed"]):row["sha256"] for row in checkpoints}})
    print(json.dumps({"policy":str(policy_path),"primary_threshold":payload["primary_binary_rule"]["threshold"],"tiers":payload["tiers"]},indent=2))


if __name__=="__main__":main()
