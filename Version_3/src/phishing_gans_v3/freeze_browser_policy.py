from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analyze_threshold_sets import recall_threshold
from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics


SOURCE_POLICY=Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
PREDICTIONS=Path("browser_extension/tests/browser_onnx_calibration_predictions.npz")
OUTPUT=Path("browser_extension/models/policy.json")
MANIFEST=Path("browser_extension/models/bundle_manifest.json")


def main()->None:
    source=load_json(SOURCE_POLICY);values=np.load(PREDICTIONS);labels=values["labels"].astype(np.uint8);scores=values["scores"].astype(np.float32)
    selected={cap:recall_threshold(labels,scores,cap) for cap in (.001,.0025,.005,.01,.05)}
    thresholds={"allow":selected[.05]["threshold"],"secondary_review":selected[.01]["threshold"],
                "caution":selected[.005]["threshold"],"warning":selected[.0025]["threshold"]}
    ordered=list(thresholds.values())
    if ordered!=sorted(ordered):raise RuntimeError(f"Browser thresholds are not ordered: {thresholds}")
    tiers=[
        {"name":"allow","lower_inclusive":None,"upper_exclusive":thresholds["allow"],"action":"allow normally"},
        {"name":"secondary_review","lower_inclusive":thresholds["allow"],"upper_exclusive":thresholds["secondary_review"],"action":"request independent reputation/DNS/content evidence; do not block from URL score alone"},
        {"name":"caution","lower_inclusive":thresholds["secondary_review"],"upper_exclusive":thresholds["caution"],"action":"show passive caution; independent evidence may escalate"},
        {"name":"warning","lower_inclusive":thresholds["caution"],"upper_exclusive":thresholds["warning"],"action":"show strong phishing warning"},
        {"name":"block","lower_inclusive":thresholds["warning"],"upper_exclusive":None,"action":"interstitial block with explicit user override"},
    ]
    payload={
        "policy_version":"conservative_hyb_gan10_browser_v1","source_policy_version":source["policy_version"],
        "model":source["model"],"status":source["status"],"maximum_characters":128,
        "ensemble_rule":"arithmetic mean of FP32 ONNX Runtime Web sigmoid outputs from seeds 42/43/44",
        "primary_binary_rule":{"threshold":thresholds["caution"],"action":source["primary_binary_rule"]["action"]},
        "tiers":tiers,"guardrails":source["guardrails"],"source_policy_sha256":sha256_file(SOURCE_POLICY),
        "portability_calibration":{
            "reason":"WebAssembly FP32 scores differ slightly from CUDA mixed-precision scores; numerical thresholds were re-derived without changing the cohort or FPR-cap rules.",
            "dataset":source["selection_evidence"]["calibration_dataset"],
            "dataset_sha256":source["selection_evidence"]["calibration_dataset_sha256"],
            "browser_predictions_sha256":sha256_file(PREDICTIONS),
            "threshold_rule":"maximize recall subject to the same calibration FPR cap; break ties by F1 then lower FPR",
            "operating_points":{str(cap):result for cap,result in selected.items()},
            "primary_metrics":classification_metrics(labels,scores,thresholds["caution"]),
            "no_final_test_tuning":True,
        },
    }
    write_json(OUTPUT,payload)
    manifest=load_json(MANIFEST);manifest["preprocessing"]["policy_sha256"]=sha256_file(OUTPUT)
    manifest["portability_policy"]={"source_policy_sha256":sha256_file(SOURCE_POLICY),"browser_policy_sha256":sha256_file(OUTPUT),
        "source_primary_threshold":source["primary_binary_rule"]["threshold"],"browser_primary_threshold":thresholds["caution"],
        "same_calibration_cohort":True,"same_fpr_cap_rules":True,"final_test_tuning":False}
    write_json(MANIFEST,manifest)
    print(json.dumps({"browser_policy":str(OUTPUT),"primary_threshold":thresholds["caution"],"thresholds":thresholds,
                      "primary_metrics":payload["portability_calibration"]["primary_metrics"]},indent=2))


if __name__=="__main__":main()
