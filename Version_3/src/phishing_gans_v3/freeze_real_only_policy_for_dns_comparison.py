from __future__ import annotations

import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from .analyze_threshold_sets import recall_threshold
from .common import sha256_file,write_json


PREDICTIONS=Path("artifacts/results/updated_models_calibration_v1/real_only/calibration_predictions.npz")
CALIBRATION=Path("data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv")
CHECKPOINT_ROOT=Path("artifacts/experiments/recent_update_gan10_v1/real_only/hyb_ms")
OUTPUT=Path("artifacts/deployment_candidates/conservative_hyb_real_v1")


def main():
    if OUTPUT.exists():raise FileExistsError(f"Real-only policy already frozen: {OUTPUT}")
    values=np.load(PREDICTIONS);labels=values["labels"].astype(np.uint8);scores=values["scores"].astype(np.float32)
    selected={cap:recall_threshold(labels,scores,cap)["threshold"] for cap in (.001,.0025,.005,.01,.05)}
    tiers=[
        {"name":"allow","lower_inclusive":None,"upper_exclusive":selected[.05],"action":"allow normally"},
        {"name":"secondary_review","lower_inclusive":selected[.05],"upper_exclusive":selected[.01],"action":"request the same frozen DNS secondary evidence"},
        {"name":"caution","lower_inclusive":selected[.01],"upper_exclusive":selected[.005],"action":"show passive caution"},
        {"name":"warning","lower_inclusive":selected[.005],"upper_exclusive":selected[.0025],"action":"show strong phishing warning"},
        {"name":"block","lower_inclusive":selected[.0025],"upper_exclusive":None,"action":"interstitial block with override"},
    ]
    checkpoints=[]
    for seed in (42,43,44):
        path=CHECKPOINT_ROOT/f"seed_{seed}/checkpoint.pt";checkpoints.append({"seed":seed,"path":str(path),"sha256":sha256_file(path)})
    payload={"policy_version":"conservative_hyb_real_v1","project_version":3,"frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"CONTROL POLICY FOR REAL-ONLY VERSUS GAN+DNS EXPERIMENT","model":"Updated real-only HYB-MS, three-seed mean-score ensemble",
        "ensemble_rule":"arithmetic mean of FP32 sigmoid probabilities from seeds 42/43/44","primary_binary_rule":{"threshold":selected[.005],"action":"warning-or-block"},
        "tiers":tiers,"threshold_selection_rule":"for each tier maximize recall on the common calibration cohort subject to its FPR cap; F1 and lower FPR break ties",
        "calibration_fpr_caps":{"block":.0025,"primary_conservative":.005,"caution":.01,"secondary_review_lower":.05},
        "calibration_dataset":str(CALIBRATION),"calibration_dataset_sha256":sha256_file(CALIBRATION),"calibration_predictions":str(PREDICTIONS),"calibration_predictions_sha256":sha256_file(PREDICTIONS),
        "checkpoints":checkpoints,"comparison_contract":{"same_real_training_rows":367184,"gan_training_rows":0,"dns_model":"artifacts/experiments/dns_secondary_v1/frozen_dns_secondary.json",
            "dns_model_must_remain_unchanged":True,"evaluation_datasets_must_match_gan_condition":True},
        "guardrails":["No threshold uses development holdout or 20K labels.","Separate score thresholds are required because real-only and GAN ensembles have different score distributions.","The calibration cohort and FPR-cap rules are identical to the GAN primary policy."]}
    OUTPUT.mkdir(parents=True);write_json(OUTPUT/"policy.json",payload);write_json(OUTPUT/"manifest.json",{"policy_sha256":sha256_file(OUTPUT/"policy.json"),"checkpoint_hashes":{str(x["seed"]):x["sha256"] for x in checkpoints}})
    print(json.dumps({"primary_threshold":selected[.005],"thresholds":selected,"tiers":tiers},indent=2))


if __name__=="__main__":main()
