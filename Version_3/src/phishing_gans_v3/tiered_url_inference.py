from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


DEFAULT_POLICY=Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")


def assign_tier(score: float, tiers: list[dict]) -> str:
    for tier in tiers:
        lower=tier["lower_inclusive"];upper=tier["upper_exclusive"]
        if (lower is None or score>=float(lower)) and (upper is None or score<float(upper)):
            return str(tier["name"])
    raise RuntimeError(f"Score {score} is outside policy tiers")


def parse_args():
    parser=argparse.ArgumentParser(description="Score a URL CSV using the frozen conservative tier policy on CUDA.")
    parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--policy",type=Path,default=DEFAULT_POLICY);parser.add_argument("--batch-size",type=int,default=1024)
    parser.add_argument("--expected-scores",type=Path,help="Optional NPZ with stage1_scores for strict reproduction verification.")
    parser.add_argument("--score-tolerance",type=float,default=1e-4,
                        help="Maximum accepted FP16/CUDA score difference during reproduction verification.")
    return parser.parse_args()


def main():
    args=parse_args();policy=load_json(args.policy);rows=list(csv.DictReader(args.input.open("r",encoding="utf-8",newline="")))
    if not rows or "url" not in rows[0]:raise ValueError("Input CSV must contain a non-empty 'url' column")
    normalized=[];hosts=[]
    for row in rows:
        url,host,_=canonicalize_url(row["url"]);normalized.append(url);hosts.append(host)
    raw=np.stack([psl_aware_lexical_features(url) for url in normalized]).astype(np.float32);device=require_cuda();seed_scores=[];started=time.perf_counter()
    for item in policy["checkpoints"]:
        path=Path(item["path"])
        if sha256_file(path)!=item["sha256"]:raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        saved=torch.load(path,map_location=device,weights_only=False);tokens=np.stack([encode_url(url,saved["vocabulary"],128) for url in normalized])
        scale=np.asarray(saved["lexical_scaler_scale"],dtype=np.float32);scale=np.where(scale==0,1,scale)
        lexical=((raw-np.asarray(saved["lexical_scaler_center"],dtype=np.float32))/scale).astype(np.float32)
        model=create_model(saved).to(device);model.load_state_dict(saved["state_dict"]);seed_scores.append(neural_score(model,tokens,lexical,args.batch_size,device))
        del model,tokens,lexical;gc.collect();torch.cuda.empty_cache()
    scores=np.mean(np.stack(seed_scores),axis=0).astype(np.float32);elapsed=time.perf_counter()-started
    tiers=[assign_tier(float(score),policy["tiers"]) for score in scores];threshold=float(policy["primary_binary_rule"]["threshold"]);predictions=scores>=threshold
    maximum_difference=None
    if args.expected_scores:
        expected=np.load(args.expected_scores)["stage1_scores"].astype(np.float32)
        if len(expected)!=len(scores):raise RuntimeError("Expected-score row count mismatch")
        maximum_difference=float(np.max(np.abs(expected-scores)))
        if maximum_difference>args.score_tolerance:raise RuntimeError(f"Ensemble reproduction failed: maximum absolute difference {maximum_difference}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys())+["normalized_url","canonical_hostname","ensemble_score","risk_tier","primary_prediction"]
    with args.output.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row,url,host,score,tier,prediction in zip(rows,normalized,hosts,scores,tiers,predictions):
            writer.writerow({**row,"normalized_url":url,"canonical_hostname":host,"ensemble_score":f"{float(score):.9f}","risk_tier":tier,"primary_prediction":"phishing" if prediction else "legitimate"})
    payload={"policy":str(args.policy),"policy_sha256":sha256_file(args.policy),"input":str(args.input),"input_sha256":sha256_file(args.input),
        "output":str(args.output),"output_sha256":sha256_file(args.output),"rows":len(rows),"tier_counts":dict(Counter(tiers)),
        "primary_prediction_counts":{"phishing":int(predictions.sum()),"legitimate":int((~predictions).sum())},
        "inference_seconds":elapsed,"urls_per_second":len(rows)/elapsed,"device":torch.cuda.get_device_name(device),
        "maximum_expected_score_difference":maximum_difference,"expected_score_tolerance":args.score_tolerance}
    if "label" in rows[0]:
        labels=np.asarray([int(row["label"]) for row in rows],dtype=np.uint8);payload["metrics"]=classification_metrics(labels,scores,threshold)
        payload["tier_label_counts"]={tier:{"legitimate":int(((np.asarray(tiers)==tier)&(labels==0)).sum()),"phishing":int(((np.asarray(tiers)==tier)&(labels==1)).sum())} for tier in dict.fromkeys(tiers)}
    report=args.output.with_suffix(".report.json");write_json(report,payload);print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
