from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics


ROOT=Path("data/development/dns_secondary_v1")
POLICY=Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
OUTPUT=Path("artifacts/experiments/dns_secondary_v1")
PREFIXES=("host_a","host_aaaa","host_cname","domain_mx","domain_ns")
FEATURES=tuple([f"{p}_{suffix}" for p in PREFIXES for suffix in ("count","minimum_ttl","nxdomain","timeout")]+
               ["dns_address_exists","dns_mail_exists","dns_nameserver_exists","dns_total_timeouts","dns_total_nxdomain"])


def vector(row):
    values=[]
    for name in FEATURES:
        value=float(row[name]);values.append(np.log1p(value) if name.endswith(("count","minimum_ttl")) or name.startswith("dns_total_") else value)
    return values


def select_combined_threshold(labels,primary_scores,routed_indices,dns_scores,primary_threshold,maximum_fpr):
    primary=primary_scores>=primary_threshold;candidates=np.r_[np.inf,np.unique(dns_scores)[::-1]];best=None
    for threshold in candidates:
        prediction=primary.copy();prediction[routed_indices]|=dns_scores>=threshold
        fp=int(((prediction)&(labels==0)).sum());tn=int(((~prediction)&(labels==0)).sum());fpr=fp/max(1,fp+tn)
        if fpr>maximum_fpr:continue
        tp=int((prediction&(labels==1)).sum());fn=int(((~prediction)&(labels==1)).sum());recall=tp/max(1,tp+fn);precision=tp/max(1,tp+fp);f1=2*precision*recall/max(1e-15,precision+recall)
        key=(recall,f1,-fpr)
        if best is None or key>best[0]:best=(key,float(threshold),prediction)
    if best is None:raise RuntimeError("No combined threshold satisfies the FPR budget")
    return best[1],best[2]


def metrics_from_prediction(labels,prediction,ranking):
    result=classification_metrics(labels,prediction.astype(np.float32),.5)
    result["roc_auc"]=float(roc_auc_score(labels,ranking));result["pr_auc"]=float(average_precision_score(labels,ranking));return result


def main():
    if OUTPUT.exists():raise FileExistsError(f"DNS secondary experiment exists: {OUTPUT}")
    policy=load_json(POLICY);primary_threshold=float(policy["primary_binary_rule"]["threshold"])
    all_rows=list(csv.DictReader((ROOT/"primary_scored.csv").open("r",encoding="utf-8",newline="")))
    dns_rows=list(csv.DictReader((ROOT/"dns_secondary_review.csv").open("r",encoding="utf-8",newline="")))
    dns_by_url={row["url"]:row for row in dns_rows};train=[row for row in dns_rows if row["split"]=="train"]
    x=np.asarray([vector(row) for row in train],dtype=np.float32);y=np.asarray([int(row["label"]) for row in train],dtype=np.uint8)
    models=[];model_records=[];OUTPUT.mkdir(parents=True)
    for seed in (42,43,44):
        model=RandomForestClassifier(n_estimators=500,max_depth=7,min_samples_leaf=6,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed)
        model.fit(x,y);path=OUTPUT/f"seed_{seed}.joblib";joblib.dump(model,path);models.append(model);model_records.append({"seed":seed,"path":str(path),"sha256":sha256_file(path)})
    def dns_score(rows):
        values=np.asarray([vector(row) for row in rows],dtype=np.float32);return np.mean(np.stack([m.predict_proba(values)[:,1] for m in models]),axis=0)
    for split in ("calibration","holdout"):
        split_rows=[row for row in all_rows if row["split"]==split];labels=np.asarray([int(row["label"]) for row in split_rows],dtype=np.uint8);primary_scores=np.asarray([float(row["ensemble_score"]) for row in split_rows],dtype=np.float32)
        routed_indices=np.asarray([i for i,row in enumerate(split_rows) if row["risk_tier"]=="secondary_review"],dtype=int);routed_rows=[dns_by_url[split_rows[i]["url"]] for i in routed_indices];secondary_scores=dns_score(routed_rows)
        if split=="calibration":
            threshold,combined_prediction=select_combined_threshold(labels,primary_scores,routed_indices,secondary_scores,primary_threshold,.0075)
        else:
            combined_prediction=primary_scores>=primary_threshold;combined_prediction[routed_indices]|=secondary_scores>=threshold
        primary_prediction=primary_scores>=primary_threshold;ranking=primary_scores.copy();ranking[routed_indices]=np.maximum(ranking[routed_indices],primary_threshold*secondary_scores)
        primary_metrics=classification_metrics(labels,primary_scores,primary_threshold);combined_metrics=metrics_from_prediction(labels,combined_prediction,ranking)
        routed_labels=labels[routed_indices];routed_metrics={"rows":len(routed_indices),"phishing_rows":int(routed_labels.sum()),"legitimate_rows":int((routed_labels==0).sum()),
            "dns_roc_auc":float(roc_auc_score(routed_labels,secondary_scores)),"dns_pr_auc":float(average_precision_score(routed_labels,secondary_scores))}
        payload={"split":split,"rows":len(split_rows),"primary_only":primary_metrics,"combined":combined_metrics,"routed":routed_metrics,
                 "additional_true_positives":int(((combined_prediction)&(labels==1)).sum()-((primary_prediction)&(labels==1)).sum()),
                 "additional_false_positives":int(((combined_prediction)&(labels==0)).sum()-((primary_prediction)&(labels==0)).sum())}
        np.savez_compressed(OUTPUT/f"{split}_predictions.npz",labels=labels,primary_scores=primary_scores,routed_indices=routed_indices,secondary_scores=secondary_scores,
                            primary_predictions=primary_prediction,combined_predictions=combined_prediction)
        write_json(OUTPUT/f"{split}_metrics.json",payload)
    importances=np.mean(np.stack([m.feature_importances_ for m in models]),axis=0);importance_rows=sorted(({"feature":name,"importance":float(value)} for name,value in zip(FEATURES,importances)),key=lambda r:r["importance"],reverse=True)
    report={"project_version":3,"experiment":"dns_secondary_v1","status":"LOCKED BEFORE 20K DNS COLLECTION","model":"three-seed random-forest DNS-only ensemble",
        "features":list(FEATURES),"excluded_features":["DNS query response time","primary model score","URL lexical or character features","HTTP/page/TLS/WHOIS evidence"],
        "training_routed_rows":len(train),"training_phishing_rows":int(y.sum()),"training_legitimate_rows":int((y==0).sum()),"seeds":[42,43,44],"models":model_records,
        "primary_policy":str(POLICY),"primary_policy_sha256":sha256_file(POLICY),"primary_threshold":primary_threshold,
        "secondary_threshold":threshold,"combined_calibration_maximum_fpr":.0075,"threshold_rule":"maximize combined calibration recall subject to total FPR <= 0.75%; break ties by F1 then lower FPR",
        "calibration":load_json(OUTPUT/"calibration_metrics.json"),"holdout":load_json(OUTPUT/"holdout_metrics.json"),"feature_importance":importance_rows,
        "data_hashes":{"development":sha256_file(ROOT/"development_urls.csv"),"primary_scores":sha256_file(ROOT/"primary_scored.csv"),"dns_evidence":sha256_file(ROOT/"dns_secondary_review.csv")},
        "guardrails":["The 20K evaluation labels were not used for training, feature choice or threshold selection.","DNS availability may encode takedown status and source timing rather than stable phishing behavior.","The holdout is identity-disjoint but drawn from the same source snapshot; temporal confirmation remains required."]}
    write_json(OUTPUT/"frozen_dns_secondary.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
