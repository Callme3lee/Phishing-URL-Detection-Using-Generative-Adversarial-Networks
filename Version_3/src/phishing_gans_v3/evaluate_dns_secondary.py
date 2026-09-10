from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/phishing_gans_v3_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score,roc_auc_score

from .common import load_json,sha256_file,write_json
from .metrics import classification_metrics
from .train_dns_secondary import FEATURES,vector


MODEL=Path("artifacts/experiments/dns_secondary_v1/frozen_dns_secondary.json")
FULL=Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/validation/20k_tiered_predictions.csv")
DNS=Path("artifacts/results/dns_secondary_v1/20k_dns_secondary_review.csv")
OUTPUT=Path("artifacts/results/dns_secondary_v1")


def projected_precision(recall,fpr,prevalence):
    numerator=recall*prevalence;return numerator/max(1e-15,numerator+fpr*(1-prevalence))


def pct(v):return f"{100*v:.2f}%"


def make_pdf(path,payload):
    p=payload["primary_only"];c=payload["combined"]
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        fig.text(.06,.92,"DNS Secondary Verifier",fontsize=25,weight="bold",color="#17324D")
        fig.text(.06,.865,"Frozen URL policy + DNS-only routed evidence • diagnostic 20K evaluation",fontsize=12,color="#526274")
        cols=["System","Accuracy","Precision","Recall","F1","ROC-AUC","FPR","FNR"]
        rows=[["Primary only",*[pct(p[k]) for k in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]],
              ["Primary + DNS",*[pct(c[k]) for k in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]]]
        ax=fig.add_axes([.05,.52,.90,.18]);ax.axis("off");t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center");t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,2.2)
        for (r,_),cell in t.get_celld().items():cell.set_edgecolor("#C8D3DE");cell.set_facecolor("#365F7F" if r==0 else ("#E7F3F1" if r==2 else "#F4F7FA"));cell.set_text_props(color="white" if r==0 else "#17324D",weight="bold" if r==0 else "normal")
        fig.text(.07,.39,f"Recovered phishing: +{payload['additional_true_positives']}    Additional false positives: +{payload['additional_false_positives']}",fontsize=15,weight="bold",color="#188977")
        fig.text(.07,.31,f"FNR change: {100*(c['false_negative_rate']-p['false_negative_rate']):+.2f} points    FPR change: {100*(c['false_positive_rate']-p['false_positive_rate']):+.3f} points",fontsize=13,weight="bold",color="#17324D")
        fig.text(.07,.21,payload["verdict"],fontsize=10.8,color="#526274",wrap=True)
        fig.text(.07,.09,"The DNS model and threshold were locked on separate development data before these DNS records were collected. The 20K cohort was already viewed, so this remains diagnostic—not temporal confirmation.",fontsize=9.7,color="#C26B32",wrap=True)
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11,8.5));fig.patch.set_facecolor("white")
        names=["Recall","F1","Precision","FNR","FPR"];keys=["recall","f1","precision","false_negative_rate","false_positive_rate"];x=np.arange(len(names));w=.34
        axes[0].bar(x-w/2,[100*p[k] for k in keys],w,label="Primary",color="#4C78A8");axes[0].bar(x+w/2,[100*c[k] for k in keys],w,label="Primary + DNS",color="#1C9A8B");axes[0].set_xticks(x,names,rotation=15);axes[0].set_ylabel("Percent");axes[0].grid(axis="y",alpha=.2);axes[0].legend();axes[0].set_title("All 20,000 URLs",weight="bold")
        top=payload["feature_importance"][:8];axes[1].barh([r["feature"].replace("_"," ") for r in top][::-1],[r["importance"] for r in top][::-1],color="#F2B134");axes[1].set_xlabel("Mean RF importance");axes[1].grid(axis="x",alpha=.2);axes[1].set_title("DNS evidence used",weight="bold")
        fig.suptitle("Locked combined-policy comparison",fontsize=19,weight="bold",color="#17324D");fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.06,.92,"Interpretation and Limits",fontsize=24,weight="bold",color="#17324D")
        items=[("Independent branch",f"The RF receives {len(FEATURES)} DNS values only; it never sees URL characters, lexical values, primary scores, response time, or 20K labels."),
               ("Routed scope",f"Only {payload['routed']['rows']} secondary-review URLs were queried. DNS ROC-AUC inside that difficult subset is {pct(payload['routed']['dns_roc_auc'])}."),
               ("Latency",f"The concurrent offline collection took {payload['dns_collection']['elapsed_seconds']:.1f}s for {payload['routed']['rows']} domains. This is not acceptable evidence of browser latency."),
               ("Main risk","DNS absence and TTL can reflect takedown timing, resolver policy, or temporary failure. Their predictive value may decay rapidly."),
               ("Safety","No phishing page, redirect, TLS endpoint, WHOIS/RDAP service, or page content was opened."),
               ("Next proof","Repeat on a later low-prevalence snapshot with a cache and timeout policy; do not change the frozen model or threshold after seeing labels.")]
        y=.82
        for title,text in items:fig.text(.07,y,title,fontsize=13,weight="bold",color="#17324D");fig.text(.25,y,text,fontsize=10.5,color="#526274",wrap=True);y-=.12
        pdf.savefig(fig);plt.close(fig)


def main():
    model_report=load_json(MODEL);full=list(csv.DictReader(FULL.open("r",encoding="utf-8",newline="")));dns=list(csv.DictReader(DNS.open("r",encoding="utf-8",newline="")))
    dns_by_url={row["url"]:row for row in dns};labels=np.asarray([int(row["label"]) for row in full],dtype=np.uint8);primary_scores=np.asarray([float(row["ensemble_score"]) for row in full],dtype=np.float32)
    routed=np.asarray([i for i,row in enumerate(full) if row["risk_tier"]=="secondary_review"],dtype=int);routed_rows=[dns_by_url[full[i]["url"]] for i in routed]
    models=[]
    for item in model_report["models"]:
        path=Path(item["path"])
        if sha256_file(path)!=item["sha256"]:raise RuntimeError(f"Model hash mismatch: {path}")
        models.append(joblib.load(path))
    x=np.asarray([vector(row) for row in routed_rows],dtype=np.float32);secondary=np.mean(np.stack([m.predict_proba(x)[:,1] for m in models]),axis=0);threshold=float(model_report["secondary_threshold"]);primary_threshold=float(model_report["primary_threshold"])
    primary_prediction=primary_scores>=primary_threshold;combined=primary_prediction.copy();combined[routed]|=secondary>=threshold
    ranking=primary_scores.copy();ranking[routed]=np.maximum(ranking[routed],primary_threshold*secondary)
    primary_metrics=classification_metrics(labels,primary_scores,primary_threshold);combined_metrics=classification_metrics(labels,combined.astype(np.float32),.5);combined_metrics["roc_auc"]=float(roc_auc_score(labels,ranking));combined_metrics["pr_auc"]=float(average_precision_score(labels,ranking))
    routed_labels=labels[routed];dns_report=load_json(DNS.with_suffix(".report.json"));feature_importance=model_report["feature_importance"]
    additional_tp=int((combined&(labels==1)).sum()-(primary_prediction&(labels==1)).sum());additional_fp=int((combined&(labels==0)).sum()-(primary_prediction&(labels==0)).sum())
    verdict=(f"The locked DNS branch recovered {additional_tp} additional phishing URLs and added {additional_fp} false positives. FNR changed from {pct(primary_metrics['false_negative_rate'])} to {pct(combined_metrics['false_negative_rate'])}; FPR changed from {pct(primary_metrics['false_positive_rate'])} to {pct(combined_metrics['false_positive_rate'])}. DNS helps modestly but does not solve the remaining URL-only misses.")
    payload={"project_version":3,"experiment":"dns_secondary_v1","status":"LOCKED DIAGNOSTIC EVALUATION COMPLETE","primary_only":primary_metrics,"combined":combined_metrics,
        "additional_true_positives":additional_tp,"additional_false_positives":additional_fp,"verdict":verdict,
        "routed":{"rows":len(routed),"legitimate_rows":int((routed_labels==0).sum()),"phishing_rows":int(routed_labels.sum()),"dns_roc_auc":float(roc_auc_score(routed_labels,secondary)),"dns_pr_auc":float(average_precision_score(routed_labels,secondary)),"secondary_threshold":threshold,
                  "escalated_rows":int((secondary>=threshold).sum()),"escalated_legitimate":int(((secondary>=threshold)&(routed_labels==0)).sum()),"escalated_phishing":int(((secondary>=threshold)&(routed_labels==1)).sum())},
        "projected_precision":{"primary":{str(p):projected_precision(primary_metrics["recall"],primary_metrics["false_positive_rate"],p) for p in (.001,.01,.05,.10)},
                               "combined":{str(p):projected_precision(combined_metrics["recall"],combined_metrics["false_positive_rate"],p) for p in (.001,.01,.05,.10)}},
        "feature_importance":feature_importance,"dns_collection":dns_report,"frozen_model":str(MODEL),"frozen_model_sha256":sha256_file(MODEL),
        "artifact_hashes":{"full_predictions":sha256_file(FULL),"dns_evidence":sha256_file(DNS)},"limitations":model_report["guardrails"]+dns_report["limitations"]}
    write_json(OUTPUT/"evaluation.json",payload);np.savez_compressed(OUTPUT/"evaluation_predictions.npz",labels=labels,primary_scores=primary_scores,routed_indices=routed,secondary_scores=secondary,primary_predictions=primary_prediction,combined_predictions=combined)
    make_pdf(OUTPUT/"dns_secondary_report.pdf",payload);print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
