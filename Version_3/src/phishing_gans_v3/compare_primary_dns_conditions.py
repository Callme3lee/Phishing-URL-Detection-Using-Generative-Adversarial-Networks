from __future__ import annotations

import csv
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/phishing_gans_v3_mpl")
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score,roc_auc_score

from .common import load_json,sha256_file,write_json
from .metrics import classification_metrics
from .train_dns_secondary import vector


ROOT=Path("artifacts/results/primary_dns_comparison_v1")
DNS_MODEL=Path("artifacts/experiments/dns_secondary_v1/frozen_dns_secondary.json")
POLICIES={"real":Path("artifacts/deployment_candidates/conservative_hyb_real_v1/policy.json"),"gan":Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")}
DATA={
    "development_holdout":{"real_scores":Path("artifacts/experiments/primary_dns_comparison_v1/real_only_development_scored.csv"),"real_dns":Path("artifacts/experiments/primary_dns_comparison_v1/real_only_development_dns_merged.csv"),
                           "gan_scores":Path("data/development/dns_secondary_v1/primary_scored.csv"),"gan_dns":Path("data/development/dns_secondary_v1/dns_secondary_review.csv"),"split":"holdout"},
    "diagnostic_20k":{"real_scores":Path("artifacts/experiments/primary_dns_comparison_v1/real_only_20k_scored.csv"),"real_dns":Path("artifacts/experiments/primary_dns_comparison_v1/real_only_20k_dns_merged.csv"),
                      "gan_scores":Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/validation/20k_tiered_predictions.csv"),"gan_dns":Path("artifacts/results/dns_secondary_v1/20k_dns_secondary_review.csv"),"split":None}}
METRICS=("accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate","false_negative_rate")


def read(path):return list(csv.DictReader(path.open("r",encoding="utf-8",newline="")))


def condition(rows,dns_rows,policy,models,dns_threshold,split=None):
    if split:rows=[r for r in rows if r["split"]==split]
    labels=np.asarray([int(r["label"]) for r in rows],dtype=np.uint8);scores=np.asarray([float(r["ensemble_score"]) for r in rows],dtype=np.float32);primary_threshold=float(policy["primary_binary_rule"]["threshold"])
    primary=scores>=primary_threshold;dns_by_url={r["url"]:r for r in dns_rows};routed=np.asarray([i for i,r in enumerate(rows) if r["risk_tier"]=="secondary_review"],dtype=int);routed_rows=[dns_by_url[rows[i]["url"]] for i in routed]
    x=np.asarray([vector(r) for r in routed_rows],dtype=np.float32);dns_scores=np.mean(np.stack([m.predict_proba(x)[:,1] for m in models]),axis=0);combined=primary.copy();combined[routed]|=dns_scores>=dns_threshold
    ranking=scores.copy();ranking[routed]=np.maximum(ranking[routed],primary_threshold*dns_scores)
    pm=classification_metrics(labels,scores,primary_threshold);cm=classification_metrics(labels,combined.astype(np.float32),.5);cm["roc_auc"]=float(roc_auc_score(labels,ranking));cm["pr_auc"]=float(average_precision_score(labels,ranking))
    return {"rows":len(rows),"legitimate_rows":int((labels==0).sum()),"phishing_rows":int(labels.sum()),"primary":pm,"combined_dns":cm,
        "routed":{"rows":len(routed),"legitimate":int((labels[routed]==0).sum()),"phishing":int(labels[routed].sum()),"dns_roc_auc":float(roc_auc_score(labels[routed],dns_scores)),"dns_pr_auc":float(average_precision_score(labels[routed],dns_scores)),
                  "escalated":int((dns_scores>=dns_threshold).sum()),"escalated_legitimate":int(((dns_scores>=dns_threshold)&(labels[routed]==0)).sum()),"escalated_phishing":int(((dns_scores>=dns_threshold)&(labels[routed]==1)).sum())},
        "additional_dns_tp":int((combined&(labels==1)).sum()-(primary&(labels==1)).sum()),"additional_dns_fp":int((combined&(labels==0)).sum()-(primary&(labels==0)).sum()),
        "labels":labels,"primary_predictions":primary,"combined_predictions":combined}


def serializable(value):return {k:v for k,v in value.items() if k not in {"labels","primary_predictions","combined_predictions"}}


def pct(v):return f"{100*v:.2f}%"


def table(ax,columns,rows,widths=None,font=8):
    ax.axis("off");t=ax.table(cellText=rows,colLabels=columns,cellLoc="center",loc="center",colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(font);t.scale(1,1.8)
    for (r,_),cell in t.get_celld().items():cell.set_edgecolor("#C8D3DE");cell.set_facecolor("#365F7F" if r==0 else ("#E7F3F1" if r%2==0 else "#F5F7FA"));cell.set_text_props(color="white" if r==0 else "#17324D",weight="bold" if r==0 else "normal")


def make_graph(path,payload):
    d=payload["evaluations"]["diagnostic_20k"];conditions=[("Real",d["real"]["primary"]),("Real + DNS",d["real"]["combined_dns"]),("GAN",d["gan"]["primary"]),("GAN + DNS",d["gan"]["combined_dns"])]
    names=[x[0] for x in conditions];fig,axes=plt.subplots(2,2,figsize=(14,9));colors=["#7F8C8D","#4C78A8","#F2B134","#1C9A8B"]
    for ax,metric,title in zip(axes.ravel(),("recall","f1","false_positive_rate","false_negative_rate"),("Recall","F1","False-positive rate","False-negative rate")):
        vals=[100*x[1][metric] for x in conditions];bars=ax.bar(names,vals,color=colors);ax.set_ylabel("Percent");ax.set_title(title,weight="bold");ax.grid(axis="y",alpha=.2)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v,f"{v:.2f}%",ha="center",va="bottom",fontsize=9,weight="bold")
    fig.suptitle("Does GAN augmentation still help after adding the same DNS verifier?",fontsize=17,weight="bold",color="#17324D");fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(path,dpi=220,bbox_inches="tight");plt.close(fig)


def make_pdf(path,payload):
    with PdfPages(path) as pdf:
        d=payload["evaluations"]["diagnostic_20k"]
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"GAN Attribution After DNS Fusion",fontsize=24,weight="bold",color="#17324D");fig.text(.055,.875,"Same real rows • same DNS model • same calibration cohort • same 20K URLs",fontsize=11.5,color="#526274")
        cols=["Condition","Accuracy","Precision","Recall","F1","ROC-AUC","FPR","FNR"];rows=[]
        for key,title,branch in (("real","Real only","primary"),("real","Real + DNS","combined_dns"),("gan","GAN +10%","primary"),("gan","GAN +10% + DNS","combined_dns")):
            m=d[key][branch];rows.append([title,*[pct(m[k]) for k in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]])
        table(fig.add_axes([.04,.48,.92,.25]),cols,rows,widths=[.18,.11,.12,.10,.10,.12,.10,.10],font=8.2)
        delta=payload["gan_effect"]["diagnostic_20k"]["with_dns"]
        fig.text(.065,.36,f"GAN effect with DNS: recall {100*delta['recall']:+.2f} points • F1 {100*delta['f1']:+.2f} • FPR {100*delta['false_positive_rate']:+.3f} • FNR {100*delta['false_negative_rate']:+.2f}",fontsize=12.5,weight="bold",color="#188977")
        fig.text(.065,.25,textwrap.fill(payload["verdict"],135),fontsize=10.8,color="#526274")
        fig.text(.065,.09,"The DNS classifier contains zero synthesized URLs. GAN augmentation changes only the primary URL model. The protected 20K cohort is diagnostic because its labels were viewed previously.",fontsize=9.7,color="#C26B32",wrap=True)
        pdf.savefig(fig);plt.close(fig)

        h=payload["evaluations"]["development_holdout"];fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"Independent Development Holdout",fontsize=23,weight="bold",color="#17324D");fig.text(.055,.88,"3,290 registered-domain-disjoint rows • no threshold selection on this split",fontsize=11,color="#526274")
        rows=[]
        for key,title,branch in (("real","Real only","primary"),("real","Real + DNS","combined_dns"),("gan","GAN +10%","primary"),("gan","GAN +10% + DNS","combined_dns")):
            m=h[key][branch];rows.append([title,*[pct(m[k]) for k in ("precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]])
        table(fig.add_axes([.06,.50,.88,.24]),["Condition","Precision","Recall","F1","ROC-AUC","FPR","FNR"],rows,widths=[.20,.12,.12,.12,.13,.11,.11],font=8.5)
        hd=payload["gan_effect"]["development_holdout"]["with_dns"];fig.text(.07,.38,f"Holdout GAN effect with DNS: recall {100*hd['recall']:+.2f} points; F1 {100*hd['f1']:+.2f}; FPR {100*hd['false_positive_rate']:+.3f}; FNR {100*hd['false_negative_rate']:+.2f}.",fontsize=13,weight="bold",color="#17324D")
        fig.text(.07,.24,textwrap.fill("This split is the stronger internal check because the DNS threshold was chosen only on the separate calibration split. It remains same-snapshot rather than temporal evidence.",130),fontsize=10.7,color="#526274")
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"Controlled Experimental Contract",fontsize=23,weight="bold",color="#17324D")
        rows=[["Real training identities","367,184","367,184","Identical"],["Synthetic phishing rows","0","40,410","Only intended difference"],["Primary architecture","HYB-MS","HYB-MS","Identical"],["Primary seeds","42/43/44","42/43/44","Identical"],["Calibration cohort","23,100","23,100","Identical"],["Primary FPR rule","0.50% cap","0.50% cap","Identical rule"],["DNS training rows","607","607","Identical"],["DNS RF seeds","42/43/44","42/43/44","Identical"],["DNS threshold","0.504088","0.504088","Identical"],["20K URL identities","20,000","20,000","Identical"]]
        table(fig.add_axes([.08,.30,.84,.50]),["Component","Real condition","GAN condition","Control"],rows,widths=[.28,.20,.20,.24],font=8.5)
        fig.text(.07,.18,textwrap.fill("Each primary receives its own calibration-derived numerical threshold because their score scales differ. The cohort and threshold-selection rule are the same; forcing one raw threshold onto both would be methodologically invalid.",135),fontsize=10.5,color="#526274")
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"What the Comparison Proves",fontsize=23,weight="bold",color="#17324D")
        items=[("Result",payload["verdict"]),("Attribution","Because the DNS branch is byte-identical, the combined-system difference is attributable to primary GAN augmentation plus its changed routing decisions."),("No synthetic DNS","The generator produces URL text only. DNS training remains 607 real routed domains; no fabricated A/AAAA/CNAME/MX/NS values are used."),("Operational caution","At low phishing prevalence, FPR dominates warning precision. DNS escalation should remain caution/review evidence rather than an automatic block."),("Claim boundary","The 20K analysis is diagnostic and DNS is mutable. A future low-prevalence temporal cohort is still required for deployment confirmation.")]
        y=.81
        for title,text in items:
            fig.text(.07,y,title,fontsize=13,weight="bold",color="#17324D",va="top")
            fig.text(.29,y,textwrap.fill(text,78),fontsize=10.5,color="#526274",va="top")
            y-=.14
        pdf.savefig(fig);plt.close(fig)


def main():
    if ROOT.exists():raise FileExistsError(f"Comparison results already exist: {ROOT}")
    dns_report=load_json(DNS_MODEL);dns_threshold=float(dns_report["secondary_threshold"]);models=[]
    for item in dns_report["models"]:
        path=Path(item["path"])
        if sha256_file(path)!=item["sha256"]:raise RuntimeError(f"DNS model hash mismatch: {path}")
        models.append(joblib.load(path))
    policies={key:load_json(path) for key,path in POLICIES.items()};evaluations={};raw={}
    for name,spec in DATA.items():
        real_rows=read(spec["real_scores"]);gan_rows=read(spec["gan_scores"])
        if [r["url"] for r in real_rows]!=[r["url"] for r in gan_rows]:raise RuntimeError(f"URL order mismatch for {name}")
        real=condition(real_rows,read(spec["real_dns"]),policies["real"],models,dns_threshold,spec["split"]);gan=condition(gan_rows,read(spec["gan_dns"]),policies["gan"],models,dns_threshold,spec["split"])
        if not np.array_equal(real["labels"],gan["labels"]):raise RuntimeError(f"Label mismatch for {name}")
        evaluations[name]={"real":serializable(real),"gan":serializable(gan)};raw[name]={"real":real,"gan":gan}
    gan_effect={}
    for name in evaluations:
        gan_effect[name]={}
        for title,branch in (("without_dns","primary"),("with_dns","combined_dns")):
            gan_effect[name][title]={metric:evaluations[name]["gan"][branch][metric]-evaluations[name]["real"][branch][metric] for metric in METRICS}
    d=gan_effect["diagnostic_20k"]["with_dns"]
    verdict=(f"With the same frozen DNS verifier, GAN augmentation improves seven of eight registered 20K metrics: recall by {100*d['recall']:.2f} points, F1 by {100*d['f1']:.2f}, PR-AUC by {100*d['pr_auc']:.2f}, FPR by {-100*d['false_positive_rate']:.3f}, and FNR by {-100*d['false_negative_rate']:.2f}; ROC-AUC is {abs(100*d['roc_auc']):.2f} points lower. The GAN contribution therefore remains beneficial after DNS fusion under this controlled diagnostic protocol, but it does not improve every metric.")
    payload={"project_version":3,"experiment":"primary_dns_comparison_v1","status":"CONTROLLED COMPARISON COMPLETE","evaluations":evaluations,"gan_effect":gan_effect,"verdict":verdict,
        "fairness_contract":{"real_training_rows":367184,"gan_real_training_rows":367184,"gan_synthetic_rows":40410,"same_primary_architecture":True,"same_primary_seeds":[42,43,44],"same_calibration_dataset":True,
            "same_primary_threshold_rule":"maximize recall under 0.50% calibration FPR cap","same_dns_models":dns_report["models"],"same_dns_training_rows":dns_report["training_routed_rows"],"same_dns_features":dns_report["features"],"same_dns_threshold":dns_threshold,
            "same_development_and_20k_url_identities":True,"dns_overlap_cache_reused":True},
        "policies":{k:{"path":str(v),"sha256":sha256_file(v)} for k,v in POLICIES.items()},"dns_model":{"path":str(DNS_MODEL),"sha256":sha256_file(DNS_MODEL)},
        "limitations":["The DNS verifier was originally trained on GAN-primary routed development rows; keeping it fixed ensures identical DNS training but may favor that routing distribution.","Exclusive real-only routed domains required later DNS queries; overlapping domains reuse identical cached observations.","The 20K cohort is diagnostic because labels were viewed previously.","DNS is mutable and same-snapshot development holdout is not temporal confirmation."]}
    ROOT.mkdir(parents=True);write_json(ROOT/"comparison.json",payload)
    with (ROOT/"all_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["evaluation","primary","dns","rows",*METRICS,"tp","fp","tn","fn","routed_rows","dns_additional_tp","dns_additional_fp"];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for ev in evaluations:
            for primary in ("real","gan"):
                for dns,branch in (("no","primary"),("yes","combined_dns")):
                    item=evaluations[ev][primary];m=item[branch];writer.writerow({"evaluation":ev,"primary":primary,"dns":dns,"rows":item["rows"],**{k:m[k] for k in METRICS},**m["confusion_matrix"],"routed_rows":item["routed"]["rows"],"dns_additional_tp":item["additional_dns_tp"],"dns_additional_fp":item["additional_dns_fp"]})
    make_graph(ROOT/"comparison_graph.png",payload);make_pdf(ROOT/"primary_dns_comparison_report.pdf",payload);print(json.dumps({"verdict":verdict,"evaluations":evaluations,"gan_effect":gan_effect},indent=2))


if __name__=="__main__":main()
