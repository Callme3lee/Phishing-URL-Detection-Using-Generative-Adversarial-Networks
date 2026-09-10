from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics


ROOT=Path("artifacts/results/updated_models_calibration_v1")
OUT=ROOT/"comparison"
CONDITIONS=("real_only","gan_10pct")
LABELS={"real_only":"Updated real-only","gan_10pct":"Updated +10% GAN"}
COLORS={"real_only":"#4C78A8","gan_10pct":"#1C9A8B"}


def pct(v):return f"{100*v:.2f}%"


def curve(labels,scores):
    order=np.argsort(-scores,kind="stable");y=labels[order];s=scores[order]
    tp=np.cumsum(y==1);fp=np.cumsum(y==0);p=max(1,int((y==1).sum()));n=max(1,int((y==0).sum()))
    distinct=np.r_[s[1:]!=s[:-1],True]
    return fp[distinct]/n,tp[distinct]/p,s[distinct]


def at_cap(analysis,cap):
    return next(row["metrics"] for row in analysis["tradeoffs"] if abs(row["maximum_fpr"]-cap)<1e-12)


def projected_precision(recall,fpr,prevalence=.01):
    return recall*prevalence/(recall*prevalence+fpr*(1-prevalence))


def title(fig,h,s):
    fig.text(.055,.94,h,fontsize=23,weight="bold",color="#17324D",va="top");fig.text(.055,.885,s,fontsize=11.5,color="#526274",va="top")
    fig.add_artist(plt.Line2D([.055,.945],[.85,.85],color="#CAD5E1",lw=1))


def graph(path,payload,curves):
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.patch.set_facecolor("white")
    for c,(fpr,recall,_) in curves.items():
        mask=fpr<=.012;axes[0].plot(100*fpr[mask],100*recall[mask],label=LABELS[c],color=COLORS[c],lw=2.5)
    for cap,style in ((.0025,"--"),(.005,":"),(.01,"-.")):axes[0].axvline(100*cap,color="#8392A5",ls=style,lw=1)
    axes[0].axhline(80,color="#C94C4C",ls="--",lw=1);axes[0].set(xlabel="False-positive rate (%)",ylabel="Recall (%)",xlim=(0,1.2),ylim=(20,95))
    axes[0].grid(alpha=.2);axes[0].legend();axes[0].set_title("FP32 threshold trade-off",weight="bold")
    caps=[.0025,.005,.01];x=np.arange(3);w=.34
    for offset,c in ((-w/2,"real_only"),(w/2,"gan_10pct")):
        values=[100*payload["conditions"][c]["caps"][str(cap)]["recall"] for cap in caps]
        bars=axes[1].bar(x+offset,values,w,label=LABELS[c],color=COLORS[c])
        for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+.8,f"{value:.1f}",ha="center",fontsize=9)
    axes[1].axhline(80,color="#C94C4C",ls="--",lw=1,label="80% recall target")
    axes[1].set_xticks(x,["0.25% FPR cap","0.50% FPR cap","1.00% FPR cap"]);axes[1].set_ylabel("Recall (%)");axes[1].set_ylim(0,100);axes[1].grid(axis="y",alpha=.2);axes[1].legend()
    axes[1].set_title("Recall permitted by each FPR budget",weight="bold")
    fig.suptitle("Updated HYB-MS ensemble calibration",fontsize=20,weight="bold",color="#17324D")
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def pdf(path,payload,curves):
    with PdfPages(path) as out:
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");title(fig,"Updated-Ensemble FP32 Calibration","New identity-disjoint cohort • same thresholds rules • no protected confirmation reuse")
        fig.text(.06,.76,"STRICT 0.25% FPR GATE: BOTH FAIL",fontsize=20,weight="bold",color="#C94C4C")
        cols=["Condition","Threshold","Precision","Recall","F1","ROC-AUC","PR-AUC","FPR","FNR"]
        rows=[]
        for c in CONDITIONS:
            d=payload["conditions"][c];m=d["selected_metrics"]
            rows.append([LABELS[c],f"{d['selected_threshold']:.6f}",*[pct(m[k]) for k in ("precision","recall","f1","roc_auc","pr_auc","false_positive_rate","false_negative_rate")]])
        ax=fig.add_axes([.04,.45,.92,.18]);ax.axis("off");tab=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center")
        tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,2.25)
        for (r,_),cell in tab.get_celld().items():
            cell.set_edgecolor("#CAD5E1");cell.set_facecolor("#365F7F" if r==0 else ("#EEF3F7" if r==1 else "#E7F3F1"))
            if r==0:cell.set_text_props(color="white",weight="bold")
        fig.text(.06,.34,"FP32 fixed score quantization, but not distribution shift.",fontsize=15,weight="bold",color="#17324D")
        fig.text(.06,.27,textwrap.fill("Real-only reaches 52.08% recall and +10% GAN reaches 43.48% recall at the strict FPR budget. Their ROC-AUC values near 98% show useful ranking, but the low-FPR tail is not sufficiently separated to meet the registered browser-extension gate.",135),fontsize=11,color="#526274")
        fig.text(.06,.13,"This calibration result supersedes the uncalibrated internal thresholds for deployment discussion; it does not alter the internal experiment itself.",fontsize=10.5,color="#526274")
        out.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11,8.5));fig.patch.set_facecolor("white")
        for c,(fpr,recall,_) in curves.items():
            mask=fpr<=.012;axes[0].plot(100*fpr[mask],100*recall[mask],label=LABELS[c],color=COLORS[c],lw=2.4)
        axes[0].axhline(80,color="#C94C4C",ls="--");axes[0].axvline(.25,color="#8392A5",ls="--");axes[0].axvline(.5,color="#8392A5",ls=":");axes[0].axvline(1,color="#8392A5",ls="-.")
        axes[0].set(xlabel="FPR (%)",ylabel="Recall (%)",xlim=(0,1.2),ylim=(20,95));axes[0].grid(alpha=.2);axes[0].legend();axes[0].set_title("Threshold trade-off",weight="bold")
        caps=[.0025,.005,.01];x=np.arange(3);w=.34
        for off,c in ((-w/2,"real_only"),(w/2,"gan_10pct")):
            vals=[100*payload["conditions"][c]["caps"][str(v)]["recall"] for v in caps];axes[1].bar(x+off,vals,w,label=LABELS[c],color=COLORS[c])
        axes[1].axhline(80,color="#C94C4C",ls="--");axes[1].set_xticks(x,["0.25%","0.50%","1.00%"]);axes[1].set(xlabel="FPR cap",ylabel="Recall (%)",ylim=(0,100));axes[1].grid(axis="y",alpha=.2);axes[1].legend();axes[1].set_title("Recall by operational budget",weight="bold")
        fig.suptitle("FP32 Calibration Curves",fontsize=21,weight="bold",color="#17324D");fig.tight_layout(rect=[0,0,1,.94]);out.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");title(fig,"Decision and Next Action","What the corrected calibration supports")
        caps=(.0025,.005,.01);y=.77
        for cap in caps:
            fig.text(.07,y,f"FPR cap {100*cap:.2f}%",fontsize=13,weight="bold",color="#17324D")
            y-=.05
            for c in CONDITIONS:
                m=payload["conditions"][c]["caps"][str(cap)]
                fig.text(.10,y,f"{LABELS[c]}: recall {pct(m['recall'])}, F1 {pct(m['f1'])}, projected precision at 1% prevalence {pct(m['projected_precision_at_1pct'])}",fontsize=10.8,color="#526274");y-=.045
            y-=.035
        fig.text(.07,.28,"VERDICT",fontsize=11,weight="bold",color="#1C9A8B")
        fig.text(.07,.22,textwrap.fill(payload["verdict"],125),fontsize=12,weight="bold",color="#17324D")
        fig.text(.07,.10,textwrap.fill("Do not choose the 0.5% or 1% setting merely because it reaches higher recall. The operational FPR budget must be justified before viewing a future confirmation set. If the project retains 0.25%, the model representation/training data—not just calibration—must improve.",135),fontsize=10.5,color="#526274")
        out.savefig(fig);plt.close(fig)


def main():
    if OUT.exists():raise FileExistsError(f"Comparison already frozen at {OUT}")
    analyses={c:load_json(ROOT/c/"calibration_analysis.json") for c in CONDITIONS};pred={c:np.load(ROOT/c/"calibration_predictions.npz") for c in CONDITIONS}
    labels=np.asarray(pred["real_only"]["labels"],dtype=np.uint8)
    if not np.array_equal(labels,pred["gan_10pct"]["labels"]):raise RuntimeError("Conditions do not share calibration labels")
    curves={c:curve(labels,np.asarray(pred[c]["scores"],dtype=np.float64)) for c in CONDITIONS}
    payload={"project_version":3,"experiment":"updated_models_calibration_v1","probabilities":"FP32 sigmoid after FP16 forward pass",
             "gate":{"maximum_fpr":.0025,"minimum_recall":.80},"conditions":{}}
    for c in CONDITIONS:
        a=analyses[c];caps={}
        for cap in (.0025,.005,.01):
            m=dict(at_cap(a,cap));m["projected_precision_at_1pct"]=projected_precision(m["recall"],m["false_positive_rate"]);caps[str(cap)]=m
        payload["conditions"][c]={"selected_threshold":a["selected_threshold"],"selected_metrics":a["selected_metrics"],
                                  "gate_passed":a["deployment_gate_passed"],"caps":caps,
                                  "prediction_sha256":sha256_file(ROOT/c/"calibration_predictions.npz")}
    payload["verdict"]=("Neither ensemble meets 80% recall at FPR <= 0.25%. Real-only is better at the strict cap; +10% GAN becomes better when the FPR budget is relaxed to 0.5% or 1%. FP32 scoring corrected threshold resolution but did not solve low-FPR distribution shift.")
    OUT.mkdir(parents=True);graph(OUT/"fp32_threshold_tradeoff.png",payload,curves);pdf(OUT/"updated_models_fp32_calibration_report.pdf",payload,curves)
    payload["artifacts"]={name:sha256_file(OUT/name) for name in ("fp32_threshold_tradeoff.png","updated_models_fp32_calibration_report.pdf")}
    write_json(OUT/"comparison.json",payload);print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
