from __future__ import annotations

import csv
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json


EXPERIMENT = Path("artifacts/experiments/recent_update_gan10_v1")
DATA = Path("data/experiments/recent_update_gan10_v1")
OUTPUT = Path("artifacts/results/recent_update_gan10_v1")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")
DISPLAY = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
           "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR", "false_negative_rate": "FNR"}


def pct(value: float) -> str:
    return f"{100*value:.2f}%"


def title(fig, heading: str, subtitle: str) -> None:
    fig.text(.055, .94, heading, fontsize=23, weight="bold", color="#17324D", va="top")
    fig.text(.055, .885, subtitle, fontsize=11.5, color="#526274", va="top")
    fig.add_artist(plt.Line2D([.055,.945],[.85,.85],color="#CAD5E1",lw=1))


def make_figure(path: Path, summary: dict) -> None:
    real = summary["conditions"]["real_only"]; gan = summary["conditions"]["gan_10pct"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9)); fig.patch.set_facecolor("white")
    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["accuracy", "precision", "recall", "f1"]; x=np.arange(len(keys)); w=.35
    axes[0,0].bar(x-w/2,[100*real[k]["mean"] for k in keys],w,label="Updated real-only",color="#4C78A8")
    axes[0,0].bar(x+w/2,[100*gan[k]["mean"] for k in keys],w,label="Updated +10% GAN",color="#1C9A8B")
    axes[0,0].set_xticks(x,names); axes[0,0].set_ylim(60,102); axes[0,0].set_ylabel("Percent"); axes[0,0].legend(); axes[0,0].grid(axis="y",alpha=.2)
    axes[0,0].set_title("Threshold-dependent internal metrics",weight="bold")
    keys=["roc_auc","pr_auc"]; x=np.arange(2)
    axes[0,1].bar(x-w/2,[100*real[k]["mean"] for k in keys],w,color="#4C78A8")
    axes[0,1].bar(x+w/2,[100*gan[k]["mean"] for k in keys],w,color="#1C9A8B")
    axes[0,1].set_xticks(x,["ROC-AUC","PR-AUC"]); axes[0,1].set_ylim(97.5,100); axes[0,1].set_ylabel("Percent"); axes[0,1].grid(axis="y",alpha=.2)
    axes[0,1].set_title("Ranking metrics: essentially tied",weight="bold")
    for i,seed in enumerate(SEEDS):
        axes[1,0].plot([0,1],[100*real["f1"]["values"][i],100*gan["f1"]["values"][i]],marker="o",label=f"Seed {seed}")
    axes[1,0].set_xticks([0,1],["Real-only","+10% GAN"]); axes[1,0].set_ylabel("F1 (%)"); axes[1,0].grid(alpha=.2); axes[1,0].legend()
    axes[1,0].set_title("Paired F1: GAN wins 2 of 3 seeds",weight="bold")
    keys=["false_positive_rate","false_negative_rate"]; x=np.arange(2)
    axes[1,1].bar(x-w/2,[100*real[k]["mean"] for k in keys],w,color="#4C78A8")
    axes[1,1].bar(x+w/2,[100*gan[k]["mean"] for k in keys],w,color="#1C9A8B")
    axes[1,1].set_xticks(x,["FPR","FNR"]); axes[1,1].set_ylabel("Percent"); axes[1,1].grid(axis="y",alpha=.2)
    axes[1,1].set_title("Mean error rates",weight="bold")
    fig.suptitle("Recent-update HYB-MS: real-only vs 10% GAN augmentation",fontsize=20,weight="bold",color="#17324D")
    fig.text(.5,.015,"Same 367,184 real rows • same filtered internal evaluation • seeds 42/43/44 • mean values",ha="center",color="#526274")
    fig.tight_layout(rect=[0,.04,1,.94]); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def make_pdf(path: Path, summary: dict) -> None:
    real=summary["conditions"]["real_only"]; gan=summary["conditions"]["gan_10pct"]
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        title(fig,"Recent-Update GAN Experiment","Controlled HYB-MS comparison • identical real data • three paired seeds")
        fig.text(.06,.77,"MAIN RESULT",fontsize=11,weight="bold",color="#1C9A8B")
        fig.text(.06,.70,"+10% GAN improves mean F1, but the advantage is not seed-consistent.",fontsize=18,weight="bold",color="#17324D")
        cols=["Condition","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","FPR","FNR"]
        table_rows=[]
        for label,values in (("Updated real-only",real),("Updated +10% GAN",gan)):
            table_rows.append([label]+[f"{pct(values[k]['mean'])}\n±{pct(values[k]['standard_deviation'])}" for k in METRICS])
        ax=fig.add_axes([.04,.39,.92,.20]);ax.axis("off")
        tab=ax.table(cellText=table_rows,colLabels=cols,cellLoc="center",loc="center")
        tab.auto_set_font_size(False);tab.set_fontsize(7.8);tab.scale(1,2.35)
        for (row,_),cell in tab.get_celld().items():
            cell.set_edgecolor("#CAD5E1");cell.set_facecolor("#365F7F" if row==0 else ("#E7F3F1" if row==2 else "#EEF3F7"))
            if row==0:cell.set_text_props(color="white",weight="bold")
        fig.text(.06,.29,f"Mean GAN delta: F1 {100*summary['paired_deltas']['f1']['mean']:+.2f} points • recall {100*summary['paired_deltas']['recall']['mean']:+.2f} • FPR {100*summary['paired_deltas']['false_positive_rate']['mean']:+.4f} • ROC-AUC {100*summary['paired_deltas']['roc_auc']['mean']:+.3f}.",fontsize=11.5,weight="bold")
        fig.text(.06,.18,textwrap.fill("Interpretation: GAN augmentation helps the selected operating point on average, especially recall/F1, but seed 44 reverses that benefit and mean ranking metrics do not improve. This supports a promising internal effect, not a robust deployment improvement.",135),fontsize=11,color="#526274")
        fig.text(.06,.07,"No protected confirmation data was used. Earlier external/calibration cohorts containing update identities are retired.",fontsize=10,color="#526274")
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        title(fig,"Paired Seed Evidence","Each line compares the same seed under real-only and +10% GAN")
        keys=["accuracy","recall","f1","roc_auc"]
        for panel,(key,loc) in enumerate(zip(keys,[(.07,.52,.39,.25),(.55,.52,.39,.25),(.07,.16,.39,.25),(.55,.16,.39,.25)])):
            ax=fig.add_axes(loc)
            for i,seed in enumerate(SEEDS):
                ax.plot([0,1],[100*real[key]["values"][i],100*gan[key]["values"][i]],marker="o",label=f"Seed {seed}")
            ax.set_xticks([0,1],["Real-only","+10% GAN"]);ax.set_ylabel(DISPLAY[key]+" (%)");ax.grid(alpha=.2)
            ax.set_title(f"{DISPLAY[key]}: GAN wins {summary['paired_deltas'][key]['gan_wins']} / 3",weight="bold",fontsize=11)
            if panel==0:ax.legend(fontsize=8)
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        title(fig,"Method, Fairness, and Claim Boundary","What changed, what stayed fixed, and what this experiment can establish")
        sections=[
            ("Fixed across both conditions",["363,692 original multi-source real rows plus the same 3,492 recent-update rows.","HYB-MS architecture, PSL-aware features, CORAL weight, weighted source sampling family, seeds 42/43/44, preprocessing, early stopping, metrics, and evaluation identities.","994 old validation rows and 602 old test rows were removed from both conditions because 42 and 35 registered domains, respectively, entered the update."]),
            ("Only experimental difference",["The GAN condition adds the already frozen 40,410 MS-HardGAN-C URLs and targets 10% synthetic exposure; the real-only condition adds none."]),
            ("Supported conclusion",["On the filtered internal test, +10% GAN raises mean F1 from 80.46% to 82.31% and mean recall from 67.35% to 69.96%, while lowering mean FNR from 32.65% to 30.04%.","The F1/recall improvement occurs in two of three paired seeds, so it is promising but not stable."]),
            ("Not supported",["The experiment does not show deployment readiness or temporal generalization.","It does not prove GAN improves ranking quality: mean ROC-AUC and PR-AUC are slightly lower.","The failed confirmation set cannot be rerun; a newly collected calibration and later confirmation are required."]),
        ]
        y=.78
        for heading,items in sections:
            fig.text(.07,y,heading.upper(),fontsize=11,weight="bold",color="#1C9A8B");y-=.045
            for item in items:
                lines=textwrap.wrap(item,125);fig.text(.09,y,"• "+lines[0],fontsize=10.4,color="#17324D")
                for line in lines[1:]:y-=.027;fig.text(.105,y,line,fontsize=10.4,color="#17324D")
                y-=.052
            y-=.015
        pdf.savefig(fig);plt.close(fig)


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"Summary already frozen at {OUTPUT}")
    runs={}
    for condition in ("real_only","gan_10pct"):
        runs[condition]=[load_json(EXPERIMENT/condition/"hyb_ms"/f"seed_{seed}"/"metrics.json") for seed in SEEDS]
    conditions={}
    for condition,rows in runs.items():
        conditions[condition]={metric:{"mean":float(np.mean([row[metric] for row in rows])),
                                       "standard_deviation":float(np.std([row[metric] for row in rows],ddof=1)),
                                       "values":[float(row[metric]) for row in rows]} for metric in METRICS}
        conditions[condition]["thresholds"]=[float(row["threshold"]) for row in rows]
        conditions[condition]["best_epochs"]=[int(row["best_epoch"]) for row in rows]
    deltas={}
    for metric in METRICS:
        values=[runs["gan_10pct"][i][metric]-runs["real_only"][i][metric] for i in range(3)]
        lower_is_better=metric in ("false_positive_rate","false_negative_rate")
        deltas[metric]={"values":[float(v) for v in values],"mean":float(np.mean(values)),
                        "standard_deviation":float(np.std(values,ddof=1)),
                        "gan_wins":sum(v<0 if lower_is_better else v>0 for v in values),
                        "ties":sum(v==0 for v in values)}
    real_report=load_json(DATA/"real_only"/"experiment_report.json")
    gan_report=load_json(DATA/"gan_10pct"/"experiment_report.json")
    summary={"project_version":3,"experiment":"recent_update_gan10_v1","seeds":list(SEEDS),
             "conditions":conditions,"paired_deltas":deltas,
             "data_contract":{"identical_real_rows":real_report["real_training_rows"],
                              "recent_update_rows":real_report["recent_update_rows"],
                              "gan_synthetic_rows":gan_report["synthetic_training_rows"],
                              "validation_rows_removed":real_report["validation_test_overlap_rows_removed"]["validation"],
                              "test_rows_removed":real_report["validation_test_overlap_rows_removed"]["test"],
                              "protected_confirmation_rows_used":0},
             "verdict":"+10% GAN improves mean threshold-dependent internal metrics, including F1 and recall, but loses on seed 44 and does not improve mean ROC-AUC/PR-AUC; external generalization remains untested.",
             "claim_boundary":"internal controlled evidence only; not deployment-ready and not externally confirmed"}
    OUTPUT.mkdir(parents=True);write_json(OUTPUT/"summary.json",summary)
    with (OUTPUT/"all_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["condition","seed",*METRICS,"threshold","best_epoch"]);writer.writeheader()
        for condition,rows in runs.items():
            for row in rows:writer.writerow({"condition":condition,"seed":row["seed"],**{m:row[m] for m in METRICS},"threshold":row["threshold"],"best_epoch":row["best_epoch"]})
    make_figure(OUTPUT/"recent_update_gan10_comparison.png",summary)
    make_pdf(OUTPUT/"recent_update_gan10_report.pdf",summary)
    summary["artifact_hashes"]={name:sha256_file(OUTPUT/name) for name in ("all_metrics.csv","recent_update_gan10_comparison.png","recent_update_gan10_report.pdf")}
    write_json(OUTPUT/"summary.json",summary)
    print(json.dumps({"verdict":summary["verdict"],"real_f1":conditions["real_only"]["f1"],"gan_f1":conditions["gan_10pct"]["f1"],"f1_delta":deltas["f1"]},indent=2))


if __name__=="__main__":main()
