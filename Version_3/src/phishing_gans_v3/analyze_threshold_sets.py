from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .calibrate_two_stage_cascade import optimize_cascade
from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics


CALIBRATION = Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/calibration_predictions.npz")
FREEZE = Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/frozen_cascade.json")
EVALUATION = Path("artifacts/results/two_stage_url_cascade_v1/predictions.npz")
EVALUATION_RESULTS = Path("artifacts/results/two_stage_url_cascade_v1/results.json")
OUTPUT = Path("artifacts/results/threshold_sets_v2")
CAPS = (.001, .0025, .005, .01, .02, .05)
NAMES = ("Ultra-strict", "Strict", "Conservative", "Balanced", "Recall-first", "High-recall")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")


def projected_precision(recall: float, fpr: float, prevalence: float) -> float:
    numerator = recall * prevalence
    return numerator / max(1e-15, numerator + fpr * (1 - prevalence))


def recall_threshold(labels: np.ndarray, scores: np.ndarray, maximum_fpr: float) -> dict:
    """Use the FPR budget to maximize recall; break ties by F1, then lower FPR."""
    order=np.argsort(-scores,kind="stable");y=labels[order];s=scores[order]
    tp=np.cumsum(y==1);fp=np.cumsum(y==0);positives=int((labels==1).sum());negatives=int((labels==0).sum())
    recall=tp/max(1,positives);fpr=fp/max(1,negatives);precision=tp/np.maximum(1,tp+fp)
    f1=2*precision*recall/np.maximum(1e-15,precision+recall);distinct=np.r_[s[1:]!=s[:-1],True]
    valid=np.flatnonzero(distinct&(fpr<=maximum_fpr))
    if not len(valid):raise RuntimeError(f"No threshold satisfies FPR cap {maximum_fpr}")
    index=max(valid,key=lambda i:(recall[i],f1[i],-fpr[i]))
    return {"threshold":float(s[index]),"recall":float(recall[index]),"fpr":float(fpr[index]),"f1":float(f1[index])}


def q(values: np.ndarray) -> dict:
    if not len(values): return {"rows": 0, "minimum": None, "q25": None, "median": None, "q75": None, "maximum": None}
    quantiles = np.quantile(values, (0, .25, .5, .75, 1))
    return {"rows": int(len(values)), **{k: float(v) for k, v in zip(("minimum", "q25", "median", "q75", "maximum"), quantiles)}}


def cascade_metrics(labels: np.ndarray, stage1: np.ndarray, stage2: np.ndarray,
                    screen: float, hard: float, verifier: float) -> tuple[dict, np.ndarray, np.ndarray]:
    routed = stage1 >= screen; immediate = stage1 >= hard
    prediction = immediate | (routed & (stage2 >= verifier))
    ranking = np.where(immediate, 2.0, np.where(routed, stage2, -1.0)).astype(np.float32)
    metrics = classification_metrics(labels, prediction.astype(np.float32), .5)
    ranking_metrics = classification_metrics(labels, ranking, .5)
    metrics["roc_auc"] = ranking_metrics["roc_auc"]; metrics["pr_auc"] = ranking_metrics["pr_auc"]
    return metrics, prediction, ranking


def error_profile(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "false_positive_scores": q(scores[(labels == 0) & predictions]),
        "false_negative_scores": q(scores[(labels == 1) & ~predictions]),
        "true_positive_scores": q(scores[(labels == 1) & predictions]),
        "true_negative_scores": q(scores[(labels == 0) & ~predictions]),
    }


def bands(labels: np.ndarray, scores: np.ndarray, thresholds: dict[float, float]) -> list[dict]:
    boundaries = [thresholds[.05], thresholds[.01], thresholds[.0025], thresholds[.001]]
    specs = [
        ("Low", -np.inf, boundaries[0]),
        ("Uncertain", boundaries[0], boundaries[1]),
        ("Likely phishing", boundaries[1], boundaries[2]),
        ("High risk", boundaries[2], boundaries[3]),
        ("Extreme risk", boundaries[3], np.inf),
    ]
    output=[]
    for name,lower,upper in specs:
        mask=(scores>=lower)&(scores<upper);legit=int(((labels==0)&mask).sum());phish=int(((labels==1)&mask).sum())
        output.append({"band":name,"lower":None if np.isneginf(lower) else float(lower),"upper":None if np.isposinf(upper) else float(upper),
                       "rows":int(mask.sum()),"legitimate":legit,"phishing":phish,"observed_phishing_fraction":phish/max(1,int(mask.sum()))})
    return output


def pct(value: float) -> str:return f"{100*value:.2f}%"


def table(axis, columns, rows, widths=None, font=8.3):
    axis.axis("off");value=axis.table(cellText=rows,colLabels=columns,cellLoc="center",loc="center",colWidths=widths)
    value.auto_set_font_size(False);value.set_fontsize(font);value.scale(1,1.65)
    for (r,_),cell in value.get_celld().items():
        cell.set_edgecolor("#C8D3DE");cell.set_facecolor("#365F7F" if r==0 else ("#E7F3F1" if r%2==0 else "#F5F7FA"))
        if r==0:cell.set_text_props(color="white",weight="bold")
    return value


def make_graph(path: Path, payload: dict) -> None:
    points=payload["operating_points"];x=np.arange(len(points));labels=[p["name"]+f"\n{100*p['calibration_fpr_cap']:.2f}%" for p in points]
    fig,axes=plt.subplots(2,2,figsize=(16,10));colors={"single":"#4C78A8","cascade":"#1C9A8B"}
    for key,title in (("single_stage","Single-stage GAN"),("cascade","Two-stage cascade")):
        axes[0,0].plot(x,[100*p[key]["recall"] for p in points],marker="o",label=title,color=colors["single" if key=="single_stage" else "cascade"])
        axes[0,1].plot(x,[100*p[key]["false_negative_rate"] for p in points],marker="o",label=title,color=colors["single" if key=="single_stage" else "cascade"])
        axes[1,0].plot(x,[100*p[key]["false_positive_rate"] for p in points],marker="o",label=title,color=colors["single" if key=="single_stage" else "cascade"])
        axes[1,1].plot(x,[100*p[key]["precision"] for p in points],marker="o",label=title,color=colors["single" if key=="single_stage" else "cascade"])
    for ax,title,ylabel in zip(axes.ravel(),("Recall rises as thresholds fall","False-negative rate","Observed legitimate false alarms","Precision"),("Recall (%)","FNR (%)","FPR (%)","Precision (%)")):
        ax.set_xticks(x,labels,rotation=12,ha="right");ax.set_title(title,weight="bold");ax.set_ylabel(ylabel);ax.grid(alpha=.22);ax.legend(frameon=False)
    fig.suptitle("Calibration-locked threshold sets on the 20K identity-disjoint cohort",fontsize=18,weight="bold",color="#17324D")
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(path,dpi=220,bbox_inches="tight");plt.close(fig)


def make_pdf(path: Path, payload: dict) -> None:
    points=payload["operating_points"]
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        fig.text(.055,.93,"Threshold-Set Diagnostic Study",fontsize=24,weight="bold",color="#17324D")
        fig.text(.055,.875,"Frozen HYB-MS +10% GAN and two-stage cascade • 20,000 identity-disjoint URLs",fontsize=11.5,color="#526274")
        cols=["Set","Cal. cap","System","Threshold","Precision","Recall","F1","FPR","FNR"]
        rows=[]
        for p in points:
            rows.append([p["name"],pct(p["calibration_fpr_cap"]),"Single",f"{p['single_stage_threshold']:.6f}",*[pct(p["single_stage"][k]) for k in ("precision","recall","f1","false_positive_rate","false_negative_rate")]])
            rows.append(["","","Cascade",("∞" if math.isinf(p["verifier_threshold"]) else f"{p['verifier_threshold']:.6f}"),*[pct(p["cascade"][k]) for k in ("precision","recall","f1","false_positive_rate","false_negative_rate")]])
        table(fig.add_axes([.025,.25,.95,.56]),cols,rows,widths=[.13,.08,.09,.12,.11,.10,.10,.10,.10],font=7.4)
        fig.text(.055,.17,textwrap.fill(payload["headline"],112),fontsize=11.2,weight="bold",color="#17324D")
        fig.text(.055,.07,textwrap.fill("Thresholds were derived only from calibration data. Because this 20K cohort was previously evaluated, this sweep is diagnostic and cannot be presented as a new untouched confirmation.",135),fontsize=9.5,color="#C26B32")
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(11,8.5));fig.patch.set_facecolor("white");x=np.arange(len(points));labels=[p["name"]+f"\n{100*p['calibration_fpr_cap']:.2f}%" for p in points]
        for key,title,color in (("single_stage","Single-stage GAN","#4C78A8"),("cascade","Cascade","#1C9A8B")):
            for ax,metric in zip(axes.ravel(),("recall","false_negative_rate","false_positive_rate","f1")):ax.plot(x,[100*p[key][metric] for p in points],marker="o",label=title,color=color)
        for ax,title in zip(axes.ravel(),("Recall","FNR","FPR","F1")):
            ax.set_xticks(x,labels,rotation=15,ha="right");ax.set_ylabel("Percent");ax.set_title(title,weight="bold");ax.grid(alpha=.2);ax.legend(frameon=False)
        fig.suptitle("What changes when the trust threshold changes?",fontsize=19,weight="bold",color="#17324D");fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"Single-Stage Score Bands",fontsize=23,weight="bold",color="#17324D")
        cols=["Band","Lower score","Upper score","URLs","Legitimate","Phishing","Observed phish %"]
        rows=[[b["band"],"−∞" if b["lower"] is None else f"{b['lower']:.6f}","+∞" if b["upper"] is None else f"{b['upper']:.6f}",f"{b['rows']:,}",f"{b['legitimate']:,}",f"{b['phishing']:,}",pct(b["observed_phishing_fraction"])] for b in payload["score_bands"]]
        table(fig.add_axes([.055,.45,.89,.34]),cols,rows,widths=[.17,.14,.14,.11,.13,.12,.16],font=8.5)
        fig.text(.06,.34,"Interpretation",fontsize=13,weight="bold",color="#17324D")
        fig.text(.06,.28,textwrap.fill("A phishing URL in the low band is not a threshold problem: even the high-recall threshold gives it a low score. Those cases point to missing representation or distribution coverage. Phishing concentrated just below a threshold can be recovered by lowering that threshold, at the cost shown by the legitimate counts entering higher bands.",135),fontsize=10.5,color="#526274")
        fig.text(.06,.14,textwrap.fill("A score is model confidence under this training/calibration setup, not a guaranteed probability of maliciousness. The observed phishing percentage in each band depends on this deliberately 10%-phishing evaluation prevalence.",135),fontsize=10.5,weight="bold",color="#17324D")
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.055,.93,"Marginal Security Cost",fontsize=23,weight="bold",color="#17324D")
        cols=["Set","Single: extra TP","Single: extra FP","TP per extra FP","Cascade: extra TP","Cascade: extra FP","TP per extra FP"]
        rows=[]
        for p in points:
            s=p["marginal_vs_previous_stricter"]["single_stage"];c=p["marginal_vs_previous_stricter"]["cascade"]
            def ratio(v):return "—" if v is None else ("∞" if math.isinf(v) else f"{v:.2f}")
            rows.append([p["name"],s["additional_true_positives"],s["additional_false_positives"],ratio(s["additional_tp_per_additional_fp"]),c["additional_true_positives"],c["additional_false_positives"],ratio(c["additional_tp_per_additional_fp"])])
        table(fig.add_axes([.035,.43,.93,.36]),cols,rows,widths=[.17,.13,.13,.14,.13,.13,.14],font=8)
        fig.text(.06,.32,"Operational reading",fontsize=13,weight="bold",color="#17324D")
        fig.text(.06,.26,textwrap.fill(payload["recommendation"],135),fontsize=10.7,color="#526274")
        fig.text(.06,.12,"Projected precision at 0.1%, 1%, 5%, and 10% phishing prevalence is stored in the JSON and CSV. Precision observed on this 10% cohort should not be assumed for ordinary browser traffic.",fontsize=10,weight="bold",color="#17324D",wrap=True)
        pdf.savefig(fig);plt.close(fig)


def main() -> None:
    if OUTPUT.exists():raise FileExistsError(f"Threshold analysis already frozen at {OUTPUT}")
    calibration=np.load(CALIBRATION);evaluation=np.load(EVALUATION);freeze=load_json(FREEZE);prior=load_json(EVALUATION_RESULTS)
    cy=calibration["labels"].astype(np.uint8);c1=calibration["stage1_scores"];c2=calibration["stage2_scores"]
    ey=evaluation["labels"].astype(np.uint8);e1=evaluation["stage1_scores"];e2=evaluation["stage2_scores"]
    screen=float(freeze["screening_rule"]["threshold"]);hard=float(freeze["immediate_block_rule"]["threshold"])
    points=[];single_thresholds={}
    for name,cap in zip(NAMES,CAPS):
        single_selection=recall_threshold(cy,c1,cap);single_threshold=float(single_selection["threshold"]);single_thresholds[cap]=single_threshold
        cascade_selection=optimize_cascade(cy,c1,c2,screen,hard,cap);verifier=float(cascade_selection["verifier_threshold"])
        single_prediction=e1>=single_threshold;single_metrics=classification_metrics(ey,e1,single_threshold)
        cascade_result,cascade_prediction,cascade_ranking=cascade_metrics(ey,e1,e2,screen,hard,verifier)
        point={"name":name,"calibration_fpr_cap":cap,"single_stage_threshold":single_threshold,"verifier_threshold":verifier,
            "single_stage":single_metrics,"cascade":cascade_result,
            "projected_precision":{"single_stage":{str(p):projected_precision(single_metrics["recall"],single_metrics["false_positive_rate"],p) for p in (.001,.01,.05,.10)},
                                   "cascade":{str(p):projected_precision(cascade_result["recall"],cascade_result["false_positive_rate"],p) for p in (.001,.01,.05,.10)}},
            "error_score_profiles":{"single_stage_raw_score":error_profile(ey,e1,single_prediction),
                                    "cascade_decision_score":error_profile(ey,cascade_ranking,cascade_prediction)}}
        points.append(point)
    for index,p in enumerate(points):
        p["marginal_vs_previous_stricter"]={}
        for key in ("single_stage","cascade"):
            cm=p[key]["confusion_matrix"]
            if index==0:
                values={"additional_true_positives":None,"additional_false_positives":None,"additional_tp_per_additional_fp":None}
            else:
                old=points[index-1][key]["confusion_matrix"];dtp=cm["tp"]-old["tp"];dfp=cm["fp"]-old["fp"]
                values={"additional_true_positives":dtp,"additional_false_positives":dfp,
                        "additional_tp_per_additional_fp":(math.inf if dfp==0 and dtp>0 else (dtp/dfp if dfp else None))}
            p["marginal_vs_previous_stricter"][key]=values
    score_bands=bands(ey,e1,single_thresholds)
    best_fnr=min((p[key]["false_negative_rate"],p["name"],key,p[key]["false_positive_rate"]) for p in points for key in ("single_stage","cascade"))
    headline=(f"The lowest observed FNR is {pct(best_fnr[0])}: {best_fnr[1]} {best_fnr[2].replace('_',' ')}, with observed FPR {pct(best_fnr[3])}. No tested threshold eliminates false negatives.")
    recommendation=("Use the Strict cascade when approximately 0.1% observed FPR is required. Use the Balanced or Recall-first single-stage setting when reducing missed phishing is more important and the extension can tolerate additional warnings. Treat High-recall as a diagnostic boundary, not an automatic deployment default.")
    payload={"project_version":3,"experiment":"threshold_sets_v2","status":"POST-HOC DIAGNOSTIC ON PREVIOUSLY VIEWED 20K COHORT",
        "calibration_only_selection":True,"caps":list(CAPS),"operating_points":points,"score_bands":score_bands,
        "headline":headline,"recommendation":recommendation,
        "score_semantics":"Scores are model outputs used for ranking; they are not guaranteed calibrated probabilities.",
        "auc_semantics":"Single-stage AUC uses Stage-1 probabilities. Cascade AUC uses immediate-block=2, routed=Stage-2 score, non-routed=-1.",
        "artifact_hashes":{"calibration_predictions":sha256_file(CALIBRATION),"cascade_freeze":sha256_file(FREEZE),
                           "evaluation_predictions":sha256_file(EVALUATION),"prior_evaluation_results":sha256_file(EVALUATION_RESULTS)},
        "threshold_selection_rule":"For each system and cap, maximize calibration recall subject to FPR <= cap; break ties by F1 and then lower FPR.",
        "test_integrity_note":"The 20K labels had already been viewed in the preceding cascade experiment. Thresholds here are calibration-derived, but these comparisons are diagnostic rather than a new confirmation."}
    OUTPUT.mkdir(parents=True);write_json(OUTPUT/"threshold_analysis.json",payload)
    with (OUTPUT/"all_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["name","calibration_fpr_cap","system","threshold",*METRICS,"tp","fp","tn","fn","projected_precision_0.1pct","projected_precision_1pct","projected_precision_5pct","projected_precision_10pct"]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for p in points:
            for system,key,threshold in (("single_stage","single_stage",p["single_stage_threshold"]),("cascade","cascade",p["verifier_threshold"])):
                m=p[key];cm=m["confusion_matrix"];proj=p["projected_precision"][key]
                writer.writerow({"name":p["name"],"calibration_fpr_cap":p["calibration_fpr_cap"],"system":system,"threshold":threshold,
                    **{k:m[k] for k in METRICS},**cm,"projected_precision_0.1pct":proj["0.001"],"projected_precision_1pct":proj["0.01"],"projected_precision_5pct":proj["0.05"],"projected_precision_10pct":proj["0.1"]})
    make_graph(OUTPUT/"threshold_tradeoffs.png",payload);make_pdf(OUTPUT/"threshold_sets_report.pdf",payload)
    write_json(OUTPUT/"artifact_manifest.json",{name:sha256_file(OUTPUT/name) for name in ("threshold_analysis.json","all_metrics.csv","threshold_tradeoffs.png","threshold_sets_report.pdf")})
    print(json.dumps({"headline":headline,"recommendation":recommendation,"operating_points":[{"name":p["name"],"cap":p["calibration_fpr_cap"],"single":p["single_stage"],"cascade":p["cascade"]} for p in points]},indent=2))


if __name__=="__main__":main()
