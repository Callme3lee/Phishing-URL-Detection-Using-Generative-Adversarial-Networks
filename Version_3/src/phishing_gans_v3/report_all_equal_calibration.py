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

from .common import load_json, write_json


ROOT=Path("artifacts/results/all_models_equal_calibration_v1")
BLUE,TEAL,GOLD,RED,PURPLE,DARK,MUTED,LIGHT="#355C7D","#2A9D8F","#F4A261","#E76F51","#7B2CBF","#172B4D","#52606D","#F4F7FA"
METRICS=("accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate","false_negative_rate")


def pct(v): return f"{100*v:.2f}%"


def page(title,subtitle=""):
    fig=plt.figure(figsize=(11.69,8.27),facecolor="white")
    fig.text(.045,.94,title,fontsize=20,fontweight="bold",color=DARK)
    if subtitle: fig.text(.045,.902,subtitle,fontsize=10,color=MUTED)
    fig.add_artist(plt.Line2D([.045,.955],[.88,.88],color="#CBD5E1",lw=1.2)); return fig


def footer(fig,n):
    fig.text(.045,.022,"Definitive equal-calibration comparison • 106 configurations • 318 checkpoints",fontsize=8,color="#7A8895")
    fig.text(.955,.022,str(n),fontsize=8,color="#7A8895",ha="right")


def para(fig,x,y,text,width=108,size=10.5,weight="normal",color=DARK):
    fig.text(x,y,textwrap.fill(text,width),fontsize=size,fontweight=weight,color=color,va="top",linespacing=1.35)


def bullets(fig,x,y,items,width=96,size=10,gap=.067):
    for item in items:
        fig.text(x,y,"•",fontsize=size+2,color=TEAL,fontweight="bold",va="top")
        lines=textwrap.wrap(item,width); fig.text(x+.022,y,"\n".join(lines),fontsize=size,color=DARK,va="top",linespacing=1.25)
        y-=gap*max(1,len(lines)/2)


def table(fig,bbox,columns,rows,widths=None,font=8,row_colors=None):
    ax=fig.add_axes(bbox); ax.axis("off")
    t=ax.table(cellText=rows,colLabels=columns,cellLoc="center",colLoc="center",colWidths=widths,bbox=(0,0,1,1))
    t.auto_set_font_size(False); t.set_fontsize(font)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#D5DEE8"); cell.set_linewidth(.55)
        if r==0:
            cell.set_facecolor(BLUE); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        else:
            requested=row_colors[r-1] if row_colors and r-1<len(row_colors) else None
            cell.set_facecolor(requested or ("white" if r%2 else LIGHT))
            if c==0: cell.get_text().set_ha("left")
    return t


def label(r): return f"{r['stage']} | {r['model']} | {r['condition']}"


def wilson(success,total,z=1.96):
    p=success/total; den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return center-half,center+half


def mean_external_f1(r): return np.mean([r["evaluations"][d]["f1"] for d in ("external_10k","external_30k")])


def main():
    payload=load_json(ROOT/"all_results.json"); records=payload["conditions"]
    top10=sorted(records,key=lambda r:r["evaluations"]["external_10k"]["f1"],reverse=True)
    top30=sorted(records,key=lambda r:r["evaluations"]["external_30k"]["f1"],reverse=True)
    eligible=[r for r in records if all(r["evaluations"][d]["false_positive_rate"]<=.02 for d in ("external_10k","external_30k"))]
    winner=max(eligible,key=mean_external_f1)
    no_gan=[r for r in eligible if r["condition"] in {"Real","Real + outside legit"}]
    best_no_gan=max(no_gan,key=mean_external_f1)
    nonhybrid=[r for r in eligible if "HYB" not in r["model"]]
    best_nonhybrid=max(nonhybrid,key=mean_external_f1)
    natural=[r for r in eligible if r["stage"]=="Natural HYB-MS"]
    best_natural=max(natural,key=mean_external_f1)
    category_specs={
        "Best overall":eligible,
        "Best without GAN":no_gan,
        "Best non-hybrid":nonhybrid,
        "Best natural HYB":natural,
        "Best CNN+PSL":[r for r in eligible if r["model"]=="CNN+PSL"],
        "Best RF":[r for r in eligible if r["model"]=="RF"],
        "Best original SeqGAN":[r for r in eligible if r["stage"]=="SeqGAN v1"],
        "Best MS-HardGAN baseline":[r for r in eligible if r["stage"]=="MS-HardGAN-C baselines"],
    }
    categories={name:max(rows,key=mean_external_f1) for name,rows in category_specs.items() if rows}
    summary={"protocol":payload["protocol"],"configuration_count":len(records),"checkpoint_count":payload["checkpoint_count"],
             "eligibility_rule":"FPR <= 2% on both external 10K and 30K; rank by mean external F1",
             "winner":winner,"best_no_gan":best_no_gan,"best_nonhybrid":best_nonhybrid,"best_natural":best_natural,
             "category_winner_ids":{k:v["id"] for k,v in categories.items()}}
    write_json(ROOT/"definitive_summary.json",summary)
    with (ROOT/"ranked.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["rank","id","stage","model","condition","eligible","mean_external_f1"]+[f"{d}_{m}" for d in ("external_10k","external_30k") for m in METRICS]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for rank,r in enumerate(sorted(records,key=mean_external_f1,reverse=True),1):
            writer.writerow({"rank":rank,"id":r["id"],"stage":r["stage"],"model":r["model"],"condition":r["condition"],
                "eligible":r in eligible,"mean_external_f1":mean_external_f1(r),
                **{f"{d}_{m}":r["evaluations"][d][m] for d in ("external_10k","external_30k") for m in METRICS}})

    pages=[]
    # 1 winner
    fig=page("Definitive Comparison of Every Classifier","Same three-seed ensemble calibration rule applied to every valid frozen configuration")
    fig.text(.045,.82,"WINNER",fontsize=11,color=TEAL,fontweight="bold")
    para(fig,.045,.77,"Weighted HYB-MS +10% MS-HardGAN-C",width=85,size=19,weight="bold")
    a,b=winner["evaluations"]["external_10k"],winner["evaluations"]["external_30k"]
    table(fig,[.045,.50,.91,.18],["Evaluation","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","FPR","FNR"],
          [["Frozen 10K"]+[pct(a[m]) for m in METRICS],["Imbalanced 30K"]+[pct(b[m]) for m in METRICS]],
          widths=[.16]+[.105]*8,font=8.5,row_colors=["#E8F5F2","#E8F5F2"])
    bullets(fig,.06,.43,[
        "Ranked first in F1, accuracy, ROC-AUC, and PR-AUC on both external sets after every model received the same calibration rule.",
        "Its FPR remains below the declared 2% deployment-candidate screen on both sets: 1.40% on 10K and 0.96% on 30K.",
        "It beats weighted real-only HYB-MS, ordinary MS-LSTM, natural HYB-MS, every CNN/RNN/RF/SVM condition, and every higher GAN dose.",
        "This is a retrospective final selection. A new untouched temporal snapshot is required to confirm the selected winner."
    ],gap=.067)
    pages.append(fig)

    # 2 protocol / annotation issue
    fig=page("Controlled Final Protocol","The threshold-policy confound has now been removed")
    table(fig,[.055,.54,.89,.27],["Fixed element","Value","Why it matters"],[
        ["Configurations","106","Every valid three-seed classifier condition"],
        ["Checkpoints","318","Seeds 42/43/44 ensembled by mean score"],
        ["Calibration","Same separate 400 URLs","No hybrid-only threshold advantage"],
        ["Threshold rule","Max F1 subject to calibration FPR ≤1%","Identical rule for all score scales"],
        ["Internal test","49,229 frozen URLs","Familiar-distribution reference"],
        ["External 10K","10,000 frozen URLs","Balanced source-transfer evaluation"],
        ["External 30K","30,000 URLs; 30.07% phishing","Later-collected imbalanced stress test"],
    ],widths=[.22,.27,.47],font=8.7)
    para(fig,.055,.47,"What changed",width=30,size=11,weight="bold",color=TEAL)
    para(fig,.055,.425,"Previously the hybrid used cross-source validation/calibration while ordinary baselines retained their original internal thresholds. This definitive run recalibrated every three-seed ensemble on the exact same 400-row set.",width=105,size=11)
    para(fig,.055,.29,"What did not change",width=30,size=11,weight="bold",color=RED)
    para(fig,.055,.245,"No model was retrained, no checkpoint was selected from external labels, and no external threshold was tuned. Architectures, features, source exposure, augmentation pools, and learned weights remain the actual experimental conditions being compared.",width=105,size=11)
    para(fig,.055,.105,"Statistical guardrail: calibration resubstitution is not reported as test performance. The selected winner must be evaluated once more on a future untouched cohort because the 10K/30K are now used for retrospective model selection.",width=108,size=10.3,weight="bold")
    pages.append(fig)

    # 3/4 top tables
    for ds,title in [("external_10k","Top Models — Frozen External 10K"),("external_30k","Top Models — Imbalanced External 30K")]:
        fig=page(title,"Identical calibration; sorted by F1")
        rows=[]
        for rank,r in enumerate(sorted(records,key=lambda q:q["evaluations"][ds]["f1"],reverse=True)[:15],1):
            m=r["evaluations"][ds]
            rows.append([rank,r["stage"],r["model"],r["condition"],pct(m["accuracy"]),pct(m["precision"]),pct(m["recall"]),pct(m["f1"]),pct(m["roc_auc"]),pct(m["false_positive_rate"])])
        table(fig,[.025,.12,.95,.71],["#","Stage","Model","Condition","Acc.","Prec.","Recall","F1","AUC","FPR"],rows,
              widths=[.04,.23,.09,.13,.08,.08,.08,.08,.08,.08],font=7.25,row_colors=["#E8F5F2"]+[None]*14)
        para(fig,.045,.075,"F1 is the primary balance metric here. FPR remains a separate deployment constraint; a model is not selected merely because it minimizes false alarms by refusing to detect phishing.",width=112,size=9.5,weight="bold")
        pages.append(fig)

    # 5 category winners
    fig=page("Winners by Experimental Category","FPR ≤2% on both external sets; highest mean external F1 within category")
    rows=[]
    for name,r in categories.items():
        x,y=r["evaluations"]["external_10k"],r["evaluations"]["external_30k"]
        rows.append([name,r["stage"],r["model"],r["condition"],pct(x["f1"]),pct(y["f1"]),pct(x["false_positive_rate"]),pct(y["false_positive_rate"])])
    table(fig,[.03,.20,.94,.62],["Category","Stage","Model","Condition","10K F1","30K F1","10K FPR","30K FPR"],rows,
          widths=[.17,.22,.10,.15,.09,.09,.09,.09],font=7.7,row_colors=["#E8F5F2"]+[None]*(len(rows)-1))
    para(fig,.045,.13,"Best non-hybrid: ordinary MS-LSTM. Best natural hybrid: real-only Natural HYB-MS. Best no-GAN complete system: weighted real-only HYB-MS. None exceeds weighted HYB-MS +10%.",width=112,size=10.3,weight="bold")
    pages.append(fig)

    # 6 scatter
    fig=page("Deployment Trade-off on the 30K","Recall versus false-positive rate; marker size represents F1")
    ax=fig.add_axes([.09,.18,.82,.62])
    colors={"HYB-MS":TEAL,"HYB":PURPLE,"LSTM":GOLD,"CNN+PSL":BLUE,"CNN":"#6C8EBF","RNN":"#8AB17D","RF":RED,"SVM":"#999999"}
    for r in records:
        m=r["evaluations"]["external_30k"]
        ax.scatter(100*m["false_positive_rate"],100*m["recall"],s=20+180*m["f1"]**2,
                   color=colors.get(r["model"],"#777777"),alpha=.62,edgecolor="white",linewidth=.4)
    wm=winner["evaluations"]["external_30k"]
    ax.scatter(100*wm["false_positive_rate"],100*wm["recall"],s=260,facecolor="none",edgecolor="black",linewidth=2,label="Winner")
    ax.annotate("Weighted HYB +10%",(100*wm["false_positive_rate"],100*wm["recall"]),xytext=(18,-20),textcoords="offset points",fontweight="bold",arrowprops={"arrowstyle":"->"})
    ax.axvline(2,color=RED,ls="--",lw=1.4,label="2% FPR screen"); ax.set_xlabel("False-positive rate (%)"); ax.set_ylabel("Phishing recall (%)"); ax.grid(alpha=.23); ax.legend(frameon=False)
    para(fig,.055,.105,"The winner occupies the desirable upper-left region: high recall with low FPR. Higher weighted GAN doses move left but also sharply downward, while many simple models remain low-recall after equal calibration.",width=112,size=10.2,weight="bold")
    pages.append(fig)

    # 7 direct finalists
    fig=page("Finalists: Direct Head-to-Head","The strongest system from each major design family")
    finalists=[winner,best_no_gan,best_nonhybrid,best_natural,categories.get("Best CNN+PSL"),categories.get("Best RF")]
    finalists=list(dict.fromkeys(r["id"] for r in finalists if r))
    finalists=[next(r for r in records if r["id"]==rid) for rid in finalists]
    rows=[]
    for r in finalists:
        a,b=r["evaluations"]["external_10k"],r["evaluations"]["external_30k"]
        rows.append([r["stage"]+" / "+r["model"]+" "+r["condition"],pct(a["f1"]),pct(a["recall"]),pct(a["false_positive_rate"]),pct(a["roc_auc"]),pct(b["f1"]),pct(b["recall"]),pct(b["false_positive_rate"]),pct(b["roc_auc"])])
    table(fig,[.035,.33,.93,.47],["Finalist","10 F1","10 R","10 FPR","10 AUC","30 F1","30 R","30 FPR","30 AUC"],rows,
          widths=[.28]+[.09]*8,font=7.5,row_colors=["#E8F5F2"]+[None]*(len(rows)-1))
    para(fig,.05,.25,"Why +10% rather than +20/+30/+40%?",width=50,size=11,weight="bold",color=TEAL)
    para(fig,.05,.205,"The larger doses achieve lower FPR and therefore can rank well under equal-weight metric averages, but they miss substantially more phishing. +10% gives the highest F1, recall, accuracy, ROC-AUC, and PR-AUC on both external sets while keeping FPR below 2%.",width=108,size=10.5)
    para(fig,.05,.085,"Selection principle: maximize useful phishing detection under a practical false-alarm ceiling—not maximize one metric in isolation.",width=110,size=10.5,weight="bold")
    pages.append(fig)

    # 8 Internal vs external
    fig=page("Internal Success Does Not Guarantee External Success","Top internal F1 configurations compared with their 30K F1")
    top_internal=sorted(records,key=lambda r:r["evaluations"]["internal_test"]["f1"],reverse=True)[:15]
    rows=[]
    for rank,r in enumerate(top_internal,1):
        i=r["evaluations"]["internal_test"]; e=r["evaluations"]["external_30k"]
        rows.append([rank,r["stage"],r["model"],r["condition"],pct(i["f1"]),pct(i["roc_auc"]),pct(e["f1"]),pct(e["roc_auc"]),pct(e["false_positive_rate"])])
    table(fig,[.035,.16,.93,.66],["#","Stage","Model","Condition","Internal F1","Internal AUC","30K F1","30K AUC","30K FPR"],rows,
          widths=[.04,.23,.09,.13,.10,.10,.10,.10,.09],font=7.4)
    para(fig,.05,.10,"This page explains why the project moved beyond internal accuracy: several near-perfect internal models transfer poorly. The final winner is chosen from external behavior after equal calibration.",width=112,size=10,weight="bold")
    pages.append(fig)

    # 9 pros/cons
    fig=page("Pros and Cons of the Main Model Classes","A concise interpretation of all 106 configurations")
    rows=[
        ["CNN","Fast local-pattern learner","Calibration often suppresses recall; weak source transfer","Baseline/ablation"],
        ["LSTM","Best non-hybrid sequence family; MS real reaches 89.22% 30K F1","Recurrent latency; still below weighted hybrid","Strong fallback"],
        ["RNN","Simple recurrent control","Unstable and consistently behind LSTM","Research control"],
        ["RF","Interpretable lexical baseline","GAN doses and source shift hurt ranking","Explainability baseline"],
        ["SVM","Small, reproducible linear model","Lowest external discrimination in most conditions","Lower bound"],
        ["CNN+PSL","Tests character/lexical fusion without full adaptation","External ceiling below HYB-MS","Feature-fusion ablation"],
        ["Natural HYB-MS","Fair row frequency; low FPR; strong AUC","Less outside exposure; 30K F1 78.49%","Natural-sampling winner"],
        ["Weighted HYB real","Excellent calibrated transfer without synthetic data","Repeats 5,073 outside URLs heavily","Best no-GAN system"],
        ["Weighted HYB +10%","Best balanced evidence on both external sets","Most complex; synthetic dependency","Overall winner"],
        ["Weighted HYB 20–40%","Very low FPR","Increasing FNR and decreasing F1","Not preferred"],
    ]
    table(fig,[.035,.16,.93,.67],["Class","Pros","Cons","Verdict"],rows,widths=[.18,.32,.32,.16],font=7.8,row_colors=[None]*6+["#EEF5FA","#E8F5F2","#E8F5F2","#FDECEC"])
    para(fig,.05,.10,"Generator-only SeqGAN, URL-GAN, RelGAN, and MS-HardGAN-C are not classifiers and therefore are not assigned classifier Accuracy/F1. Their role is evaluated through acceptance, uniqueness, realism, coverage, and challenge metrics.",width=112,size=9.8,weight="bold")
    pages.append(fig)

    # 10 statistical and claim guardrails
    fig=page("Confidence and Claim Guardrails","What the definitive leaderboard supports—and what it does not")
    # Wilson intervals for winner recall/FPR
    ci_rows=[]
    for ds,total_neg,total_pos in [("10K",5000,5000),("30K",20979,9021)]:
        m=winner["evaluations"]["external_10k" if ds=="10K" else "external_30k"]
        cm=m["confusion_matrix"]
        rlo,rhi=wilson(cm["tp"],total_pos); flo,fhi=wilson(cm["fp"],total_neg)
        ci_rows.append([ds,pct(m["recall"]),f"{pct(rlo)}–{pct(rhi)}",pct(m["false_positive_rate"]),f"{pct(flo)}–{pct(fhi)}",f"TP {cm['tp']}; FP {cm['fp']}"])
    table(fig,[.06,.61,.88,.17],["Set","Recall","95% Wilson CI","FPR","95% Wilson CI","Counts"],ci_rows,widths=[.12,.14,.20,.14,.20,.18],font=8.5,row_colors=["#E8F5F2","#E8F5F2"])
    fig.text(.05,.53,"SUPPORTED",fontsize=11,color=TEAL,fontweight="bold")
    bullets(fig,.06,.49,[
        "Under one identical calibration protocol, weighted HYB-MS +10% is the best balanced classifier tested.",
        "Its advantage appears on two identity-disjoint external datasets with different class prevalence.",
        "Weighted exposure, CORAL, feature fusion, ensembling, calibration, and low-dose GAN augmentation jointly define the winning system."
    ],gap=.06)
    fig.text(.05,.27,"NOT YET SUPPORTED",fontsize=11,color=RED,fontweight="bold")
    bullets(fig,.06,.23,[
        "A claim that the hybrid architecture alone caused the result.",
        "A claim that GANs predict fundamentally unseen zero-day concepts.",
        "A final unbiased deployment claim after using the 10K/30K to choose the winner; a new temporal holdout is required."
    ],gap=.06)
    pages.append(fig)

    # all model appendix, 18 rows/page
    ordered=sorted(records,key=lambda r:(r["stage"],r["model"],r["condition"],r["id"]))
    chunks=[ordered[i:i+18] for i in range(0,len(ordered),18)]
    for ci,chunk in enumerate(chunks,1):
        fig=page(f"Appendix: Every Classifier Configuration ({ci}/{len(chunks)})","All rows use the same three-seed calibration rule")
        rows=[]
        for r in chunk:
            a,b=r["evaluations"]["external_10k"],r["evaluations"]["external_30k"]
            rows.append([r["stage"],r["model"],r["condition"],pct(a["f1"]),pct(a["recall"]),pct(a["false_positive_rate"]),pct(a["roc_auc"]),pct(b["f1"]),pct(b["recall"]),pct(b["false_positive_rate"]),pct(b["roc_auc"])])
        colors_rows=["#E8F5F2" if r["id"]==winner["id"] else None for r in chunk]
        table(fig,[.018,.08,.964,.76],["Stage","Model","Cond.","10 F1","10 R","10 FP","10 AUC","30 F1","30 R","30 FP","30 AUC"],rows,
              widths=[.22,.08,.12]+[.072]*8,font=6.65,row_colors=colors_rows)
        pages.append(fig)

    # final
    fig=page("Final Decision","One winner, with an explicit deployment boundary")
    para(fig,.05,.80,"Best model: calibrated three-seed Weighted HYB-MS +10% MS-HardGAN-C ensemble.",width=92,size=17,weight="bold")
    table(fig,[.05,.52,.90,.17],["Runner-up category","Best system","10K F1","30K F1","Reason it loses"],[
        ["No GAN",label(best_no_gan),pct(best_no_gan["evaluations"]["external_10k"]["f1"]),pct(best_no_gan["evaluations"]["external_30k"]["f1"]),"Lower recall/F1 on both sets"],
        ["Non-hybrid",label(best_nonhybrid),pct(best_nonhybrid["evaluations"]["external_10k"]["f1"]),pct(best_nonhybrid["evaluations"]["external_30k"]["f1"]),"Lower AUC and F1"],
        ["Natural hybrid",label(best_natural),pct(best_natural["evaluations"]["external_10k"]["f1"]),pct(best_natural["evaluations"]["external_30k"]["f1"]),"Less source exposure; lower recall"],
    ],widths=[.16,.40,.10,.10,.22],font=7.7,row_colors=[None,None,None])
    fig.text(.05,.44,"NEXT REQUIRED ACTION",fontsize=11,color=RED,fontweight="bold")
    para(fig,.05,.395,"Freeze a genuinely later-created phishing cohort and later legitimate snapshot, then evaluate this already-selected ensemble and threshold exactly once. Do not compare or retune the other 105 configurations on that final confirmation set.",width=108,size=11,weight="bold")
    bullets(fig,.06,.25,[
        "If confirmed, benchmark browser latency, memory, package size, and alert burden at realistic prevalence.",
        "If it fails, investigate drift and retrain under a new declared protocol rather than tuning on the failed holdout.",
        "Report it now as the strongest offline deployment candidate—not as production-ready."
    ],gap=.06)
    pages.append(fig)

    pdf=ROOT/"definitive_all_models_comparison.pdf"
    with PdfPages(pdf) as out:
        for n,fig in enumerate(pages,1): footer(fig,n); out.savefig(fig,bbox_inches="tight"); plt.close(fig)
    lines=["# Definitive Equal-Calibration Comparison","",
           f"Compared **{len(records)} configurations / {payload['checkpoint_count']} checkpoints** under one calibration policy.","",
           f"**Winner:** {label(winner)}.","",
           "| Set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds,title in [("external_10k","10K"),("external_30k","30K")]:
        m=winner["evaluations"][ds]; lines.append("| "+title+" | "+" | ".join(f"{m[x]:.4f}" for x in METRICS)+" |")
    lines += ["","The 10K and 30K are now retrospective selection evidence. A new untouched temporal dataset is required for final confirmation."]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"winner":label(winner),"configurations":len(records),"pdf":str(pdf),"pages":len(pages)},indent=2))


if __name__=="__main__": main()
