from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, write_json


OUT = Path("artifacts/reports/complete_project_comparison_2026-08-13")
BLUE, TEAL, GOLD, RED, PURPLE = "#355C7D", "#2A9D8F", "#F4A261", "#E76F51", "#7B2CBF"
LIGHT, DARK, MUTED = "#F4F7FA", "#172B4D", "#52606D"
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
SHORT = {"accuracy": "Acc.", "precision": "Prec.", "recall": "Recall", "f1": "F1",
         "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR",
         "false_negative_rate": "FNR"}


def page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(.055, .935, title, fontsize=21, fontweight="bold", color=DARK)
    if subtitle:
        fig.text(.055, .895, subtitle, fontsize=10.5, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.875, .875], color="#CBD5E1", lw=1.3))
    return fig


def footer(fig, number: int):
    fig.text(.055, .025, "Version 3 • Complete comparison • frozen evidence through 2026-08-13",
             fontsize=8.5, color="#7A8895")
    fig.text(.945, .025, str(number), fontsize=8.5, color="#7A8895", ha="right")


def paragraph(fig, x, y, text, width=105, size=11, color=DARK, weight="normal", spacing=1.35):
    fig.text(x, y, textwrap.fill(text, width), fontsize=size, color=color, fontweight=weight,
             va="top", linespacing=spacing)


def bullet_list(fig, x, y, items, width=92, size=10.5, gap=.071, color=DARK):
    for item in items:
        fig.text(x, y, "•", fontsize=size + 2, color=TEAL, va="top", fontweight="bold")
        fig.text(x + .022, y, textwrap.fill(item, width), fontsize=size, color=color, va="top", linespacing=1.28)
        y -= gap * max(1, len(textwrap.wrap(item, width)) / 2.1)
    return y


def table(fig, bbox, columns, rows, widths=None, font=8.4, header_color=BLUE, row_colors=None):
    ax = fig.add_axes(bbox); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
                   colWidths=widths, bbox=(0, 0, 1, 1))
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5DEE8"); cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor(header_color); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        else:
            requested = row_colors[r - 1] if row_colors and r - 1 < len(row_colors) else None
            color = requested if requested else ("#FFFFFF" if r % 2 else LIGHT)
            cell.set_facecolor(color)
            if c == 0: cell.get_text().set_ha("left")
    return tbl


def mean_metrics(group):
    return {metric: group[metric]["mean"] for metric in METRICS}


def pct(v): return f"{100*v:.2f}%"


def load_evidence():
    natural = load_json(Path("artifacts/results/hybrid_ms_natural_fair_v1/summary.json"))
    weighted10 = load_json(Path("artifacts/results/fair_hybrid_gan_v1/calibrated_ensembles.json"))["results"]
    weighted30 = load_json(Path("artifacts/results/external/2026-08-13-imbalanced-30k/calibrated_hybrid_ensembles.json"))["results"]
    ms = load_json(Path("artifacts/results/multisource_baselines_v1/summary/summary.json"))
    orig30 = load_json(Path("artifacts/results/external/2026-08-13-imbalanced-30k/original_baselines/summary.json"))["grouped"]
    ms30 = load_json(Path("artifacts/results/external/2026-08-13-imbalanced-30k/multisource_baselines/summary.json"))["grouped"]
    hardgan = load_json(Path("artifacts/results/ms_hardgan_v2/quality/quality.json"))
    return natural, weighted10, weighted30, ms, orig30, ms30, hardgan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    natural, weighted10, weighted30, ms, orig30, ms30, hardgan = load_evidence()
    natural_eval = natural["conditions"]["real_only"]["evaluations"]
    best10, best30 = weighted10["gan_10pct"]["external_10k"], weighted30["gan_10pct"]["external_10k"]
    report_data = {
        "created": "2026-08-13", "recommended_candidate": "Weighted HYB-MS +10% MS-HardGAN-C, calibrated three-seed ensemble",
        "external_10k": best10, "external_30k": best30,
        "natural_hybrid": natural["conditions"]["real_only"],
        "guardrail": "deployment candidate, not deployment-ready",
    }
    write_json(OUT / "report_data.json", report_data)

    pages = []
    # 1 — Executive summary
    fig = page("Complete Project Comparison", "Classifiers, hybrid systems, GAN generators, fairness controls, and deployment evidence")
    fig.text(.055, .81, "BOTTOM LINE", fontsize=11, color=TEAL, fontweight="bold")
    paragraph(fig, .055, .765,
              "The strongest complete system is the calibrated three-seed Weighted HYB-MS +10% MS-HardGAN-C ensemble. "
              "It wins on both frozen external evaluations, but weighted exposure—not the hybrid architecture alone—is a material part of that success.",
              width=95, size=15, weight="bold")
    table(fig, [.055, .48, .89, .19],
          ["Evaluation", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FPR", "FNR"],
          [["Frozen external 10K", pct(best10["accuracy"]), pct(best10["precision"]), pct(best10["recall"]), pct(best10["f1"]), pct(best10["roc_auc"]), pct(best10["false_positive_rate"]), pct(best10["false_negative_rate"])],
           ["Imbalanced external 30K", pct(best30["accuracy"]), pct(best30["precision"]), pct(best30["recall"]), pct(best30["f1"]), pct(best30["roc_auc"]), pct(best30["false_positive_rate"]), pct(best30["false_negative_rate"])]],
          widths=[.22,.11,.11,.11,.11,.12,.10,.10], font=9.2, row_colors=["#E8F5F2", "#E8F5F2"])
    bullet_list(fig, .065, .41, [
        "Natural-sampling HYB-MS is the best operationally usable naturally sampled model, but its conservative threshold misses about two thirds of external phishing URLs.",
        "Ordinary CNN/LSTM/RNN baselines achieve high recall by marking roughly 93–99% of external legitimate URLs as phishing; that is unusable for a browser extension.",
        "Low-dose weighted GAN augmentation (+10%) improves the calibrated weighted hybrid; 20–40% progressively sacrifices recall and F1.",
        "The current evidence supports a deployment candidate, not an unconditional deployment-ready claim."
    ], gap=.067)
    pages.append(fig)

    # 2 — Dataset and protocol
    fig = page("Data and Experimental Protocol", "What was frozen, what changed, and what stayed fixed")
    table(fig, [.055, .47, .89, .35], ["Artifact", "Rows", "Role", "Important detail"], [
        ["ORIG real train", "358,619", "Original baselines", "URLs_New; 276,400 legitimate + 82,219 phishing"],
        ["MS stored real pool", "363,692", "MS baselines and HYB-MS", "ORIG + 5,073 unique outside rows"],
        ["Outside rows in MS", "5,073", "Coverage", "3,012 legit + 2,000 PhishTank + 61 OpenPhish"],
        ["Internal test", "49,229", "Frozen evaluation", "Hostname-disjoint; never GAN training"],
        ["External 10K", "10,000", "Source-transfer test", "5,000 legitimate + 5,000 phishing"],
        ["External 30K", "30,000", "Later imbalanced test", "20,979 legitimate + 9,021 phishing"],
        ["Calibration", "400", "Threshold only", "Separate 50/50 set; max F1 subject to FPR ≤1%"],
        ["GAN challenge holdout", "61,728", "Synthetic robustness", "Never used in augmentation training"],
    ], widths=[.19,.10,.19,.46], font=8.5)
    fig.text(.055, .41, "THE 65/35 DISTINCTION", fontsize=11, color=RED, fontweight="bold")
    paragraph(fig, .055, .37,
              "The MS file is not 35% unique external data. Physically it is 98.61% original and 1.39% external. "
              "Original HYB-MS repeatedly samples the 5,073 outside URLs until outside exposure is 35% of an epoch. "
              "The natural control visits every stored row once per epoch and therefore exposes only 1.39% external rows.", width=103, size=11.5)
    bullet_list(fig, .065, .23, [
        "All final neural training used CUDA mixed precision on an RTX 3060 Laptop GPU.",
        "Seeds 42/43/44 and frozen validation/test identities were retained.",
        "External labels were not used to choose checkpoints or thresholds."
    ], gap=.052)
    pages.append(fig)

    # 3 — System taxonomy and pros/cons
    fig = page("Model Families: What Each One Contributes", "Architectural strengths do not guarantee threshold transfer")
    rows = [
        ["RF", "27 or 35 lexical features", "Fast inference; interpretable feature importance", "Weak external ranking; augmentation degrades"],
        ["Linear SVM", "Scaled lexical features", "Simple; compact; strong linear baseline", "Poor external ranking/threshold transfer"],
        ["Char-CNN", "First 128 URL characters", "Fast; learns local suspicious patterns", "External legitimate FPR ≈98–99%"],
        ["LSTM", "Character sequence", "Strongest ordinary external ranker", "External FPR ≈93%; slow recurrent path"],
        ["Vanilla RNN", "Character sequence", "Simple recurrent control", "Unstable; weakest recurrent robustness"],
        ["CNN+PSL", "Characters + 35 PSL lexical", "Tests feature fusion directly", "Features alone do not solve source shift"],
        ["HYB-MS natural", "CNN + PSL + CORAL", "Very low external FPR; strong ranking", "Conservative threshold; high FNR"],
        ["HYB-MS weighted", "Hybrid + 65/35 exposure", "Best source transfer after calibration", "Repeats tiny outside pools heavily"],
        ["Weighted +10% GAN", "Complete system + synthetic phish", "Best balanced 10K/30K evidence", "More complex; depends on synthetic quality"],
    ]
    table(fig, [.04, .19, .92, .64], ["System", "Input/training", "Main advantage", "Main limitation"], rows,
          widths=[.15,.23,.28,.30], font=8.1, row_colors=[None]*6+["#EEF5FA","#E8F5F2","#FFF4E6"])
    paragraph(fig, .055, .12,
              "Interpretation rule: a high recall score is not useful if nearly every legitimate URL is also blocked. For browser deployment, FPR, calibration stability, latency, and drift matter alongside F1 and ROC-AUC.", width=110, size=10.5, weight="bold")
    pages.append(fig)

    # 4 — Research progression
    fig = page("What the Experiments Established", "A progression from internal performance to distribution-shift controls")
    stages = [
        ("1", "ORIG baselines", "Excellent internal metrics; catastrophic external legitimate FPR."),
        ("2", "Coverage + structure balancing", "Real outside coverage helped, but small additions did not solve shift."),
        ("3", "HYB-ORIG", "Hybrid architecture alone failed externally; features were not the whole answer."),
        ("4", "Ordinary MS baselines", "Same 363,692 rows improved ranking, but thresholds still failed."),
        ("5", "Weighted HYB-MS", "Source-balanced exposure + PSL fusion + CORAL + calibration transferred well."),
        ("6", "SeqGAN/RelGAN/URL-GAN", "Generator realism and uniqueness were measured independently of classifier gains."),
        ("7", "MS-HardGAN-C", "Controlled, multi-source hard-example generator; held-out challenge set frozen."),
        ("8", "Weighted GAN doses", "+10% improved the complete calibrated hybrid; higher doses harmed recall/F1."),
        ("9", "Natural HYB fairness control", "Removed 65/35 resampling; real-only beat natural GAN externally."),
        ("10", "Later imbalanced 30K", "Confirmed weighted +10% as the strongest balanced candidate."),
    ]
    y=.82
    for number, title, finding in stages:
        fig.text(.065, y, number, fontsize=12, fontweight="bold", color="white", ha="center", va="center",
                 bbox={"boxstyle":"circle,pad=.36", "facecolor":BLUE, "edgecolor":"none"})
        fig.text(.105, y+.012, title, fontsize=11, fontweight="bold", color=DARK, va="center")
        fig.text(.29, y+.012, finding, fontsize=9.7, color=MUTED, va="center")
        if number != "10": fig.add_artist(plt.Line2D([.065,.065],[y-.055,y-.025],color="#B8C5D1",lw=2))
        y -= .072
    pages.append(fig)

    # 5 — Natural baseline comparison
    fig = page("Controlled Natural-Sampling Comparison", "Same 363,692 stored real URLs; no 65/35 resampling")
    models = ["cnn", "lstm", "rnn", "random_forest", "linear_svm"]
    names = ["CNN", "LSTM", "RNN", "RF", "SVM"]
    ms10 = ms["external_frozen_10k"]["multisource"]
    natural10 = natural_eval["external_10k"]
    natural30 = natural_eval["external_30k"]
    rows10=[]
    for key,name in zip(models,names):
        x=mean_metrics(ms10[key]); rows10.append([name,pct(x["accuracy"]),pct(x["precision"]),pct(x["recall"]),pct(x["f1"]),pct(x["roc_auc"]),pct(x["false_positive_rate"])])
    rows10.append(["Natural HYB",pct(natural10["accuracy"]["mean"]),pct(natural10["precision"]["mean"]),pct(natural10["recall"]["mean"]),pct(natural10["f1"]["mean"]),pct(natural10["roc_auc"]["mean"]),pct(natural10["false_positive_rate"]["mean"])])
    table(fig,[.055,.51,.89,.29],["10K system","Accuracy","Precision","Recall","F1","ROC-AUC","FPR"],rows10,
          widths=[.18,.13,.13,.13,.13,.14,.12],font=8.5,row_colors=[None]*5+["#E8F5F2"])
    rows30=[]
    for key,name in zip(models,names):
        x=mean_metrics(ms30[key]["metrics"]); rows30.append([name,pct(x["accuracy"]),pct(x["precision"]),pct(x["recall"]),pct(x["f1"]),pct(x["roc_auc"]),pct(x["false_positive_rate"])])
    rows30.append(["Natural HYB",pct(natural30["accuracy"]["mean"]),pct(natural30["precision"]["mean"]),pct(natural30["recall"]["mean"]),pct(natural30["f1"]["mean"]),pct(natural30["roc_auc"]["mean"]),pct(natural30["false_positive_rate"]["mean"])])
    table(fig,[.055,.18,.89,.27],["30K system","Accuracy","Precision","Recall","F1","ROC-AUC","FPR"],rows30,
          widths=[.18,.13,.13,.13,.13,.14,.12],font=8.5,row_colors=[None]*5+["#E8F5F2"])
    paragraph(fig,.055,.12,"Verdict: Natural HYB is operationally superior because its FPR is below 0.4%, whereas sequence baselines obtain high recall by flagging about 93–99% of legitimate URLs. It does not win every raw metric or the internal F1 comparison.",width=111,size=10.2,weight="bold")
    pages.append(fig)

    # 6 — Weighted dose response
    fig = page("Weighted HYB-MS: Calibrated GAN Dose Response", "Three-seed ensembles; identical calibration policy; external labels evaluation-only")
    keys=["real_only","gan_10pct","gan_20pct","gan_30pct","gan_40pct"]
    labels=["Real","+10%","+20%","+30%","+40%"]
    axes=[fig.add_axes([.065,.51,.4,.28]),fig.add_axes([.535,.51,.4,.28])]
    for ax,data,title in zip(axes,[weighted10,weighted30],["Frozen external 10K","Imbalanced external 30K"]):
        f1=[100*data[k]["external_10k"]["f1"] for k in keys]
        recall=[100*data[k]["external_10k"]["recall"] for k in keys]
        auc=[100*data[k]["external_10k"]["roc_auc"] for k in keys]
        x=np.arange(5); ax.plot(x,f1,"o-",label="F1",lw=2.4,color=BLUE); ax.plot(x,recall,"o-",label="Recall",lw=2.2,color=RED); ax.plot(x,auc,"o-",label="ROC-AUC",lw=2.2,color=TEAL)
        ax.set_xticks(x,labels); ax.set_ylim(65,100); ax.grid(alpha=.22); ax.set_title(title,fontweight="bold"); ax.legend(frameon=False,ncol=3,fontsize=8)
    rows=[]
    for k,label in zip(keys,labels):
        a,b=weighted10[k]["external_10k"],weighted30[k]["external_10k"]
        rows.append([label,pct(a["f1"]),pct(a["recall"]),pct(a["false_positive_rate"]),pct(b["f1"]),pct(b["recall"]),pct(b["false_positive_rate"])])
    table(fig,[.075,.17,.85,.24],["Condition","10K F1","10K recall","10K FPR","30K F1","30K recall","30K FPR"],rows,
          widths=[.16,.14,.14,.14,.14,.14,.14],font=8.6,row_colors=[None,"#E8F5F2",None,None,None])
    paragraph(fig,.065,.115,"Winner: +10%. It improves weighted real-only on all eight registered metrics on both external sets. At 20–40%, lower FPR is bought by increasingly large recall/F1 losses.",width=108,size=10.5,weight="bold")
    pages.append(fig)

    # 7 — Natural GAN dose response
    fig = page("Natural HYB-MS: GAN Augmentation Fairness Control", "Same real rows, ordinary shuffling, and frozen nested synthetic pools")
    evaluations=[("internal","Internal"),("external_10k","External 10K"),("external_30k","External 30K")]
    axes=[]
    for i,(ev,title) in enumerate(evaluations):
        ax=fig.add_axes([.06+i*.31,.51,.27,.28]); axes.append(ax)
        vals=[]
        for k in keys:
            vals.append(100*natural["conditions"][k]["evaluations"][ev]["f1"]["mean"])
        ax.bar(np.arange(5),vals,color=[BLUE,TEAL,GOLD,RED,PURPLE]); ax.set_xticks(np.arange(5),labels,rotation=25,ha="right"); ax.set_ylim(0,100); ax.grid(axis="y",alpha=.2); ax.set_title(title+" F1",fontweight="bold")
    rows=[]
    for k,label in zip(keys,labels):
        c=natural["conditions"][k]["evaluations"]
        rows.append([label,pct(c["internal"]["f1"]["mean"]),pct(c["external_10k"]["f1"]["mean"]),pct(c["external_10k"]["roc_auc"]["mean"]),pct(c["external_30k"]["f1"]["mean"]),pct(c["external_30k"]["roc_auc"]["mean"])])
    table(fig,[.075,.18,.85,.23],["Condition","Internal F1","10K F1","10K AUC","30K F1","30K AUC"],rows,
          widths=[.17,.16,.16,.16,.16,.16],font=8.8,row_colors=["#E8F5F2",None,None,None,None])
    paragraph(fig,.065,.12,"Result: +10% has only a tiny internal F1 increase, while real-only is best on both external sets. External ranking declines as synthetic dose increases. The GAN pool cannot replace genuine outside-source coverage.",width=110,size=10.5,weight="bold")
    pages.append(fig)

    # 8 — GAN quality
    fig = page("GAN Generator Comparison", "Quality, realism, controllability, and downstream meaning")
    rows=[
        ["SeqGAN (SG)","82,219 ORIG phish","69.94%","99.84%","0.7357","94%","10.91%","Best original generator"],
        ["URL-GAN","82,219 ORIG phish","47.87%","99.81%","0.8817","90%","5.05%","Stable, but behind SG"],
        ["RelGAN","82,219 ORIG phish","2.57%","99.39%","0.9888","86%","0.09%","Failed realism/volume gate"],
        ["MS-HardGAN-C","84,280 MS phish","49.08%","accepted unique","0.7992","90%","9.95%","Best controlled MS generator"],
    ]
    table(fig,[.045,.49,.91,.30],["Generator","Training","Strict accept.","Uniqueness","Source AUC↓","Cluster cov.","Majority hard","Decision"],rows,
          widths=[.16,.17,.12,.13,.11,.11,.12,.18],font=7.8,row_colors=["#E8F5F2",None,"#FDECEC","#FFF4E6"])
    fig.text(.055,.42,"MS-HardGAN-C DETAILS",fontsize=11,color=TEAL,fontweight="bold")
    bullet_list(fig,.065,.38,[
        f"309,208 accepted unique URLs; 247,480 augmentation-eligible and 61,728 permanently held out as a synthetic challenge set.",
        f"Requested structural-pattern agreement: {100*hardgan['generation_audit']['category_agreement']['requested_pattern_agreement']:.2f}%; requested source agreement: {100*hardgan['generation_audit']['category_agreement']['requested_source_agreement']:.2f}%.",
        f"52.83% challenge at least one frozen classifier, but only 9.95% challenge the majority; these are stress examples, not proof of future zero-day prediction.",
        "Zero accepted overlap with real training data was enforced by strict filtering."
    ],gap=.061)
    fig.text(.055,.135,"Interpretation",fontsize=11,color=RED,fontweight="bold")
    paragraph(fig,.055,.10,"Uniqueness is necessary but insufficient. RelGAN is highly unique yet unrealistic. A useful generator must also pass validity, TLD, overlap, template, lexical-inlier, cluster-coverage, and downstream external-transfer checks.",width=112,size=10.2)
    pages.append(fig)

    # 9 — Pros/cons of experiment classes
    fig = page("Pros and Cons of Every Experimental Direction", "What each result can and cannot establish")
    rows=[
        ["ORIG baselines","Clean architecture references; simple reproduction","Severe source shift; unsafe external FPR","Baseline only"],
        ["Structure-balanced ORIG","Tests URL-shape imbalance without outside data","Did not remove lexical source separability","Negative control"],
        ["Coverage-plus","Shows real outside coverage helps","Added pool too small; thresholds still unsafe","Useful diagnosis"],
        ["Ordinary MS baselines","Same stored rows as hybrid; strong fairness control","No source-balanced exposure/calibration","Architecture/data control"],
        ["HYB-ORIG","Isolates hybrid architecture","Fails externally; no source adaptation","Critical ablation"],
        ["Natural HYB-MS","Same stored row frequency as baselines; very low FPR","High FNR with robust threshold","Best natural candidate"],
        ["Weighted HYB-MS","Strong external transfer and calibration","Heavily repeats only 5,073 outside URLs","Best real-only system"],
        ["SeqGAN study","Large realistic pool; honest negative result","No legitimate-side shift correction","Generator baseline"],
        ["MS-HardGAN-C","Conditional patterns, hard-example reward, held-out challenge","Synthetic source remains distinguishable","Preferred GAN"],
        ["Weighted +10%","Best 10K and 30K complete-system metrics","More components and confounds","Deployment candidate"],
        ["20–40% GAN","Tests dose response and robustness","Progressive external recall/F1 loss","Do not deploy"],
    ]
    table(fig,[.035,.13,.93,.70],["Direction","Pros","Cons","Proper interpretation"],rows,widths=[.17,.31,.31,.19],font=7.7,
          row_colors=[None,None,None,None,None,"#EEF5FA","#E8F5F2",None,"#FFF4E6","#E8F5F2","#FDECEC"])
    pages.append(fig)

    # 10 — Fairness and validity
    fig = page("Fairness, Confounds, and Validity", "How to describe the findings accurately to a professor")
    fig.text(.055,.81,"VALID CLAIMS",fontsize=12,color=TEAL,fontweight="bold")
    bullet_list(fig,.065,.765,[
        "All MS systems can access the same 363,692 unique stored real URLs.",
        "Natural HYB versus ordinary MS baselines controls stored rows and sampling frequency, but model features and adaptation objectives remain architecture-specific.",
        "Weighted HYB +10% is the best complete system tested on the two large frozen external sets.",
        "Low-dose synthetic phishing can help when combined with weighted multi-source exposure and calibration; it did not help the natural hybrid externally.",
        "High synthetic fractions generally harm real-world transfer."
    ],gap=.06)
    fig.text(.055,.42,"INVALID OR OVERSTATED CLAIMS",fontsize=12,color=RED,fontweight="bold")
    bullet_list(fig,.065,.375,[
        "The HYB-MS file contains 35% unique external URLs. It does not; the unique external share is 1.39%.",
        "The hybrid architecture alone beats every baseline. HYB-ORIG disproves this.",
        "GAN uniqueness proves augmentation value or zero-day prediction. It does not.",
        "The +10% result proves every classifier benefits. The advantage is specific to the complete weighted hybrid condition.",
        "Strong offline metrics alone make the extension deployment-ready. Operational tests are still required."
    ],gap=.06)
    pages.append(fig)

    # 11 — Deployment decision
    fig = page("Deployment Recommendation", "Best candidate, fallback, and required gates")
    fig.text(.055,.81,"RECOMMENDED CANDIDATE",fontsize=11,color=TEAL,fontweight="bold")
    paragraph(fig,.055,.765,"Calibrated three-seed Weighted HYB-MS +10% MS-HardGAN-C ensemble",width=90,size=17,weight="bold")
    table(fig,[.055,.50,.89,.17],["Candidate","External evidence","Why use it","Why be cautious"],[
        ["Weighted HYB +10%","10K F1 92.70%; 30K F1 91.98%","Best accuracy/F1/AUC; low FPR","Ensemble cost; weighted repeats; calibration set small"],
        ["Weighted HYB real","10K F1 91.20%; 30K F1 90.53%","Simpler; no synthetic dependency","Slightly worse on all eight external metrics"],
        ["Natural HYB real","10K F1 48.20%; 30K F1 52.13%","Very low FPR; fair sampling","Misses roughly two thirds of phish"],
    ],widths=[.18,.25,.28,.28],font=8.2,row_colors=["#E8F5F2",None,None])
    fig.text(.055,.43,"REQUIRED BEFORE “DEPLOYMENT-READY”",fontsize=11,color=RED,fontweight="bold")
    bullet_list(fig,.065,.385,[
        "Benchmark browser-side latency, memory, extension package size, and three-model ensemble throughput.",
        "Freeze a genuinely later-created phishing cohort and a later legitimate snapshot; evaluate once without recalibration.",
        "Measure performance at realistic browser prevalence and report alerts per 10,000 visited URLs.",
        "Add safe fallback behavior, allowlisting, score explanations, privacy controls, and drift monitoring.",
        "Define retraining and threshold-recalibration triggers before field use."
    ],gap=.058)
    fig.text(.055,.105,"Final wording",fontsize=11,color=BLUE,fontweight="bold")
    paragraph(fig,.055,.072,"“Weighted HYB-MS +10% is the strongest deployment candidate in the controlled offline study; it is not yet claimed to be production-ready.”",width=112,size=11.3,weight="bold")
    pages.append(fig)

    # 12 — Full weighted metrics appendix
    fig = page("Appendix A: Complete Weighted-Hybrid Metrics", "Calibrated ensemble results on both large frozen external sets")
    for idx,(data,title,y) in enumerate([(weighted10,"Frozen external 10K",.53),(weighted30,"Imbalanced external 30K",.18)]):
        rows=[]
        for key,label in zip(keys,labels):
            x=data[key]["external_10k"]
            rows.append([label]+[pct(x[m]) for m in METRICS])
        fig.text(.055,y+.285,title,fontsize=11,fontweight="bold",color=BLUE)
        table(fig,[.055,y,.89,.25],["Condition"]+[SHORT[m] for m in METRICS],rows,
              widths=[.13]+[.108]*8,font=7.9,row_colors=[None,"#E8F5F2",None,None,None])
    pages.append(fig)

    # 13 — Full natural metrics appendix
    fig = page("Appendix B: Complete Natural-Hybrid Metrics", "Mean across seeds 42/43/44; robust validation/domain-validation thresholds")
    for title,ev,y in [("Internal test","internal",.61),("Frozen external 10K","external_10k",.34),("Imbalanced external 30K","external_30k",.07)]:
        rows=[]
        for key,label in zip(keys,labels):
            x=natural["conditions"][key]["evaluations"][ev]
            rows.append([label]+[pct(x[m]["mean"]) for m in METRICS])
        fig.text(.055,y+.22,title,fontsize=10.5,fontweight="bold",color=BLUE)
        table(fig,[.055,y,.89,.19],["Condition"]+[SHORT[m] for m in METRICS],rows,
              widths=[.13]+[.108]*8,font=7.6,row_colors=["#E8F5F2",None,None,None,None])
    pages.append(fig)

    pdf_path=OUT/"complete_project_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for number,fig in enumerate(pages,1):
            footer(fig,number); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    md = "# Complete Project Comparison\n\n"
    md += "**Recommended deployment candidate:** calibrated three-seed Weighted HYB-MS +10% MS-HardGAN-C ensemble.\n\n"
    md += "This report separates controlled natural-sampling comparisons from complete weighted-system comparisons. The MS pool contains 363,692 unique rows, but only 5,073 (1.39%) are external; 35% describes weighted epoch exposure, not unique composition.\n\n"
    md += "## External winner\n\n| Set | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for title,x in [("10K",best10),("30K",best30)]:
        md += "| "+title+" | "+" | ".join(f"{x[m]:.4f}" for m in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate"))+" |\n"
    md += "\n## Main conclusion\n\nWeighted +10% is the strongest complete system. Natural real-only is the strongest natural hybrid externally. Higher GAN doses harm external transfer. The project supports a deployment candidate, not a deployment-ready claim.\n"
    (OUT/"README.md").write_text(md,encoding="utf-8")
    print(json.dumps({"pdf":str(pdf_path),"pages":len(pages),"data":str(OUT/"report_data.json")},indent=2))


if __name__ == "__main__":
    main()
