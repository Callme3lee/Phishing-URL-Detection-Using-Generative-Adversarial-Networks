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

from .common import load_json, write_json


CONDITIONS = ("real_only", "gan_10pct", "gan_20pct", "gan_30pct", "gan_40pct")
CDISPLAY = {"real_only": "Real", "gan_10pct": "+10%", "gan_20pct": "+20%", "gan_30pct": "+30%", "gan_40pct": "+40%"}
MODELS = ("random_forest", "linear_svm", "cnn_psl", "hyb_ms")
MDISPLAY = {"random_forest": "RF-PSL", "linear_svm": "SVM-PSL", "cnn_psl": "CNN+PSL", "hyb_ms": "HYB-MS"}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")
SHORT = {"accuracy": "ACC", "precision": "PREC", "recall": "REC", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "false_positive_rate": "FPR", "false_negative_rate": "FNR"}
ROOT = Path("artifacts/results/fair_hybrid_gan_v1")
CHECKPOINT = Path("artifacts/experiments/fair_hybrid_gan_v1")
REAL_HYB = Path("artifacts/experiments/hybrid_cnn_coral_v1")


def stats(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": values}


def metric_payload(path: Path, model: str, policy: str = "native") -> dict:
    raw = load_json(path)
    if model == "hyb_ms" and policy == "fair_original_validation":
        return raw["fair_original_threshold_metrics"]
    if model == "hyb_ms" and policy == "native":
        return raw["robust_threshold_metrics"]
    return raw


def internal_grouped(policy: str) -> dict:
    grouped = {}
    for condition in CONDITIONS:
        grouped[condition] = {}
        for model in MODELS:
            values = []
            for seed in (42, 43, 44):
                if condition == "real_only" and model == "hyb_ms":
                    raw = load_json(REAL_HYB / f"seed_{seed}" / "metrics.json")
                    if policy == "fair_original_validation":
                        path = REAL_HYB / f"seed_{seed}" / "test_predictions.npz"
                        pred = np.load(path)
                        threshold = float(raw["original_validation_threshold"]["threshold"])
                        from .metrics import classification_metrics
                        item = classification_metrics(pred["labels"], pred["scores"], threshold)
                    else: item = raw
                else:
                    item = metric_payload(CHECKPOINT / condition / model / f"seed_{seed}" / "metrics.json", model, policy)
                values.append(item)
            grouped[condition][model] = {metric: stats([float(row[metric]) for row in values]) for metric in METRICS}
    return grouped


def old_real_only() -> dict:
    roots = {"CNN-Char": "cnn", "LSTM-Char": "lstm", "RNN-Char": "rnn", "RF-27": "random_forest", "SVM-27": "linear_svm"}
    output = {}
    external = load_json(Path("artifacts/results/multisource_baselines_v1/external_large_2026-08-12/summary.json"))
    for display, model in roots.items():
        rows = [load_json(Path("artifacts/experiments/multisource_baselines_v1") / model / f"seed_{seed}" / "metrics.json") for seed in (42, 43, 44)]
        output[display] = {
            "internal": {metric: stats([float(row[metric]) for row in rows]) for metric in METRICS},
            "external": external["grouped"][model]["metrics"],
        }
    return output


def fmt(value: dict) -> str:
    return f"{value['mean']:.4f}±{value['standard_deviation']:.4f}"


def title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("#F7F9FC")
    fig.text(.06, .84, "Fair Classifier and HYB-MS\nGAN Augmentation Study", fontsize=28, fontweight="bold", color="#17324D")
    fig.text(.06, .68, "Same 363,692 real URLs • Same 35 PSL features where applicable\nSeeds 42/43/44 • Frozen external 10K • MS-HardGAN-C", fontsize=15, color="#355C7D")
    paragraphs = [
        "Purpose: remove feature-access favoritism, isolate feature fusion and adaptation, and test whether GAN augmentation improves each unchanged model.",
        "Main result: 35 PSL-aware features substantially strengthen RF/SVM internally but do not solve external source shift. The separately calibrated HYB-MS +10% ensemble is the strongest balanced augmented candidate on the frozen external 10K.",
        "Scientific guard: strong metrics mean deployment candidate—not deployment-ready. A later temporal test, realistic prevalence, browser latency/memory, drift monitoring, and operational false-alert testing remain required.",
    ]
    y = .50
    for paragraph in paragraphs:
        fig.text(.08, y, textwrap.fill(paragraph, 100), fontsize=13, va="top", color="#263238")
        y -= .14
    fig.text(.06, .05, "Generated reproducibly from frozen Version 3 artifacts", fontsize=10, color="#607D8B")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def protocol_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27)); ax.axis("off")
    ax.set_title("What is controlled—and what is intentionally different", fontsize=20, fontweight="bold", pad=20)
    rows = [
        ["CNN/LSTM/RNN-Char", "363,692", "128 chars", "Ordinary training", "Original validation"],
        ["RF-27 / SVM-27", "363,692", "27 lexical", "Class weighting", "Original validation"],
        ["RF-PSL / SVM-PSL", "363,692", "Same 35 PSL", "Class weighting", "Original validation"],
        ["CNN+PSL", "363,692", "Chars + same 35 PSL", "Ordinary training", "Original validation"],
        ["HYB-MS", "363,692", "Chars + same 35 PSL", "Source balance + CORAL", "Original + domain val."],
        ["HYB-MS calibrated", "363,692", "Same as HYB-MS", "3-seed ensemble", "Separate 400-row cal."],
    ]
    table = ax.table(cellText=rows, colLabels=["System", "Real rows", "Feature view", "Training distinction", "Threshold source"],
                     loc="center", cellLoc="left", colLoc="left", colWidths=[.20,.11,.21,.24,.22])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.2)
    for (r,c), cell in table.get_celld().items():
        if r == 0: cell.set_facecolor("#355C7D"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0: cell.set_facecolor("#EAF0F6")
    ax.text(.02, .10, textwrap.fill("Fairness rule: RF-PSL, SVM-PSL, CNN+PSL, and HYB-MS receive the exact same 35 robust-scaled PSL features. HYB-MS balancing/CORAL are reported as explicit method components, not hidden advantages. Synthetic rows are a declared group and are excluded from CORAL pairs.", 145), transform=ax.transAxes, fontsize=11)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def line_dashboard(pdf: PdfPages, grouped: dict, title: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5)); x = np.arange(len(CONDITIONS))
    colors = ("#E45756", "#72B7B2", "#4C78A8", "#F58518")
    for ax, metric in zip(axes.ravel(), METRICS):
        for model, color in zip(MODELS, colors):
            means = [100 * grouped[c][model][metric]["mean"] for c in CONDITIONS]
            errors = [100 * grouped[c][model][metric]["standard_deviation"] for c in CONDITIONS]
            ax.errorbar(x, means, yerr=errors, marker="o", linewidth=2, capsize=3, label=MDISPLAY[model], color=color)
        ax.set_title(SHORT[metric], fontweight="bold"); ax.set_xticks(x, [CDISPLAY[c] for c in CONDITIONS], rotation=18)
        ax.set_ylabel("Percent"); ax.grid(alpha=.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels(); axes.ravel()[-1].legend(handles, labels, loc="center", frameon=False)
    axes.ravel()[-1].axis("off"); fig.suptitle(title + "\nmean ± SD across seeds 42/43/44", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.93)); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def table_pages(pdf: PdfPages, title: str, grouped: dict) -> None:
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            rows.append([MDISPLAY[model], CDISPLAY[condition]] + [fmt(grouped[condition][model][m]) for m in METRICS])
    for start in range(0, len(rows), 10):
        fig, ax = plt.subplots(figsize=(16, 8.5)); ax.axis("off")
        ax.set_title(title + f" (rows {start+1}–{min(start+10,len(rows))})", fontsize=17, fontweight="bold", pad=15)
        table = ax.table(cellText=rows[start:start+10], colLabels=["Model", "Condition"] + [SHORT[m] for m in METRICS],
                         loc="center", cellLoc="center", colWidths=[.10,.08]+[.10]*8)
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 2.0)
        for (r,c), cell in table.get_celld().items():
            if r == 0: cell.set_facecolor("#355C7D"); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0: cell.set_facecolor("#EEF3F8")
        ax.text(.01,.04,"All values are mean ± sample SD across three seeds. FPR/FNR are threshold-dependent; ROC-AUC/PR-AUC evaluate ranking.",transform=ax.transAxes,fontsize=10)
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def real_ablation_page(pdf: PdfPages, old: dict, internal: dict, external: dict, calibrated: dict) -> None:
    systems = list(old) + ["RF-PSL", "SVM-PSL", "CNN+PSL", "HYB-MS native", "HYB-MS calibrated"]
    internal_f1 = [old[x]["internal"]["f1"]["mean"] for x in old] + [internal["real_only"][m]["f1"]["mean"] for m in MODELS] + [calibrated["results"]["real_only"]["internal_test"]["f1"]]
    external_f1 = [old[x]["external"]["f1"]["mean"] for x in old] + [external["real_only"][m]["native"]["f1"]["mean"] for m in MODELS] + [calibrated["results"]["real_only"]["external_10k"]["f1"]]
    external_fpr = [old[x]["external"]["false_positive_rate"]["mean"] for x in old] + [external["real_only"][m]["native"]["false_positive_rate"]["mean"] for m in MODELS] + [calibrated["results"]["real_only"]["external_10k"]["false_positive_rate"]]
    x=np.arange(len(systems)); width=.26; fig,ax=plt.subplots(figsize=(16,8.5))
    ax.bar(x-width,100*np.asarray(internal_f1),width,label="Internal F1",color="#4C78A8")
    ax.bar(x,100*np.asarray(external_f1),width,label="External 10K F1",color="#54A24B")
    ax.bar(x+width,100*np.asarray(external_fpr),width,label="External 10K FPR",color="#E45756")
    ax.set_xticks(x,systems,rotation=25,ha="right"); ax.set_ylabel("Percent"); ax.grid(axis="y",alpha=.25); ax.legend(ncol=3)
    ax.set_title("Real-only ablation: feature quality helps internally; adaptation + calibration control external FPR",fontweight="bold",fontsize=15)
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def calibrated_page(pdf: PdfPages, calibrated: dict) -> None:
    results=calibrated["results"]; x=np.arange(len(CONDITIONS)); fig,axes=plt.subplots(2,3,figsize=(15,8.5))
    selected=("accuracy","precision","recall","f1","roc_auc","false_positive_rate")
    for ax,metric in zip(axes.ravel(),selected):
        values=[100*results[c]["external_10k"][metric] for c in CONDITIONS]
        bars=ax.bar(x,values,color=["#607D8B","#2A9D8F","#4C78A8","#F58518","#E45756"])
        ax.set_xticks(x,[CDISPLAY[c] for c in CONDITIONS]); ax.set_title(SHORT[metric],fontweight="bold"); ax.grid(axis="y",alpha=.25)
        ax.bar_label(bars,fmt="%.1f",fontsize=8)
    fig.suptitle("Separately calibrated three-seed HYB-MS ensembles — frozen external 10K",fontsize=16,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.94)); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    rows=[]
    for c in CONDITIONS:
        r=results[c]
        rows.append([CDISPLAY[c],f"{r['threshold']:.6f}"]+[f"{r['external_10k'][m]:.4f}" for m in METRICS]+[f"{r['synthetic_challenge']['recall']:.4f}"])
    fig,ax=plt.subplots(figsize=(16,8.5)); ax.axis("off"); ax.set_title("Calibrated ensemble: complete frozen-test metrics",fontsize=18,fontweight="bold")
    table=ax.table(cellText=rows,colLabels=["Condition","Threshold"]+[SHORT[m] for m in METRICS]+["GAN REC"],loc="center",cellLoc="center")
    table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,2.2)
    for (rr,cc),cell in table.get_celld().items():
        if rr==0:cell.set_facecolor("#355C7D");cell.set_text_props(color="white",weight="bold")
        elif rr==2:cell.set_facecolor("#DDF2E9")
    ax.text(.02,.12,textwrap.fill("Best balanced condition: HYB-MS +10%. It improves external F1, accuracy, precision, recall, ROC-AUC and challenge recall relative to calibrated real-only while reducing FPR. Ratios 20–40% progressively sacrifice recall/F1 despite lower FPR.",145),transform=ax.transAxes,fontsize=11)
    pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def deployment_page(pdf: PdfPages, calibrated: dict) -> None:
    best=calibrated["results"]["gan_10pct"]["external_10k"]
    fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis("off");ax.set_title("Deployment-readiness decision",fontsize=22,fontweight="bold",pad=20)
    checks=[
        ("Frozen balanced external 10K", "PASS as candidate", f"F1 {best['f1']:.3f}, ROC-AUC {best['roc_auc']:.3f}, FPR {best['false_positive_rate']:.2%}"),
        ("Independent threshold calibration", "PASS", "Separate frozen 400-row calibration set"),
        ("Later untouched temporal confirmation", "NOT PASSED", "Required before deployment claim"),
        ("Real browser prevalence / alert burden", "NOT TESTED", "Balanced 50/50 test is not browsing prevalence"),
        ("Extension latency, memory, packaging", "NOT TESTED", "Must benchmark target browser/device"),
        ("Drift, obfuscation, monitoring", "NOT COMPLETE", "Needs recurring updates and adversarial tests"),
        ("Context beyond URL text", "LIMITATION", "DNS/TLS/domain age/page signals recommended"),
    ]
    table=ax.table(cellText=checks,colLabels=["Gate","Status","Evidence / remaining work"],loc="center",cellLoc="left",colWidths=[.30,.20,.48])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.1)
    for (r,c),cell in table.get_celld().items():
        if r==0:cell.set_facecolor("#355C7D");cell.set_text_props(color="white",weight="bold")
        elif r in (1,2):cell.set_facecolor("#DDF2E9")
        else:cell.set_facecolor("#FCE8E6")
    ax.text(.03,.08,"Verdict: HYB-MS +10% is the strongest deployment candidate in this experiment. It is not yet deployment-ready.",transform=ax.transAxes,fontsize=14,fontweight="bold",color="#8B1E1E")
    pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def main() -> None:
    ROOT.mkdir(parents=True,exist_ok=True)
    internal_native=internal_grouped("native"); internal_fair=internal_grouped("fair_original_validation")
    evaluation=load_json(ROOT/"evaluation"/"external_10k.json"); external=evaluation["grouped"]
    challenge=load_json(ROOT/"evaluation"/"synthetic_challenge.json")
    calibrated=load_json(ROOT/"calibrated_ensembles.json"); old=old_real_only()
    summary={"project_version":3,"conditions":list(CONDITIONS),"models":list(MODELS),"metrics":list(METRICS),
             "internal_native":internal_native,"internal_fair_threshold":internal_fair,"external_10k":external,
             "synthetic_challenge":challenge["grouped"],"calibrated_hyb_ensembles":calibrated["results"],"legacy_real_only_context":old,
             "best_balanced_condition":"gan_10pct calibrated HYB-MS ensemble",
             "deployment_verdict":"deployment candidate; not deployment-ready"}
    write_json(ROOT/"summary.json",summary)
    with (ROOT/"all_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["scope","policy","condition","model","metric","mean","sd"];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for scope,grouped,policy in (("internal","native",None),): pass
        for scope,grouped in (("internal_native",internal_native),("internal_fair",internal_fair)):
            for c in CONDITIONS:
                for model in MODELS:
                    for metric in METRICS:
                        value=grouped[c][model][metric];writer.writerow({"scope":scope,"policy":scope.split("_",1)[-1],"condition":c,"model":model,"metric":metric,"mean":value["mean"],"sd":value["standard_deviation"]})
        for c in CONDITIONS:
            for model in MODELS:
                for policy in ("fair_original_validation","native"):
                    for metric in METRICS:
                        value=external[c][model][policy][metric];writer.writerow({"scope":"external_10k","policy":policy,"condition":c,"model":model,"metric":metric,"mean":value["mean"],"sd":value["standard_deviation"]})
    report_path=ROOT/"fair_hybrid_gan_complete_report.pdf"
    with PdfPages(report_path) as pdf:
        title_page(pdf);protocol_page(pdf);real_ablation_page(pdf,old,internal_native,external,calibrated)
        line_dashboard(pdf,internal_fair,"Feature-equal comparison — unchanged internal real test, common threshold policy")
        table_pages(pdf,"Internal real test — common original-validation threshold",internal_fair)
        external_native={c:{m:external[c][m]["native"] for m in MODELS} for c in CONDITIONS}
        line_dashboard(pdf,external_native,"Feature-equal comparison — frozen external 10K, native thresholds")
        table_pages(pdf,"Frozen external 10K — native threshold policy",external_native)
        calibrated_page(pdf,calibrated);deployment_page(pdf,calibrated)
        info=pdf.infodict();info["Title"]="Fair classifier and HYB-MS GAN augmentation study";info["Author"]="Version 3 reproducible experiment"
    lines=["# Fair Classifier and HYB-MS GAN Augmentation Study","",f"Complete PDF: `{report_path}`.","",
           "## Main result","","The separately calibrated HYB-MS +10% ensemble is the best balanced augmented condition on the frozen external 10K. It is a deployment candidate, not deployment-ready.","",
           "## Calibrated external 10K","","| Condition | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | GAN challenge recall |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        r=calibrated["results"][c];lines.append("| "+CDISPLAY[c]+" | "+" | ".join(f"{r['external_10k'][m]:.4f}" for m in METRICS)+f" | {r['synthetic_challenge']['recall']:.4f} |")
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"pdf":str(report_path),"summary":str(ROOT/"summary.json"),"csv":str(ROOT/"all_metrics.csv")},indent=2))


if __name__ == "__main__":
    main()
