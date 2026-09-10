from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .train_lexical_verifier import LexicalVerifier
from .url_data import encode_url, psl_aware_lexical_features


ROOT=Path("data/evaluation/two_stage_url_cascade_v1")
FREEZE=Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/frozen_cascade.json")
STAGE1_ROOT=Path("artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms")
OUTPUT=Path("artifacts/results/two_stage_url_cascade_v1")
VERIFIERS=Path("artifacts/experiments/two_stage_url_cascade_v1/lexical_verifier")


@torch.inference_mode()
def verifier_score(model, values, device):
    model.eval();out=[]
    for start in range(0,len(values),2048):
        batch=torch.from_numpy(values[start:start+2048]).to(device)
        with torch.autocast(device_type="cuda",dtype=torch.float16):logits=model(batch)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def pct(value):return f"{100*value:.2f}%"


def make_pdf(path,payload):
    primary=payload["operating_points"][0];single=primary["single_stage"];cascade=primary["cascade"]
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white")
        fig.text(.06,.92,"Two-Stage URL Cascade",fontsize=25,weight="bold",color="#17324D")
        fig.text(.06,.865,"One-shot identity-disjoint evaluation • 18,000 legitimate + 2,000 phishing URLs",fontsize=12,color="#526274")
        fig.text(.06,.77,"STRICT 0.25% CALIBRATION BUDGET",fontsize=17,weight="bold",color="#17324D")
        cols=["System","Accuracy","Precision","Recall","F1","ROC-AUC","FPR","FNR"]
        rows=[["Single-stage GAN",*[pct(single[k]) for k in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]],
              ["Two-stage cascade",*[pct(cascade[k]) for k in ("accuracy","precision","recall","f1","roc_auc","false_positive_rate","false_negative_rate")]]]
        ax=fig.add_axes([.05,.48,.90,.20]);ax.axis("off");table=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center")
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2.2)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1");cell.set_facecolor("#365F7F" if r==0 else ("#E7F3F1" if r==2 else "#EEF3F7"))
            if r==0:cell.set_text_props(color="white",weight="bold")
        delta=primary["delta_cascade_minus_single"]
        fig.text(.07,.36,f"Recall delta: {100*delta['recall']:+.2f} points    FNR delta: {100*delta['false_negative_rate']:+.2f} points",fontsize=15,weight="bold",color="#188977" if delta["recall"]>0 else "#C94C4C")
        fig.text(.07,.28,f"Observed counts — cascade: TP {cascade['confusion_matrix']['tp']:,}, FN {cascade['confusion_matrix']['fn']:,}, FP {cascade['confusion_matrix']['fp']:,}, TN {cascade['confusion_matrix']['tn']:,}",fontsize=11)
        fig.text(.07,.18,payload["verdict"],fontsize=11,color="#526274",wrap=True)
        fig.text(.07,.08,"This test was not used for training or threshold selection. It is not a later temporal snapshot and does not prove deployment readiness.",fontsize=10,weight="bold",color="#C26B32")
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11,8.5));fig.patch.set_facecolor("white")
        caps=[100*r["maximum_fpr"] for r in payload["operating_points"]];x=np.arange(len(caps));w=.34
        axes[0].bar(x-w/2,[100*r["single_stage"]["recall"] for r in payload["operating_points"]],w,label="Single-stage GAN",color="#4C78A8")
        axes[0].bar(x+w/2,[100*r["cascade"]["recall"] for r in payload["operating_points"]],w,label="Cascade",color="#1C9A8B")
        axes[0].set_xticks(x,[f"{v:.2f}%" for v in caps]);axes[0].set(xlabel="Calibration FPR budget",ylabel="Evaluation recall (%)",ylim=(0,100));axes[0].grid(axis="y",alpha=.2);axes[0].legend();axes[0].set_title("Phishing detection",weight="bold")
        axes[1].bar(x-w/2,[100*r["single_stage"]["false_positive_rate"] for r in payload["operating_points"]],w,label="Single-stage GAN",color="#4C78A8")
        axes[1].bar(x+w/2,[100*r["cascade"]["false_positive_rate"] for r in payload["operating_points"]],w,label="Cascade",color="#1C9A8B")
        axes[1].set_xticks(x,[f"{v:.2f}%" for v in caps]);axes[1].set(xlabel="Calibration FPR budget",ylabel="Observed evaluation FPR (%)");axes[1].grid(axis="y",alpha=.2);axes[1].legend();axes[1].set_title("Legitimate false alarms",weight="bold")
        fig.suptitle("Frozen operating points on the untouched cohort",fontsize=19,weight="bold",color="#17324D");fig.tight_layout(rect=[0,.06,1,.94]);pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11,8.5));fig.patch.set_facecolor("white");fig.text(.06,.92,"What This Experiment Means",fontsize=24,weight="bold",color="#17324D")
        lines=[("Architecture","Stage 1: updated HYB-MS +10% GAN screen. Stage 2: independent real-only lexical MLP verifier."),
               ("Strict result",payload["verdict"]),
               ("Routing",f"{pct(primary['routed_fraction'])} of URLs entered Stage 2 at the strict operating point."),
               ("AUC definition","Cascade ROC-AUC/PR-AUC use a continuous cascade decision score: immediate blocks rank first, routed rows use verifier scores, non-routed rows rank last."),
               ("Scope","Both stages remain URL-only. HTML, redirect chains, DNS age, TLS, reputation, and user context are absent."),
               ("Research conclusion","A second independent representation can reduce FNR under a very strict false-alarm budget, but its benefit must be judged per operating point and reconfirmed temporally.")]
        y=.82
        for title,text in lines:
            fig.text(.07,y,title,fontsize=13,weight="bold",color="#17324D");fig.text(.25,y,text,fontsize=10.5,color="#526274",wrap=True);y-=.115
        pdf.savefig(fig);plt.close(fig)


def main():
    if OUTPUT.exists():raise FileExistsError(f"One-shot results already exist: {OUTPUT}")
    report=load_json(ROOT/"collection_report.json");freeze=load_json(FREEZE);dataset=ROOT/"evaluation_urls.csv"
    if sha256_file(dataset)!=report["dataset_sha256"] or sha256_file(FREEZE)!=report["cascade_freeze_sha256"]:raise RuntimeError("Frozen artifact hash mismatch")
    rows=list(csv.DictReader(dataset.open("r",encoding="utf-8",newline="")));labels=np.asarray([int(r["label"]) for r in rows],dtype=np.uint8);urls=[r["url"] for r in rows]
    raw=np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32);device=require_cuda()
    stage1_seed=[];stage1_checkpoints=[]
    for seed in (42,43,44):
        path=STAGE1_ROOT/f"seed_{seed}/checkpoint.pt";saved=torch.load(path,map_location=device,weights_only=False)
        tokens=np.stack([encode_url(url,saved["vocabulary"],128) for url in urls]);scale=np.where(np.asarray(saved["lexical_scaler_scale"])==0,1,np.asarray(saved["lexical_scaler_scale"]))
        lexical=((raw-np.asarray(saved["lexical_scaler_center"]))/scale).astype(np.float32);model=create_model(saved).to(device);model.load_state_dict(saved["state_dict"])
        stage1_seed.append(neural_score(model,tokens,lexical,1024,device));stage1_checkpoints.append({"seed":seed,"path":str(path),"sha256":sha256_file(path)})
        del model,tokens,lexical;gc.collect();torch.cuda.empty_cache()
    stage1=np.mean(np.stack(stage1_seed),axis=0)
    scaler=np.load("data/experiments/recent_update_gan10_v1/real_only/processed/psl_lexical_scaler.npz");scale=np.where(scaler["scale"]==0,1,scaler["scale"]);features=((raw-scaler["center"])/scale).astype(np.float32)
    stage2_seed=[]
    for item in freeze["checkpoints"]:
        path=Path(item["path"])
        if sha256_file(path)!=item["sha256"]:raise RuntimeError(f"Verifier hash mismatch: {path}")
        saved=torch.load(path,map_location=device,weights_only=False);model=LexicalVerifier(saved["feature_count"]).to(device);model.load_state_dict(saved["state_dict"]);stage2_seed.append(verifier_score(model,features,device));del model;gc.collect();torch.cuda.empty_cache()
    stage2=np.mean(np.stack(stage2_seed),axis=0);screen=freeze["screening_rule"]["threshold"];hard=freeze["immediate_block_rule"]["threshold"];routed=stage1>=screen;immediate=stage1>=hard
    points=[]
    for cascade_cfg,single_cfg in zip(freeze["cascade_operating_points"],freeze["single_stage_operating_points"]):
        vt=cascade_cfg["verifier_threshold"];cascade_score=np.where(immediate,2.0,np.where(routed,stage2,-1.0));cascade=classification_metrics(labels,cascade_score,vt);single=classification_metrics(labels,stage1,single_cfg["threshold"])
        keys=("accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate","false_negative_rate")
        points.append({"maximum_fpr":cascade_cfg["maximum_fpr"],"single_stage_threshold":single_cfg["threshold"],"verifier_threshold":vt,
            "single_stage":single,"cascade":cascade,"delta_cascade_minus_single":{k:cascade[k]-single[k] for k in keys},
            "routed_rows":int(routed.sum()),"routed_fraction":float(routed.mean()),"immediate_block_rows":int(immediate.sum())})
    primary=points[0];d=primary["delta_cascade_minus_single"]
    verdict=(f"At the strict 0.25% calibration budget, the cascade changed recall by {100*d['recall']:+.2f} points and FNR by {100*d['false_negative_rate']:+.2f} points; observed FPR changed by {100*d['false_positive_rate']:+.3f} points. This is evidence for the cascade only at this locked operating point, not proof that it eliminates false negatives.")
    payload={"project_version":3,"experiment":"two_stage_url_cascade_v1","evaluation_status":"ONE-SHOT COMPLETE; NO POST-TEST TUNING",
        "dataset":report,"stage1":"Updated HYB-MS +10% GAN three-seed ensemble","stage2":"real-only lexical MLP three-seed ensemble",
        "operating_points":points,"verdict":verdict,"stage1_checkpoints":stage1_checkpoints,"verifier_checkpoints":freeze["checkpoints"],
        "limitations":report["limitations"],"auc_contract":"Single-stage AUC uses Stage-1 probabilities. Cascade AUC uses its continuous decision score (immediate=2, routed=verifier probability, non-routed=-1)."}
    OUTPUT.mkdir(parents=True);write_json(OUTPUT/"results.json",payload);np.savez_compressed(OUTPUT/"predictions.npz",labels=labels,stage1_scores=stage1,stage2_scores=stage2,routed=routed,immediate=immediate)
    make_pdf(OUTPUT/"two_stage_url_cascade_report.pdf",payload)
    lines=["# Two-stage URL cascade — one-shot evaluation","",verdict,"", "| Calibration FPR cap | System | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in points:
        for name,key in (("Single-stage GAN","single_stage"),("Two-stage cascade","cascade")):
            m=p[key];lines.append(f"| {pct(p['maximum_fpr'])} | {name} | "+" | ".join(pct(m[k]) for k in ("accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate","false_negative_rate"))+" |")
    lines.extend(["","This is identity-disjoint, not temporally newer. No post-test threshold tuning was performed."]);(OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":verdict,"operating_points":points},indent=2))


if __name__=="__main__":main()
