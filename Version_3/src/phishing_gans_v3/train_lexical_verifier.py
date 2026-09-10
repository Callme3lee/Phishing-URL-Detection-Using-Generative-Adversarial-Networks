from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .metrics import classification_metrics, select_threshold
from .train_hybrid_domain import TARGET_GROUP_PROPORTIONS


class FeatureDataset(Dataset):
    def __init__(self, root: Path, split: str, groups: bool=False):
        self.x=np.load(root/f"{split}_psl_lexical.npy",mmap_mode="r")
        self.y=np.load(root/f"{split}_labels.npy",mmap_mode="r")
        self.g=np.load(root/f"{split}_groups.npy",mmap_mode="r") if groups else None
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        values=(torch.from_numpy(np.array(self.x[i],dtype=np.float32,copy=True)),torch.tensor(np.float32(self.y[i])))
        return (*values,torch.tensor(np.int64(self.g[i]))) if self.g is not None else values


class LexicalVerifier(nn.Module):
    def __init__(self,n:int):
        super().__init__();self.network=nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.30),
            nn.Linear(128,64),nn.GELU(),nn.Dropout(.20),nn.Linear(64,1))
    def forward(self,x):return self.network(x).squeeze(1)


@torch.inference_mode()
def score(model,loader,device):
    model.eval();scores=[];labels=[]
    for x,y,*_ in loader:
        x=x.to(device,non_blocking=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16):logits=model(x)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy());labels.append(y.numpy())
    return np.concatenate(scores),np.concatenate(labels).astype(np.uint8)


def parse_args():
    p=argparse.ArgumentParser(description="Train the lexical-only Stage 2 verifier on CUDA.")
    p.add_argument("--data",type=Path,default=Path("data/experiments/recent_update_gan10_v1/real_only/processed"))
    p.add_argument("--data-report",type=Path,default=Path("data/experiments/recent_update_gan10_v1/real_only/experiment_report.json"))
    p.add_argument("--output",type=Path,default=Path("artifacts/experiments/two_stage_url_cascade_v1/lexical_verifier"))
    p.add_argument("--seeds",default="42,43,44");p.add_argument("--batch-size",type=int,default=2048);p.add_argument("--workers",type=int,default=2)
    return p.parse_args()


def train_one(seed,args,project):
    run=args.output/f"seed_{seed}";metrics_path=run/"metrics.json"
    if metrics_path.exists():return load_json(metrics_path)
    device=require_cuda();set_seed(seed);train=FeatureDataset(args.data,"train",True);groups=np.load(args.data/"train_groups.npy",mmap_mode="r")
    counts={g:max(1,int((groups==g).sum())) for g in TARGET_GROUP_PROPORTIONS}
    weights=np.asarray([TARGET_GROUP_PROPORTIONS[int(g)]/counts[int(g)] for g in groups],dtype=np.float64)
    sampler=WeightedRandomSampler(torch.from_numpy(weights),num_samples=len(train),replacement=True,generator=torch.Generator().manual_seed(seed))
    train_loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    loaders={split:DataLoader(FeatureDataset(args.data,split),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
             for split in ("validation","domain_validation","test")}
    feature_count=np.load(args.data/"train_psl_lexical.npy",mmap_mode="r").shape[1];model=LexicalVerifier(feature_count).to(device)
    cfg=project["neural_training"];optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    criterion=nn.BCEWithLogitsLoss();scaler=torch.amp.GradScaler("cuda");best=-np.inf;best_state=None;best_epoch=stale=0;history=[];total=0.
    for epoch in range(1,int(cfg["epochs"])+1):
        started=time.perf_counter();model.train();loss_total=seen=0;observed=Counter()
        for x,y,g in train_loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):loss=criterion(model(x),y)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),float(cfg["gradient_clip_norm"]));scaler.step(optimizer);scaler.update()
            loss_total+=float(loss.item())*len(y);seen+=len(y);observed.update(g.tolist())
        torch.cuda.synchronize();seconds=time.perf_counter()-started;total+=seconds
        vs,vy=score(model,loaders["validation"],device);ds,dy=score(model,loaders["domain_validation"],device)
        vp=float(average_precision_score(vy,vs));dp=float(average_precision_score(dy,ds));selection=.5*(vp+dp)
        record={"epoch":epoch,"loss":loss_total/seen,"validation_pr_auc":vp,"domain_validation_pr_auc":dp,
                "domain_validation_roc_auc":float(roc_auc_score(dy,ds)),"selection_score":selection,"training_seconds":seconds,
                "observed_group_proportions":{str(k):v/seen for k,v in observed.items()}}
        history.append(record);print(json.dumps({"seed":seed,**record}),flush=True)
        if selection>best:best=selection;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=int(cfg["early_stopping_patience"]):break
    model.load_state_dict(best_state);vs,vy=score(model,loaders["validation"],device);ds,dy=score(model,loaders["domain_validation"],device)
    vt=select_threshold(vy,vs,.01);dt=select_threshold(dy,ds,.01);threshold=max(vt["threshold"],dt["threshold"]);ts,ty=score(model,loaders["test"],device)
    metrics=classification_metrics(ty,ts,threshold);run.mkdir(parents=True,exist_ok=True)
    checkpoint={"project_version":3,"condition":"two_stage_url_cascade_v1","model":"lexical_mlp_verifier","seed":seed,
                "feature_count":feature_count,"state_dict":best_state,"threshold":threshold,"best_epoch":best_epoch,
                "data_report_sha256":sha256_file(args.data_report),"probability_contract":"FP32 sigmoid after FP16 forward pass"}
    torch.save(checkpoint,run/"checkpoint.pt");np.savez_compressed(run/"test_predictions.npz",labels=ty,scores=ts)
    payload={**metrics,"seed":seed,"model":"lexical_mlp_verifier","threshold":threshold,"best_epoch":best_epoch,
             "selection_score":best,"training_seconds":total,"device":torch.cuda.get_device_name(device),"training_rows":len(train),
             "target_group_proportions":TARGET_GROUP_PROPORTIONS,"data_report_sha256":sha256_file(args.data_report)}
    write_json(run/"history.json",history);write_json(metrics_path,payload);del model,train_loader,loaders,optimizer,scaler;gc.collect();torch.cuda.empty_cache();return payload


def main():
    args=parse_args();project=load_json(Path("configs/project.json"));require_cuda()
    for seed in [int(v) for v in args.seeds.split(",") if v]:
        result=train_one(seed,args,project);print(json.dumps({"seed":seed,"f1":result["f1"],"roc_auc":result["roc_auc"]}),flush=True)


if __name__=="__main__":main()
