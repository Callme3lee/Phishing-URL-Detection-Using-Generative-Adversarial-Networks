from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import load_json, require_cuda, set_seed, write_json
from .datasets import GeneratorDataset
from .generation import basic_generation_metrics, decode_tokens, sample_hard, teacher_forcing_inputs
from .models import AutoregressiveGenerator
from .url_data import BOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the shared Version 2 URL generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/generators/pretrained"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


@torch.inference_mode()
def evaluate_nll(model, loader, device, bos_id) -> tuple[float, int]:
    model.eval(); total_loss = 0.0; total_tokens = 0
    for targets in loader:
        targets = targets.to(device, non_blocking=True)
        inputs = teacher_forcing_inputs(targets, bos_id)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=0, reduction="sum",
            )
        total_loss += float(loss.item()); total_tokens += int(targets.ne(0).sum().item())
    return total_loss / max(1, total_tokens), total_tokens


def protected_hashes(database_path: Path) -> set[bytes]:
    connection = sqlite3.connect(database_path)
    result = {
        hashlib.blake2b(url.encode("utf-8"), digest_size=16).digest()
        for (url,) in connection.execute("SELECT normalized_url FROM urls")
    }
    connection.close(); return result


def main() -> None:
    args = parse_args(); project = load_json(args.config); shared = project["generators"]["shared"]
    device = require_cuda(); seed = int(project["seed"]); set_seed(seed)
    vocabulary = load_json(args.data_dir / "vocabulary.json"); inverse = {int(v):k for k,v in vocabulary.items()}
    train_data = GeneratorDataset(args.data_dir, "train", shared["training_sample_cap"], seed)
    validation_data = GeneratorDataset(args.data_dir, "validation", shared["validation_sample_cap"], seed)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    validation_loader = DataLoader(validation_data,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0)
    model = AutoregressiveGenerator(
        len(vocabulary), shared["embedding_dim"], shared["hidden_dim"], shared["layers"], shared["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda"); bos_id = vocabulary[BOS]
    history=[]; best_nll=float("inf"); best_state=None; hashes=protected_hashes(args.data_dir/"master.sqlite")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,int(shared["pretraining_epochs"])+1):
        model.train(); total_loss=0.0; total_tokens=0; started=time.perf_counter()
        for targets in train_loader:
            targets=targets.to(device,non_blocking=True);inputs=teacher_forcing_inputs(targets,bos_id)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits,_=model(inputs)
                loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=0)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update()
            count=int(targets.ne(0).sum().item());total_loss+=float(loss.item())*count;total_tokens+=count
        validation_nll,_=evaluate_nll(model,validation_loader,device,bos_id)
        with torch.inference_mode():
            sample_tokens,_=sample_hard(
                model, 2000, project["data"]["maximum_encoded_length"], bos_id,
                vocabulary["<EOS>"], vocabulary["<PAD>"], shared["temperature"], device
            )
        urls=[decode_tokens(row,inverse) for row in sample_tokens.cpu().numpy()]
        quality=basic_generation_metrics(urls,hashes)
        record={"epoch":epoch,"train_nll":total_loss/max(1,total_tokens),"validation_nll":validation_nll,"validation_perplexity":math.exp(min(20,validation_nll)),"seconds":time.perf_counter()-started,"quality":quality}
        history.append(record);write_json(args.output_dir/f"epoch_{epoch:02d}_metrics.json",record);(args.output_dir/f"epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls)+"\n",encoding="utf-8")
        print(json.dumps(record),flush=True)
        if validation_nll<best_nll:best_nll=validation_nll;best_state=deepcopy(model.state_dict())
    if best_state is None:raise RuntimeError("No pretrained checkpoint")
    checkpoint={"project_version":2,"model":"autoregressive_pretrained","state_dict":best_state,"vocabulary":vocabulary,"model_config":shared,"validation_nll":best_nll,"master_sha256":project["data"]["master_sha256"]}
    torch.save(checkpoint,args.output_dir/"best.pt");write_json(args.output_dir/"history.json",history)


if __name__ == "__main__":main()
