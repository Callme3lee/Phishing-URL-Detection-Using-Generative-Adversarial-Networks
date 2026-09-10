from __future__ import annotations

import argparse
import hashlib
import json
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
from .generation import basic_generation_metrics, decode_tokens, sample_gumbel_soft, sample_hard, strided_rollout_rewards, teacher_forcing_inputs
from .models import AutoregressiveGenerator, SequenceDiscriminator
from .url_data import BOS


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Train Gumbel-LSGAN or SeqGAN on the shared malicious split.")
    parser.add_argument("--model",choices=("gumbel_lsgan","seqgan"),required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/project.json"))
    parser.add_argument("--data-dir",type=Path,default=Path("data/processed"))
    parser.add_argument("--pretrained",type=Path,default=Path("artifacts/generators/pretrained/best.pt"))
    parser.add_argument("--output-root",type=Path,default=Path("artifacts/generators"))
    parser.add_argument("--num-workers",type=int,default=2)
    return parser.parse_args()


def protected_hashes(path:Path)->set[bytes]:
    connection=sqlite3.connect(path);result={hashlib.blake2b(url.encode(),digest_size=16).digest() for (url,) in connection.execute("SELECT normalized_url FROM urls")};connection.close();return result


def mle_loss(generator,real,bos_id):
    logits,_=generator(teacher_forcing_inputs(real,bos_id))
    return nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),real.reshape(-1),ignore_index=0)


def train_discriminator(discriminator,optimizer,real,fake,model_name):
    optimizer.zero_grad(set_to_none=True);real_score,_=discriminator(real);fake_score,_=discriminator(fake.detach())
    if model_name=="gumbel_lsgan":loss=.5*((real_score-1).square().mean()+fake_score.square().mean())
    else:loss=nn.functional.binary_cross_entropy_with_logits(real_score,torch.ones_like(real_score))+nn.functional.binary_cross_entropy_with_logits(fake_score,torch.zeros_like(fake_score))
    loss.backward();nn.utils.clip_grad_norm_(discriminator.parameters(),1.0);optimizer.step();return float(loss.item())


def seqgan_policy_loss(generator,discriminator,batch_size,max_length,bos_id,eos_id,pad_id,temperature,device,mle,weight,rollout_count,rollout_stride):
    tokens,logs=sample_hard(generator,batch_size,max_length,bos_id,eos_id,pad_id,temperature,device,True)
    rewards=strided_rollout_rewards(generator,discriminator,tokens,bos_id,eos_id,pad_id,temperature,rollout_count,rollout_stride);baseline=rewards.mean(dim=0,keepdim=True)
    mask=tokens.ne(0).float();policy=-((((rewards-baseline).detach())*logs*mask).sum()/mask.sum().clamp(min=1))
    return policy+weight*mle,tokens,float(rewards.mean().item())


def main()->None:
    args=parse_args();project=load_json(args.config);shared=project["generators"]["shared"];specific=project["generators"][args.model]
    device=require_cuda();seed=project["seed"];set_seed(seed);checkpoint=torch.load(args.pretrained,map_location=device,weights_only=False);vocabulary=checkpoint["vocabulary"];inverse={int(v):k for k,v in vocabulary.items()};bos_id=vocabulary[BOS]
    eos_id=vocabulary["<EOS>"];pad_id=vocabulary["<PAD>"]
    generator=AutoregressiveGenerator(len(vocabulary),shared["embedding_dim"],shared["hidden_dim"],shared["layers"],shared["dropout"]).to(device);generator.load_state_dict(checkpoint["state_dict"])
    discriminator=SequenceDiscriminator(len(vocabulary)).to(device)
    generator_optimizer=torch.optim.AdamW(generator.parameters(),lr=specific["learning_rate_generator"])
    discriminator_optimizer=torch.optim.AdamW(discriminator.parameters(),lr=specific["learning_rate_discriminator"])
    dataset=GeneratorDataset(args.data_dir,"train",shared["training_sample_cap"],seed);loader=DataLoader(dataset,batch_size=shared["adversarial_batch_size"],shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,drop_last=True)
    hashes=protected_hashes(args.data_dir/"master.sqlite");output=args.output_root/args.model;output.mkdir(parents=True,exist_ok=True);history=[];best_score=-float("inf");best_generator=None;best_discriminator=None
    iterator=iter(loader);max_length=project["data"]["maximum_encoded_length"]
    for epoch in range(1,int(shared["adversarial_epochs"])+1):
        generator.train();discriminator.train();d_losses=[];g_losses=[];rewards=[];started=time.perf_counter()
        steps = int(specific.get("adversarial_steps_per_epoch", shared["adversarial_steps_per_epoch"]))
        for _ in range(steps):
            try:real=next(iterator)
            except StopIteration:iterator=iter(loader);real=next(iterator)
            real=real.to(device,non_blocking=True);batch=len(real)
            with torch.no_grad():fake_hard,_=sample_hard(generator,batch,max_length,bos_id,eos_id,pad_id,shared["temperature"],device)
            d_losses.append(train_discriminator(discriminator,discriminator_optimizer,real,fake_hard,args.model))
            generator_optimizer.zero_grad(set_to_none=True);language=mle_loss(generator,real,bos_id)
            if args.model=="gumbel_lsgan":
                soft=sample_gumbel_soft(generator,batch,max_length,bos_id,shared["temperature"],device);fake_score,fake_features=discriminator.forward_soft(soft)
                with torch.no_grad():_,real_features=discriminator(real)
                adversarial=.5*(fake_score-1).square().mean();feature=(fake_features.mean(0)-real_features.mean(0)).square().mean();marginal=soft.mean((0,1));diversity=(marginal*torch.log(marginal.clamp(min=1e-8))).sum()
                loss=adversarial+specific["language_model_weight"]*language+specific["feature_matching_weight"]*feature+specific["diversity_weight"]*diversity
            else:
                loss,_,reward=seqgan_policy_loss(generator,discriminator,batch,max_length,bos_id,eos_id,pad_id,shared["temperature"],device,language,specific["mle_regularization_weight"],specific["rollout_count"],specific["rollout_stride"]);rewards.append(reward)
            loss.backward();nn.utils.clip_grad_norm_(generator.parameters(),1.0);generator_optimizer.step();g_losses.append(float(loss.item()))
        generator.eval()
        with torch.inference_mode():sampled,_=sample_hard(generator,2000,max_length,bos_id,eos_id,pad_id,shared["temperature"],device)
        urls=[decode_tokens(row,inverse) for row in sampled.cpu().numpy()];quality=basic_generation_metrics(urls,hashes);selection=quality["validity"]*quality["uniqueness"]*(1-quality["memorization_rate"])*(1-quality["repeated_template_rate"])
        record={"epoch":epoch,"generator_loss_mean":float(np.mean(g_losses)),"discriminator_loss_mean":float(np.mean(d_losses)),"reward_mean":float(np.mean(rewards)) if rewards else None,"selection_score":selection,"seconds":time.perf_counter()-started,"quality":quality};history.append(record);write_json(output/f"epoch_{epoch:02d}_metrics.json",record);(output/f"epoch_{epoch:02d}_samples.txt").write_text("\n".join(urls)+"\n",encoding="utf-8");print(json.dumps(record),flush=True)
        if selection>best_score:best_score=selection;best_generator=deepcopy(generator.state_dict());best_discriminator=deepcopy(discriminator.state_dict())
        torch.save(
            {"project_version":2,"model":args.model,"epoch":epoch,
             "generator_state":generator.state_dict(),"discriminator_state":discriminator.state_dict(),
             "generator_config":shared,"vocabulary":vocabulary,"selection_score":selection,
             "master_sha256":project["data"]["master_sha256"]},
            output/"latest.pt",
        )
        torch.save(
            {"project_version":2,"model":args.model,"epoch":epoch,
             "generator_state":best_generator,"discriminator_state":best_discriminator,
             "generator_config":shared,"vocabulary":vocabulary,"selection_score":best_score,
             "master_sha256":project["data"]["master_sha256"]},
            output/"best_so_far.pt",
        )
    torch.save({"project_version":2,"model":args.model,"generator_state":best_generator,"discriminator_state":best_discriminator,"generator_config":shared,"vocabulary":vocabulary,"selection_score":best_score,"master_sha256":project["data"]["master_sha256"],"seqgan_note":f"strided Monte-Carlo rollouts: count={specific['rollout_count']}, stride={specific['rollout_stride']}" if args.model=="seqgan" else None},output/"best.pt");write_json(output/"history.json",history)


if __name__=="__main__":main()
