from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .generation import basic_generation_metrics, decode_tokens, sample_hard
from .models import AutoregressiveGenerator
from .relgan import RelGANGenerator
from .url_data import BOS, EOS, PAD, encode_url, encoded_digest, normalize_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and strictly filter one GAN corpus.")
    parser.add_argument("--model", choices=("gumbel_lsgan", "seqgan", "relgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/generated"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--generation-budget", type=int, help="Optional smoke-test override; default uses project config.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last committed generation chunk.")
    parser.add_argument("--max-gpu-temperature", type=int, default=84,
                        help="Pause generation at this Celsius temperature; use 0 to disable.")
    parser.add_argument("--commit-interval", type=int, default=100000,
                        help="Number of generated rows per resumable disk/RNG checkpoint.")
    parser.add_argument("--template-prefix-length", type=int, default=24)
    parser.add_argument("--template-cap", type=int, default=5)
    return parser.parse_args()


def protected_sets(database_path: Path) -> tuple[set[str], set[str], set[bytes]]:
    connection = sqlite3.connect(database_path)
    urls = {row[0] for row in connection.execute("SELECT normalized_url FROM urls")}
    hosts = {row[0] for row in connection.execute("SELECT DISTINCT hostname FROM urls")}
    encoded = {bytes(row[0]) for row in connection.execute("SELECT encoded_hash FROM encoded_rows")}
    connection.close()
    return urls, hosts, encoded


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    shared = project["generators"]["shared"]
    checkpoint_path = args.checkpoint or Path("artifacts/generators") / args.model / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["master_sha256"] != project["data"]["master_sha256"]:
        raise RuntimeError("Generator checkpoint belongs to a different real dataset")
    vocabulary = checkpoint["vocabulary"]
    inverse = {int(value): token for token, value in vocabulary.items()}
    if args.model == "relgan":
        relgan = checkpoint["relgan_config"]
        model = RelGANGenerator(
            len(vocabulary), relgan["embedding_dim"], relgan["memory_slots"],
            relgan["memory_size"], relgan["attention_heads"], relgan["latent_dim"],
        )
    else:
        model = AutoregressiveGenerator(
            len(vocabulary), shared["embedding_dim"], shared["hidden_dim"],
            shared["layers"], shared["dropout"],
        )
    model.load_state_dict(checkpoint["generator_state"])
    device = require_cuda(); model.to(device).eval(); set_seed(int(project["seed"]))
    budget = int(args.generation_budget if args.generation_budget is not None else shared["generation_budget"])
    if budget <= 0:
        raise ValueError("generation budget must be positive")
    if args.commit_interval <= 0:
        raise ValueError("commit interval must be positive")
    output = args.output_root / args.model; output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_samples.txt"; state_path = output / "generation_state.pt"
    generated = 0
    if args.resume:
        if not state_path.exists() or not raw_path.exists():
            raise FileNotFoundError("Resume requires both generation_state.pt and raw_samples.txt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["budget"] != budget or state["checkpoint_sha256"] != sha256_file(checkpoint_path):
            raise RuntimeError("Generation resume state does not match this budget/checkpoint")
        generated = int(state["generated"])
        with raw_path.open("r", encoding="utf-8") as existing:
            existing_rows = sum(1 for _ in existing)
        if existing_rows != generated:
            raise RuntimeError(f"Resume state says {generated:,} rows but raw file has {existing_rows:,}")
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    elif state_path.exists() or (raw_path.exists() and raw_path.stat().st_size > 0):
        raise FileExistsError(f"Generation output already exists at {output}; use --resume")
    checkpoint_hash = sha256_file(checkpoint_path); pending: list[str] = []
    batches_since_temperature_check = 0
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as raw_handle, torch.inference_mode():
        while generated + len(pending) < budget:
            batch = min(args.batch_size, budget - generated - len(pending))
            if args.model == "relgan":
                sampled = model.sample_ids(
                    batch, int(project["data"]["maximum_encoded_length"]), vocabulary[BOS],
                    float(checkpoint["relgan_config"]["initial_temperature"]), device=device,
                )
            else:
                sampled, _ = sample_hard(
                    model, batch, int(project["data"]["maximum_encoded_length"]),
                    vocabulary[BOS], vocabulary[EOS], vocabulary[PAD],
                    float(shared["temperature"]), device,
                )
            pending.extend(decode_tokens(row, inverse) for row in sampled.cpu().numpy())
            batches_since_temperature_check += 1
            if args.max_gpu_temperature > 0 and batches_since_temperature_check >= 100:
                batches_since_temperature_check = 0
                temperature = int(subprocess.check_output(
                    ("nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"),
                    text=True,
                ).strip().splitlines()[0])
                paused = temperature >= args.max_gpu_temperature
                if paused:
                    print(json.dumps({"model": args.model, "thermal_pause_c": temperature,
                                      "resume_below_c": args.max_gpu_temperature - 6}), flush=True)
                while temperature >= args.max_gpu_temperature - 6:
                    time.sleep(10)
                    temperature = int(subprocess.check_output(
                        ("nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"),
                        text=True,
                    ).strip().splitlines()[0])
                if paused:
                    print(json.dumps({"model": args.model, "thermal_resume_c": temperature}), flush=True)
            if len(pending) >= args.commit_interval or generated + len(pending) == budget:
                raw_handle.write("\n".join(pending) + "\n"); raw_handle.flush(); generated += len(pending); pending.clear()
                state_temporary = state_path.with_suffix(".pt.tmp")
                torch.save({"project_version": 2, "model": args.model, "budget": budget,
                            "generated": generated, "checkpoint_sha256": checkpoint_hash,
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state(device)}, state_temporary)
                state_temporary.replace(state_path)
                print(json.dumps({"model": args.model, "generated": generated, "budget": budget,
                                  "committed": True}), flush=True)

    raw_urls = raw_path.read_text(encoding="utf-8").splitlines()
    if len(raw_urls) != budget:
        raise RuntimeError(f"Raw corpus line count {len(raw_urls):,} does not match budget {budget:,}")
    protected_urls, protected_hosts, protected_encoded = protected_sets(args.data_dir / "master.sqlite")
    protected_hashes = {
        hashlib.blake2b(url.encode("utf-8"), digest_size=16).digest()
        for url in protected_urls
    }

    reasons = Counter(); accepted: list[tuple[str, str]] = []
    seen_urls: set[str] = set(); templates = Counter()
    maximum_raw = int(project["data"]["maximum_raw_url_length"])
    maximum_encoded = int(project["data"]["maximum_encoded_length"])
    for raw in raw_urls:
        try:
            normalized, hostname = normalize_url(raw, maximum_raw)
        except ValueError as error:
            reasons[f"invalid::{error}"] += 1; continue
        if normalized in seen_urls:
            reasons["generated_duplicate"] += 1; continue
        seen_urls.add(normalized)
        if normalized in protected_urls:
            reasons["exact_real_overlap"] += 1; continue
        if hostname in protected_hosts:
            reasons["protected_hostname_overlap"] += 1; continue
        if encoded_digest(encode_url(normalized, vocabulary, maximum_encoded)) in protected_encoded:
            reasons["encoded_real_overlap"] += 1; continue
        template = normalized[:args.template_prefix_length]
        if templates[template] >= args.template_cap:
            reasons["template_cap"] += 1; continue
        templates[template] += 1
        accepted.append((normalized, hostname))

    accepted_path = output / "accepted.csv"
    with accepted_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("url", "label", "generator", "hostname"))
        writer.writerows((url, 1, args.model, host) for url, host in accepted)
    raw_quality = basic_generation_metrics(raw_urls, protected_hashes)
    accepted_quality = basic_generation_metrics([row[0] for row in accepted], protected_hashes)
    metrics = {
        "project_version": 2, "model": args.model, "generation_budget": budget,
        "raw": raw_quality, "accepted": accepted_quality,
        "accepted_count": len(accepted), "acceptance_rate": len(accepted) / max(1, budget),
        "rejection_reasons": dict(sorted(reasons.items())),
        "filters": {
            "exact_real_overlap": True, "encoded_real_overlap": True,
            "protected_hostname_overlap": True,
            "template_prefix_length": args.template_prefix_length,
            "template_cap": args.template_cap,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "raw_samples_sha256": sha256_file(raw_path),
        "accepted_csv_sha256": sha256_file(accepted_path),
        "master_sha256": project["data"]["master_sha256"],
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
