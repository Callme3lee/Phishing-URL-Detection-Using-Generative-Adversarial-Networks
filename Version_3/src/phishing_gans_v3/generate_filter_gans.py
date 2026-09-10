from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, set_seed, sha256_file, write_json
from .gan_utils import basic_quality, decode_tokens, sample_hard
from .train_gans import create_models
from .url_data import (BOS, EOS, PAD, PSL_EXTRACTOR, canonicalize_url,
                       encode_url, encoded_digest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and freeze strictly filtered Version 3 GAN URL pools.")
    parser.add_argument("--model", choices=("seqgan", "relgan", "urlgan"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/generators"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def protected_sets(database: Path) -> tuple[set[str], set[str], set[bytes]]:
    connection = sqlite3.connect(database)
    urls = {row[0] for row in connection.execute("SELECT normalized_url FROM urls")}
    hosts = {row[0] for row in connection.execute("SELECT hostname FROM urls")}
    hashes = {row[0] for row in connection.execute("SELECT encoded_hash FROM encoded_rows")}
    connection.close()
    return urls, hosts, hashes


def strict_filter(raw_urls: list[str], protected_urls: set[str], protected_hosts: set[str],
                  protected_hashes: set[bytes], vocabulary: dict[str, int], maximum: int):
    accepted: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    templates: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for raw in raw_urls:
        try:
            normalized, hostname, _ = canonicalize_url(raw)
        except ValueError as error:
            reasons[str(error)] += 1
            continue
        extracted = PSL_EXTRACTOR(hostname)
        if not extracted.domain or not extracted.suffix:
            reasons["not_tld_like"] += 1
            continue
        if normalized in seen:
            reasons["duplicate_generated_url"] += 1
            continue
        if normalized in protected_urls:
            reasons["real_url_overlap"] += 1
            continue
        if hostname in protected_hosts:
            reasons["real_hostname_overlap"] += 1
            continue
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if digest in protected_hashes:
            reasons["real_encoded_overlap"] += 1
            continue
        template = normalized[:24]
        if templates[template] >= 5:
            reasons["template_cap"] += 1
            continue
        seen.add(normalized)
        templates[template] += 1
        accepted.append((normalized, hostname, hashlib.sha256(raw.encode("utf-8")).hexdigest()))
    return accepted, reasons


@torch.inference_mode()
def generate(name: str, generator, count: int, batch_size: int, shared: dict,
             vocabulary: dict[str, int], device: torch.device) -> list[str]:
    inverse = {int(value): token for token, value in vocabulary.items()}
    output: list[str] = []
    generator.eval()
    while len(output) < count:
        batch = min(batch_size, count - len(output))
        if name == "seqgan":
            ids, _ = sample_hard(generator, batch, int(shared["maximum_length"]), vocabulary[BOS],
                                 vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device)
        elif name == "urlgan":
            ids = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                       vocabulary[EOS], vocabulary[PAD], float(shared["temperature"]), device)
        else:
            ids = generator.sample_ids(batch, int(shared["maximum_length"]), vocabulary[BOS],
                                       float(shared["temperature"]), device)
        output.extend(decode_tokens(row, inverse) for row in ids.cpu().numpy())
        if len(output) % (batch_size * 100) == 0:
            print(json.dumps({"stage": "generate", "model": name, "samples": len(output)}), flush=True)
    return output


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    shared = project["generators"]["shared"]
    output = args.output_root / args.model
    if (output / "audit.json").exists():
        raise FileExistsError(f"Frozen generated pool already exists at {output}")
    device = require_cuda()
    set_seed(int(project["seed"]))
    vocabulary = load_json(args.data_dir / "vocabulary.json")
    checkpoint_path = args.checkpoint_root / args.model / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["master_sha256"] != project["data"]["master_sha256"]:
        raise RuntimeError("Generator checkpoint belongs to a different real-data split")
    generator, _ = create_models(args.model, len(vocabulary), project, device)
    generator.load_state_dict(checkpoint["generator_state"])
    budget = int(shared["generation_budget"])
    started = time.perf_counter()
    raw_urls = generate(args.model, generator, budget, args.batch_size, shared, vocabulary, device)
    generation_seconds = time.perf_counter() - started
    protected_urls, protected_hosts, protected_hashes = protected_sets(args.data_dir / "master.sqlite")
    accepted, reasons = strict_filter(raw_urls, protected_urls, protected_hosts, protected_hashes,
                                      vocabulary, int(shared["maximum_length"]))
    real_training_rows = len(np.load(args.data_dir / "train_labels.npy", mmap_mode="r"))
    required = {
        str(int(round(fraction * 100))): int(round(real_training_rows * fraction / (1 - fraction)))
        for fraction in project["augmentation"]["generated_fractions"]
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_samples.txt").write_text("\n".join(raw_urls) + "\n", encoding="utf-8")
    with (output / "accepted.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("url", "hostname", "raw_sha256")); writer.writerows(accepted)
    accepted_urls = [row[0] for row in accepted]
    report = {
        "project_version": 3, "model": args.model, "seed": int(project["seed"]),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "generation_budget": budget, "generation_seconds": generation_seconds,
        "real_training_rows": real_training_rows,
        "raw_quality": basic_quality(raw_urls, protected_urls),
        "accepted_rows": len(accepted), "accepted_fraction": len(accepted) / budget,
        "accepted_quality": basic_quality(accepted_urls, protected_urls),
        "rejection_counts": dict(reasons), "required_generated_rows": required,
        "supports_all_frozen_fractions": len(accepted) >= max(required.values()),
        "filter_contract": ["canonical URL", "TLD-like", "no generated duplicate",
                            "no real URL/hostname/encoded overlap", "maximum five per 24-character prefix"],
    }
    write_json(output / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
