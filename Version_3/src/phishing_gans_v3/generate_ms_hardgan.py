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
from .gan_utils import basic_quality, decode_tokens
from .generate_filter_gans import strict_filter
from .ms_hardgan_models import ConditionalURLGenerator, MultiHeadURLDiscriminator
from .url_data import BOS, EOS, PAD, encode_url, encoded_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and freeze the defensive MS-HardGAN pool.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments/ms_hardgan_v1"))
    parser.add_argument("--original-data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/generators/ms_hardgan_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/ms_hardgan_v1"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--generator-config", default="ms_hardgan")
    return parser.parse_args()


def protected_sets(original_data_dir: Path, ms_data_dir: Path, vocabulary: dict, maximum: int):
    connection = sqlite3.connect(original_data_dir / "master.sqlite")
    urls = {row[0] for row in connection.execute("SELECT normalized_url FROM urls")}
    hosts = {row[0] for row in connection.execute("SELECT hostname FROM urls")}
    hashes = {row[0] for row in connection.execute("SELECT encoded_hash FROM encoded_rows")}
    connection.close()
    with (ms_data_dir / "train_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            urls.add(row["url"]); hosts.add(row["canonical_hostname"])
            hashes.add(encoded_digest(encode_url(row["url"], vocabulary, maximum)))
    return urls, hosts, hashes


@torch.inference_mode()
def category_agreement(discriminator, rows: list[tuple[str, int, int]], vocabulary: dict,
                       maximum: int, device: torch.device) -> dict:
    pattern_correct = source_correct = total = 0
    for start in range(0, len(rows), 1024):
        batch = rows[start:start + 1024]
        tokens = torch.from_numpy(np.stack([
            encode_url(row[0], vocabulary, maximum) for row in batch
        ]).astype(np.int64)).to(device)
        _, pattern_logits, source_logits, _ = discriminator(tokens)
        patterns = torch.tensor([row[1] for row in batch], device=device)
        sources = torch.tensor([row[2] for row in batch], device=device)
        pattern_correct += int(pattern_logits.argmax(1).eq(patterns).sum().item())
        source_correct += int(source_logits.argmax(1).eq(sources).sum().item())
        total += len(batch)
    return {
        "rows": total,
        "requested_pattern_agreement": pattern_correct / max(1, total),
        "requested_source_agreement": source_correct / max(1, total),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Frozen output already exists: {args.output_dir}")
    project = load_json(args.config); config = project["generators"][args.generator_config]
    shared = project["generators"]["shared"]; vocabulary = load_json(args.data_dir / "vocabulary.json")
    maximum = int(shared["maximum_length"]); device = require_cuda(); set_seed(int(project["seed"]))
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if saved["preparation_sha256"] != sha256_file(args.data_dir / "preparation_report.json"):
        raise RuntimeError("Checkpoint and frozen MS corpus disagree")
    generator = ConditionalURLGenerator(
        len(vocabulary), int(config["pattern_count"]), int(config["source_count"]),
        int(shared["embedding_dim"]), int(config["condition_dim"]), int(shared["hidden_dim"]),
        int(config["generator_layers"]), int(config["latent_dim"]), float(config["generator_dropout"]),
    ).to(device)
    discriminator = MultiHeadURLDiscriminator(
        len(vocabulary), int(config["pattern_count"]), int(config["source_count"]),
        int(shared["embedding_dim"]), int(config["discriminator_channels"]),
        tuple(config["discriminator_kernel_sizes"]), float(config["discriminator_dropout"]),
    ).to(device)
    generator.load_state_dict(saved["generator_state"]); discriminator.load_state_dict(saved["discriminator_state"])
    generator.eval(); discriminator.eval()
    sources_array = np.load(args.data_dir / "train_sources.npy", mmap_mode="r")
    source_probabilities = torch.as_tensor(
        np.bincount(sources_array, minlength=int(config["source_count"])) / len(sources_array),
        dtype=torch.float32, device=device,
    )
    budget = args.budget or int(config["generation_budget"])
    inverse = {int(value): token for token, value in vocabulary.items()}
    raw_urls: list[str] = []; requested_patterns: list[int] = []; requested_sources: list[int] = []
    started = time.perf_counter()
    while len(raw_urls) < budget:
        batch = min(args.batch_size, budget - len(raw_urls))
        patterns = torch.randint(int(config["pattern_count"]), (batch,), device=device)
        sources = torch.multinomial(source_probabilities, batch, replacement=True)
        ids = generator.sample_ids(
            patterns, sources, maximum, vocabulary[BOS], vocabulary[EOS], vocabulary[PAD],
            float(shared["temperature"]),
        )
        raw_urls.extend(decode_tokens(row, inverse) for row in ids.cpu().numpy())
        requested_patterns.extend(patterns.cpu().tolist()); requested_sources.extend(sources.cpu().tolist())
        if len(raw_urls) % (args.batch_size * 100) == 0:
            print(json.dumps({"stage": "generate", "samples": len(raw_urls)}), flush=True)
    generation_seconds = time.perf_counter() - started

    protected_urls, protected_hosts, protected_hashes = protected_sets(
        args.original_data_dir, args.data_dir, vocabulary, maximum
    )
    accepted, reasons = strict_filter(
        raw_urls, protected_urls, protected_hosts, protected_hashes, vocabulary, maximum
    )
    condition_by_hash: dict[str, tuple[int, int]] = {}
    for raw, pattern, source in zip(raw_urls, requested_patterns, requested_sources):
        condition_by_hash.setdefault(hashlib.sha256(raw.encode("utf-8")).hexdigest(), (pattern, source))
    accepted_rows = [
        (url, hostname, raw_hash, *condition_by_hash[raw_hash])
        for url, hostname, raw_hash in accepted
    ]
    agreement_rows = [(url, pattern, source) for url, _, _, pattern, source in accepted_rows]
    agreement = category_agreement(discriminator, agreement_rows, vocabulary, maximum, device)

    challenge_fraction = float(config["held_out_challenge_fraction"])
    challenge_rows, augmentation_rows = [], []
    for row in accepted_rows:
        fraction = int(hashlib.blake2b(f"42:{row[0]}".encode(), digest_size=8).hexdigest(), 16) / 2**64
        (challenge_rows if fraction < challenge_fraction else augmentation_rows).append(row)
    args.output_dir.mkdir(parents=True)
    fieldnames = ("url", "hostname", "raw_sha256", "requested_pattern", "requested_source")
    for name, rows in (("accepted.csv", accepted_rows), ("challenge_holdout.csv", challenge_rows),
                       ("augmentation_pool.csv", augmentation_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(fieldnames); writer.writerows(rows)
    with (args.output_dir / "raw_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("raw_url", "requested_pattern", "requested_source"))
        writer.writerows(zip(raw_urls, requested_patterns, requested_sources))

    accepted_urls = [row[0] for row in accepted_rows]
    report = {
        "project_version": 3, "generator": args.generator_config, "defensive_offline_only": True,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "generation_budget": budget, "generation_seconds": generation_seconds,
        "raw_quality": basic_quality(raw_urls, protected_urls),
        "accepted_rows": len(accepted_rows), "accepted_fraction": len(accepted_rows) / max(1, budget),
        "accepted_quality": basic_quality(accepted_urls, protected_urls),
        "rejection_counts": dict(reasons), "category_agreement": agreement,
        "requested_pattern_counts": dict(Counter(map(str, requested_patterns))),
        "requested_source_counts": dict(Counter(map(str, requested_sources))),
        "held_out_challenge_rows": len(challenge_rows), "augmentation_pool_rows": len(augmentation_rows),
        "split_method": "BLAKE2b-64(seed 42, canonical URL); challenge holdout never used for augmentation selection",
        "interpretation": "hard examples approximate plausible unseen variants; they do not predict zero-day attacks",
    }
    write_json(args.output_dir / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
