from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_json, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze nested pattern-diverse MS-HardGAN-C augmentation pools.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated/ms_hardgan_v2_final"))
    parser.add_argument("--base-data-dir", type=Path, default=Path("data/experiments/hybrid_cnn_coral_v1/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/augmentations/ms_hardgan_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Frozen augmentation directory exists: {args.output_dir}")
    project = load_json(args.config); audit = load_json(args.generated_dir / "audit.json")
    with (args.generated_dir / "augmentation_pool.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    base_rows = sum(1 for _ in (args.base_data_dir / "train_manifest.csv").open("r", encoding="utf-8")) - 1
    required = {
        str(int(round(fraction * 100))): int(round(base_rows * fraction / (1 - fraction)))
        for fraction in project["augmentation"]["generated_fractions"]
    }
    if len(rows) < max(required.values()):
        raise RuntimeError(f"Need {max(required.values())} rows; only {len(rows)} passed the frozen filters")

    # Stable round-robin across requested patterns. Sorting by a seeded hash makes
    # every smaller condition a strict prefix of every larger condition.
    buckets: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["requested_pattern"])].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: hashlib.blake2b(f"42:{row['url']}".encode(), digest_size=16).digest())
    ordered: list[dict[str, str]] = []
    offset = 0
    while len(ordered) < len(rows):
        progressed = False
        for pattern in sorted(buckets):
            if offset < len(buckets[pattern]):
                ordered.append(buckets[pattern][offset]); progressed = True
        if not progressed:
            break
        offset += 1

    args.output_dir.mkdir(parents=True)
    files = {}
    for ratio, count in required.items():
        target = args.output_dir / f"generated_{ratio}pct.csv"
        selected = ordered[:count]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
        files[ratio] = {
            "path": str(target), "sha256": sha256_file(target), "generated_rows": count,
            "final_training_rows": base_rows + count,
            "actual_fraction": count / (base_rows + count),
            "pattern_counts": dict(Counter(row["requested_pattern"] for row in selected)),
            "source_counts": dict(Counter(row["requested_source"] for row in selected)),
        }
    report = {
        "condition": "ms_hardgan_v2_nested_augmentations", "project_version": 3,
        "base_real_rows": base_rows, "generator": "ms_hardgan_v2",
        "generator_checkpoint_sha256": audit["checkpoint_sha256"],
        "accepted_rows": audit["accepted_rows"], "challenge_holdout_rows": audit["held_out_challenge_rows"],
        "eligible_augmentation_rows": len(rows), "required_rows": required,
        "nested": True, "selection": "seeded BLAKE2b ordering with round-robin requested-pattern balancing while each pattern remains available",
        "challenge_holdout_excluded": True, "files": files,
    }
    write_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
