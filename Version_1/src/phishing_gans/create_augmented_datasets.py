from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .common import hostname_group, iter_csv, load_json, split_name
from .prepare_data import clean_row


REAL_COUNT = 69_300
GENERATED_COUNTS = {20: 17_325, 30: 29_700, 40: 46_200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create unique, controlled GAN-augmented URL datasets."
    )
    parser.add_argument(
        "--real-source",
        type=Path,
        default=Path("Data sets/urls_dataset2M.csv"),
    )
    parser.add_argument(
        "--data-metadata",
        type=Path,
        default=Path("artifacts/data_2m/metadata.json"),
    )
    parser.add_argument(
        "--lsgan-urls",
        type=Path,
        default=Path("artifacts/corpus_lsgan_multisource/filtered_urls.txt"),
    )
    parser.add_argument(
        "--wgan-urls",
        type=Path,
        default=Path("artifacts/corpus_wgan_gp_multisource/filtered_urls.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("augmented data sets")
    )
    parser.add_argument("--seed", type=int, default=2040)
    return parser.parse_args()


def normalized(url: str) -> str:
    return url.strip().lower()


def stratified_targets(metadata: dict) -> dict[int, int]:
    counts = metadata["counts"]["train"]
    benign = int(counts["label_0"])
    malicious = int(counts["label_1"])
    benign_target = round(REAL_COUNT * benign / (benign + malicious))
    return {0: benign_target, 1: REAL_COUNT - benign_target}


def sample_real_training_rows(
    source: Path, metadata: dict, seed: int
) -> list[tuple[str, str]]:
    targets = stratified_targets(metadata)
    reservoirs: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    observed = {0: 0, 1: 0}
    seen: set[str] = set()
    generators = {label: random.Random(seed + label) for label in (0, 1)}
    label_names = {0: "benign", 1: "malicious"}

    for row in iter_csv(source):
        item = clean_row(row)
        if item is None:
            continue
        url, label = item
        if url in seen:
            continue
        seen.add(url)
        split = split_name(
            hostname_group(url),
            metadata["seed"],
            metadata["train_ratio"],
            metadata["val_ratio"],
        )
        if split != "train":
            continue
        count = observed[label]
        observed[label] += 1
        candidate = (url, label_names[label])
        if len(reservoirs[label]) < targets[label]:
            reservoirs[label].append(candidate)
            continue
        replacement = generators[label].randrange(count + 1)
        if replacement < targets[label]:
            reservoirs[label][replacement] = candidate

    for label in (0, 1):
        if observed[label] != int(metadata["counts"]["train"][f"label_{label}"]):
            raise RuntimeError(
                f"Reconstructed {observed[label]} training rows for label {label}; "
                f"expected {metadata['counts']['train'][f'label_{label}']}"
            )
        if len(reservoirs[label]) != targets[label]:
            raise RuntimeError(f"Could not sample enough label-{label} real URLs.")

    result = reservoirs[0] + reservoirs[1]
    random.Random(seed).shuffle(result)
    return result


def load_generated(path: Path, seed: int, forbidden: set[str]) -> list[str]:
    values = []
    seen = set(forbidden)
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        key = normalized(url)
        if not url or key in seen:
            continue
        seen.add(key)
        values.append(url)
    random.Random(seed).shuffle(values)
    if len(values) < GENERATED_COUNTS[40]:
        raise RuntimeError(
            f"{path} has {len(values)} usable unique URLs; "
            f"{GENERATED_COUNTS[40]} are required."
        )
    return values


def combined_pool(
    lsgan: list[str], wgan: list[str], forbidden: set[str], seed: int
) -> list[str]:
    target_each = GENERATED_COUNTS[40] // 2
    selected_lsgan = lsgan[:target_each]
    seen = set(forbidden)
    seen.update(normalized(url) for url in selected_lsgan)
    selected_wgan = []
    for url in wgan:
        key = normalized(url)
        if key in seen:
            continue
        seen.add(key)
        selected_wgan.append(url)
        if len(selected_wgan) == target_each:
            break
    if len(selected_wgan) != target_each:
        raise RuntimeError("Not enough cross-GAN-unique WGAN-GP URLs.")
    result = selected_lsgan + selected_wgan
    random.Random(seed).shuffle(result)
    return result


def write_dataset(
    path: Path, real_rows: list[tuple[str, str]], generated: list[str]
) -> dict:
    seen: set[str] = set()
    benign = malicious_real = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        # CRLF forces csv.writer to quote fields containing either CR or LF.
        # Some real URLs contain an embedded carriage return.
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(("url", "label"))
        for url, label in real_rows:
            key = normalized(url)
            if key in seen:
                raise RuntimeError(f"Duplicate real URL in {path}: {url}")
            seen.add(key)
            if label == "benign":
                benign += 1
            else:
                malicious_real += 1
            writer.writerow((url, label))
        for url in generated:
            key = normalized(url)
            if key in seen:
                raise RuntimeError(f"Duplicate generated URL in {path}: {url}")
            seen.add(key)
            writer.writerow((url, "malicious"))

    # Hash the completed file, including CSV escaping produced by csv.writer.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    total = len(real_rows) + len(generated)
    return {
        "file": str(path),
        "real_rows": len(real_rows),
        "real_benign": benign,
        "real_malicious": malicious_real,
        "generated_malicious": len(generated),
        "total_rows": total,
        "generated_fraction": len(generated) / total,
        "unique_rows": len(seen),
        "uniqueness": len(seen) / total,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    metadata = load_json(args.data_metadata)
    real_rows = sample_real_training_rows(args.real_source, metadata, args.seed)
    real_keys = {normalized(url) for url, _ in real_rows}
    if len(real_keys) != len(real_rows):
        raise RuntimeError("The selected real subset is not unique after normalization.")

    lsgan = load_generated(args.lsgan_urls, args.seed + 10, real_keys)
    wgan = load_generated(args.wgan_urls, args.seed + 20, real_keys)
    pools = {
        "lsgan": lsgan,
        "wgan_gp": wgan,
        "combined_gans": combined_pool(lsgan, wgan, real_keys, args.seed + 30),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for name, pool in pools.items():
        for percentage, generated_count in GENERATED_COUNTS.items():
            path = args.output_dir / f"{name}_{percentage}_percent.csv"
            report = write_dataset(path, real_rows, pool[:generated_count])
            report["gan_condition"] = name
            report["target_generated_percentage"] = percentage
            datasets.append(report)
            print(
                f"{path}: {report['total_rows']} rows, "
                f"{100 * report['generated_fraction']:.2f}% generated, "
                f"{100 * report['uniqueness']:.2f}% unique"
            )

    reference = args.output_dir / "real_only_reference_69300.csv"
    reference_report = write_dataset(reference, real_rows, [])
    manifest = {
        "purpose": "controlled limited-real-data GAN augmentation dose experiment",
        "real_source": str(args.real_source),
        "real_selection": "class-stratified reservoir sample from original hostname-hash training split",
        "real_seed": args.seed,
        "same_real_rows_in_every_dataset": True,
        "generated_subsets_are_nested_within_each_condition": True,
        "lsgan_source": str(args.lsgan_urls),
        "wgan_gp_source": str(args.wgan_urls),
        "combined_gan_ratio": "50% LSGAN / 50% WGAN-GP at the 40% level",
        "reference": reference_report,
        "datasets": datasets,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
