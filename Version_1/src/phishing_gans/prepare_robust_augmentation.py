from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import load_json, save_json, split_name
from .lexical_features import FEATURE_NAMES, extract_lexical_features
from .prepare_multisource_gan_data import decode_row, safe_hostname_group


SPLITS = ("train", "val", "test")
LABELS = (0, 1)


@dataclass(frozen=True)
class Source:
    name: str
    directory: Path
    already_split: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare source-balanced real-only and quality-filtered GAN augmentation "
            "conditions with identical validation and test sets."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument(
        "--external-phishing-dir",
        type=Path,
        default=Path("artifacts/external_phishing_test"),
    )
    parser.add_argument(
        "--external-new-dir",
        type=Path,
        default=Path("artifacts/external_new_data_urls"),
    )
    parser.add_argument(
        "--lsgan-tokens",
        type=Path,
        default=Path("artifacts/corpus_lsgan_multisource/filtered_tokens.npy"),
    )
    parser.add_argument(
        "--wgan-tokens",
        type=Path,
        default=Path("artifacts/corpus_wgan_gp_multisource/filtered_tokens.npy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/robust_augmentation"),
    )
    parser.add_argument("--train-per-source-class", type=int, default=15_000)
    parser.add_argument("--val-per-source-class", type=int, default=3_000)
    parser.add_argument("--test-per-source-class", type=int, default=5_000)
    parser.add_argument("--generated-percent", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2742)
    return parser.parse_args()


def token_hash(row: np.ndarray) -> int:
    return int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "big")


def sample_indices(
    indices: np.ndarray, target: int, seed: int, description: str
) -> np.ndarray:
    if len(indices) < target:
        raise RuntimeError(f"{description}: only {len(indices)} rows; {target} required")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=target, replace=False))


def external_candidates(
    source: Source,
    inverse: dict[int, str],
    seed: int,
    seen_hashes: set[int],
) -> dict[str, dict[int, np.ndarray]]:
    tokens = np.load(source.directory / "test_tokens.npy", mmap_mode="r")
    labels = np.load(source.directory / "test_labels.npy", mmap_mode="r")
    values: dict[str, dict[int, list[int]]] = {
        split: {label: [] for label in LABELS} for split in SPLITS
    }
    duplicates = 0
    for index in range(len(labels)):
        row = tokens[index]
        key = token_hash(row)
        if key in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(key)
        url = decode_row(row, inverse)
        split = split_name(safe_hostname_group(url), seed, 0.8, 0.1)
        values[split][int(labels[index])].append(index)
    print(f"{source.name}: excluded {duplicates} cross-source encoded duplicates")
    return {
        split: {
            label: np.asarray(values[split][label], dtype=np.int64)
            for label in LABELS
        }
        for split in SPLITS
    }


def base_candidates(source: Source) -> dict[str, dict[int, np.ndarray]]:
    result = {}
    seen: set[int] = set()
    for split in SPLITS:
        tokens = np.load(source.directory / f"{split}_tokens.npy", mmap_mode="r")
        labels = np.load(source.directory / f"{split}_labels.npy", mmap_mode="r")
        selected = {label: [] for label in LABELS}
        for index in range(len(labels)):
            key = token_hash(tokens[index])
            if key in seen:
                continue
            seen.add(key)
            selected[int(labels[index])].append(index)
        result[split] = {
            label: np.asarray(selected[label], dtype=np.int64) for label in LABELS
        }
    return result


def encoded_hashes_for_split_source(source: Source) -> set[int]:
    """Protect every base encoding, including collisions caused by truncation."""
    hashes: set[int] = set()
    duplicate_encodings = 0
    for split in SPLITS:
        tokens = np.load(source.directory / f"{split}_tokens.npy", mmap_mode="r")
        for row in tokens:
            key = token_hash(row)
            if key in hashes:
                duplicate_encodings += 1
            hashes.add(key)
    print(
        f"{source.name}: protected {len(hashes)} encoded rows "
        f"({duplicate_encodings} post-truncation collisions)"
    )
    return hashes


def selected_rows(
    sources: list[Source],
    candidates: dict[str, dict[str, dict[int, np.ndarray]]],
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[tuple[Source, np.ndarray]]]:
    result = {split: [] for split in SPLITS}
    for source_number, source in enumerate(sources):
        for split_number, split in enumerate(SPLITS):
            chosen = []
            for label in LABELS:
                indices = sample_indices(
                    candidates[source.name][split][label],
                    targets[split],
                    seed + 1000 * source_number + 100 * split_number + label,
                    f"{source.name}/{split}/label-{label}",
                )
                chosen.append(indices)
            result[split].append((source, np.concatenate(chosen)))
    return result


def source_arrays(source: Source, split: str) -> tuple[np.ndarray, np.ndarray]:
    actual_split = split if source.already_split else "test"
    return (
        np.load(source.directory / f"{actual_split}_tokens.npy", mmap_mode="r"),
        np.load(source.directory / f"{actual_split}_labels.npy", mmap_mode="r"),
    )


def materialize_real_splits(
    output: Path,
    selections: dict[str, list[tuple[Source, np.ndarray]]],
    inverse: dict[int, str],
    max_length: int,
) -> tuple[dict[str, int], set[int], np.ndarray]:
    data_dir = output / "data" / "real_only"
    feature_dir = output / "features" / "real_only"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    real_hashes: set[int] = set()
    malicious_train = []
    for split in SPLITS:
        total = sum(len(indices) for _, indices in selections[split])
        counts[split] = total
        out_tokens = np.lib.format.open_memmap(
            data_dir / f"{split}_tokens.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, max_length),
        )
        out_labels = np.lib.format.open_memmap(
            data_dir / f"{split}_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
        )
        out_features = np.lib.format.open_memmap(
            feature_dir / f"{split}_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total, len(FEATURE_NAMES)),
        )
        feature_labels = np.lib.format.open_memmap(
            feature_dir / f"{split}_labels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total,),
        )
        offset = 0
        for source, indices in selections[split]:
            tokens, labels = source_arrays(source, split)
            for index in indices:
                row = np.asarray(tokens[index])
                label = int(labels[index])
                key = token_hash(row)
                if key in real_hashes:
                    raise RuntimeError("A selected encoded URL appears in more than one split/source")
                real_hashes.add(key)
                out_tokens[offset] = row
                out_labels[offset] = label
                feature_labels[offset] = label
                out_features[offset] = extract_lexical_features(decode_row(row, inverse))
                if split == "train" and label == 1:
                    malicious_train.append(row.copy())
                offset += 1
        out_tokens.flush()
        out_labels.flush()
        out_features.flush()
        feature_labels.flush()
    return counts, real_hashes, np.asarray(malicious_train, dtype=np.uint8)


def generated_candidates(path: Path, forbidden: set[int]) -> np.ndarray:
    tokens = np.load(path, mmap_mode="r")
    accepted = []
    seen = set(forbidden)
    for row in tokens:
        key = token_hash(row)
        if key in seen or not np.any(row):
            continue
        seen.add(key)
        accepted.append(np.asarray(row).copy())
    return np.asarray(accepted, dtype=np.uint8)


def quality_order(pool: np.ndarray, real_malicious: np.ndarray) -> np.ndarray:
    vocab_size = max(int(pool.max()), int(real_malicious.max())) + 1
    counts = np.bincount(real_malicious.ravel(), minlength=vocab_size).astype(np.float64)
    counts[0] = 0
    probabilities = (counts + 1.0) / (counts.sum() + vocab_size)
    real_lengths = np.count_nonzero(real_malicious, axis=1)
    median_length = float(np.median(real_lengths))
    mad_length = float(np.median(np.abs(real_lengths - median_length))) + 1.0
    scores = np.empty(len(pool), dtype=np.float64)
    for index, row in enumerate(pool):
        values = row[row != 0]
        nll = float(-np.log(probabilities[values]).mean())
        length_penalty = abs(len(values) - median_length) / mad_length
        unique_ratio = len(np.unique(values)) / max(len(values), 1)
        scores[index] = nll + 0.20 * length_penalty - 0.15 * unique_ratio
    return np.argsort(scores, kind="stable")


def take_cross_unique(
    first: np.ndarray,
    second: np.ndarray,
    total: int,
    first_order: np.ndarray,
    second_order: np.ndarray,
) -> np.ndarray:
    first_target = total // 2
    second_target = total - first_target
    result = []
    seen: set[int] = set()
    for pool, order, target in (
        (first, first_order, first_target),
        (second, second_order, second_target),
    ):
        added = 0
        for index in order:
            row = pool[int(index)]
            key = token_hash(row)
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
            added += 1
            if added == target:
                break
        if added != target:
            raise RuntimeError(f"Needed {target} generated rows, selected {added}")
    return np.asarray(result, dtype=np.uint8)


def link_fixed_splits(source: Path, destination: Path, suffix: str) -> None:
    for split in ("val", "test"):
        target = destination / f"{split}_{suffix}.npy"
        if target.exists():
            target.unlink()
        target.hardlink_to(source / f"{split}_{suffix}.npy")


def write_condition(
    root: Path,
    name: str,
    generated: np.ndarray,
    base_metadata: dict,
    source_names: list[str],
) -> dict:
    real_data = root / "data" / "real_only"
    real_features = root / "features" / "real_only"
    real_tokens = np.load(real_data / "train_tokens.npy", mmap_mode="r")
    real_labels = np.load(real_data / "train_labels.npy", mmap_mode="r")
    real_x = np.load(real_features / "train_features.npy", mmap_mode="r")
    data_dir = root / "data" / name
    feature_dir = root / "features" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    total = len(real_labels) + len(generated)
    tokens = np.lib.format.open_memmap(
        data_dir / "train_tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, real_tokens.shape[1]),
    )
    labels = np.lib.format.open_memmap(
        data_dir / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    features = np.lib.format.open_memmap(
        feature_dir / "train_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, len(FEATURE_NAMES)),
    )
    feature_labels = np.lib.format.open_memmap(
        feature_dir / "train_labels.npy", mode="w+", dtype=np.uint8, shape=(total,)
    )
    tokens[: len(real_tokens)] = real_tokens
    labels[: len(real_labels)] = real_labels
    features[: len(real_x)] = real_x
    feature_labels[: len(real_labels)] = real_labels
    inverse = {
        int(value): key
        for key, value in base_metadata["vocab"].items()
        if len(key) == 1
    }
    for offset, row in enumerate(generated, start=len(real_tokens)):
        tokens[offset] = row
        labels[offset] = 1
        feature_labels[offset] = 1
        features[offset] = extract_lexical_features(decode_row(row, inverse))
    tokens.flush()
    labels.flush()
    features.flush()
    feature_labels.flush()
    link_fixed_splits(real_data, data_dir, "tokens")
    link_fixed_splits(real_data, data_dir, "labels")
    link_fixed_splits(real_features, feature_dir, "features")
    link_fixed_splits(real_features, feature_dir, "labels")
    val_total = len(np.load(real_data / "val_labels.npy", mmap_mode="r"))
    test_total = len(np.load(real_data / "test_labels.npy", mmap_mode="r"))
    label_0 = int(np.count_nonzero(real_labels == 0))
    label_1 = int(np.count_nonzero(real_labels == 1)) + len(generated)
    metadata = dict(base_metadata)
    metadata.update(
        {
            "experiment_condition": name,
            "sources": source_names,
            "source_balanced": True,
            "generated_rows": len(generated),
            "counts": {
                "train": {"total": total, "label_0": label_0, "label_1": label_1},
                "val": {"total": val_total},
                "test": {"total": test_total},
            },
        }
    )
    save_json(data_dir / "metadata.json", metadata)
    save_json(
        feature_dir / "metadata.json",
        {
            "experiment_condition": name,
            "feature_names": list(FEATURE_NAMES),
            "counts": {"train": total, "val": val_total, "test": test_total},
        },
    )
    return {"name": name, "real": len(real_labels), "generated": len(generated), "total": total}


def main() -> None:
    args = parse_args()
    if not 0 < args.generated_percent < 50:
        raise ValueError("generated-percent must be between 0 and 50")
    metadata = load_json(args.base_dir / "metadata.json")
    inverse = {
        int(value): key for key, value in metadata["vocab"].items() if len(key) == 1
    }
    sources = [
        Source("base_2m", args.base_dir, True),
        Source("url_dataset", args.external_phishing_dir, False),
        Source("new_data_urls", args.external_new_dir, False),
    ]
    candidates = {sources[0].name: base_candidates(sources[0])}
    # The raw external preparation already removes exact URL overlap. Protecting
    # encoded base rows additionally catches distinct long URLs that become equal
    # after the classifier's 128-character truncation.
    seen_external = encoded_hashes_for_split_source(sources[0])
    for source in sources[1:]:
        candidates[source.name] = external_candidates(
            source, inverse, metadata["seed"], seen_external
        )
    targets = {
        "train": args.train_per_source_class,
        "val": args.val_per_source_class,
        "test": args.test_per_source_class,
    }
    selections = selected_rows(sources, candidates, targets, args.seed)
    counts, real_hashes, real_malicious = materialize_real_splits(
        args.output_root, selections, inverse, metadata["max_length"]
    )
    real_data_dir = args.output_root / "data" / "real_only"
    real_feature_dir = args.output_root / "features" / "real_only"
    base_condition = dict(metadata)
    per_split_labels = {
        split: np.load(real_data_dir / f"{split}_labels.npy", mmap_mode="r")
        for split in SPLITS
    }
    base_condition.update(
        {
            "experiment_condition": "real_only",
            "sources": [source.name for source in sources],
            "source_balanced": True,
            "selection_seed": args.seed,
            "counts": {
                split: {
                    "total": counts[split],
                    "label_0": int(np.count_nonzero(per_split_labels[split] == 0)),
                    "label_1": int(np.count_nonzero(per_split_labels[split] == 1)),
                }
                for split in SPLITS
            },
        }
    )
    save_json(real_data_dir / "metadata.json", base_condition)
    save_json(
        real_feature_dir / "metadata.json",
        {
            "experiment_condition": "real_only",
            "feature_names": list(FEATURE_NAMES),
            "counts": counts,
        },
    )

    generated_count = math.ceil(
        counts["train"] * args.generated_percent / (100 - args.generated_percent)
    )
    lsgan = generated_candidates(args.lsgan_tokens, real_hashes)
    wgan = generated_candidates(args.wgan_tokens, real_hashes)
    lsgan_quality = quality_order(lsgan, real_malicious)
    wgan_quality = quality_order(wgan, real_malicious)
    rng = np.random.default_rng(args.seed + 999)
    lsgan_random = rng.permutation(len(lsgan))
    wgan_random = rng.permutation(len(wgan))
    conditions = {
        "combined_random_30": take_cross_unique(
            lsgan, wgan, generated_count, lsgan_random, wgan_random
        ),
        "combined_quality_30": take_cross_unique(
            lsgan, wgan, generated_count, lsgan_quality, wgan_quality
        ),
        "lsgan_quality_30": lsgan[lsgan_quality[:generated_count]],
    }
    reports = [
        {"name": "real_only", "real": counts["train"], "generated": 0, "total": counts["train"]}
    ]
    for name, generated in conditions.items():
        reports.append(
            write_condition(
                args.output_root,
                name,
                generated,
                metadata,
                [source.name for source in sources],
            )
        )
        print(f"{name}: {len(generated)} generated rows")
    save_json(
        args.output_root / "manifest.json",
        {
            "purpose": "source-balanced robust augmentation comparison",
            "sources": [source.name for source in sources],
            "selection_seed": args.seed,
            "per_source_class": targets,
            "generated_target_percent": args.generated_percent,
            "all_conditions_share_real_rows": True,
            "all_conditions_share_validation_and_test": True,
            "quality_filter": (
                "rank by character-language likelihood, robust length distance, and "
                "within-URL character diversity"
            ),
            "conditions": reports,
        },
    )
    print(f"Prepared robust experiment under {args.output_root}")


if __name__ == "__main__":
    main()
