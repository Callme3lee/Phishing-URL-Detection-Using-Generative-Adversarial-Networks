from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch

from .common import load_json, require_cuda
from .models import create_character_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated URL tokens.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--classifier", type=Path, default=Path("artifacts/classifier/best.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def decode_rows(rows: np.ndarray, vocab: dict[str, int]) -> list[str]:
    inverse = {value: key for key, value in vocab.items() if len(key) == 1}
    urls = []
    for row in rows:
        chars = []
        for raw in row:
            value = int(raw)
            if value == 0:
                break
            chars.append(inverse.get(value, "�") if value != 1 else "�")
        urls.append("".join(chars))
    return urls


def valid_structure(url: str) -> bool:
    try:
        parsed = urlsplit(url if "://" in url else f"http://{url}")
    except ValueError:
        return False
    host = (parsed.hostname or "").strip(".")
    if not host or "." not in host or " " in host:
        return False
    tld = host.rsplit(".", 1)[-1]
    return 2 <= len(tld) <= 24 and tld.isalpha()


def ngram_diversity(urls: list[str], n: int = 2) -> float:
    all_ngrams = []
    for url in urls:
        all_ngrams.extend(url[index : index + n] for index in range(len(url) - n + 1))
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def memorization_rate(samples: np.ndarray, data_dir: Path) -> float:
    generated = {row.tobytes() for row in samples}
    matches: set[bytes] = set()
    train_tokens = np.load(data_dir / "train_tokens.npy", mmap_mode="r")
    train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    for index in np.flatnonzero(train_labels == 1):
        value = train_tokens[index].tobytes()
        if value in generated:
            matches.add(value)
    return len(matches) / max(len(generated), 1)


@torch.inference_mode()
def classifier_scores(samples: np.ndarray, checkpoint_path: Path, batch_size: int):
    device = require_cuda()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn")
    model = create_character_model(model_type, **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities = []
    for start in range(0, len(samples), batch_size):
        tokens = torch.from_numpy(samples[start : start + batch_size].copy()).long().to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities.append(torch.sigmoid(model(tokens)).cpu().numpy())
    values = np.concatenate(probabilities)
    return values, float(checkpoint["threshold"])


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples)
    metadata = load_json(args.data_dir / "metadata.json")
    urls = decode_rows(samples, metadata["vocab"])
    unique = len(set(urls))
    scores, threshold = classifier_scores(samples, args.classifier, args.batch_size)
    lengths = np.asarray([len(url) for url in urls])
    report = {
        "sample_file": str(args.samples),
        "count": len(urls),
        "unique_rate": unique / max(len(urls), 1),
        "empty_rate": sum(not url for url in urls) / max(len(urls), 1),
        "valid_structure_rate": sum(valid_structure(url) for url in urls) / max(len(urls), 1),
        "training_memorization_rate": memorization_rate(samples, args.data_dir),
        "bigram_diversity": ngram_diversity(urls, 2),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "cnn_mean_malicious_probability": float(scores.mean()),
        "cnn_malicious_detection_rate": float((scores >= threshold).mean()),
        "cnn_threshold": threshold,
    }
    output = args.output or args.samples.with_suffix(".metrics.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
