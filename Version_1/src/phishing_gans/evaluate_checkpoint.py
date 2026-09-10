from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import require_cuda
from .data import EncodedURLDataset
from .models import create_character_model
from .train_classifier import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen neural checkpoint on encoded URLs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Optional compressed NPZ containing labels, probabilities and frozen-threshold predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = require_cuda()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn")
    model = create_character_model(model_type, **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = EncodedURLDataset(args.data_dir, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    probabilities = []
    labels = []
    for tokens, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities.append(torch.sigmoid(model(tokens)).cpu().numpy())
        labels.append(targets.numpy())
    probabilities_array = np.concatenate(probabilities)
    labels_array = np.concatenate(labels).astype(np.uint8)
    result = {
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "frozen_threshold": float(checkpoint["threshold"]),
        "metrics": metrics(
            labels_array, probabilities_array, float(checkpoint["threshold"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            labels=labels_array,
            probabilities=probabilities_array,
            predictions=(probabilities_array >= float(checkpoint["threshold"])).astype(np.uint8),
            threshold=np.asarray(float(checkpoint["threshold"]), dtype=np.float32),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
