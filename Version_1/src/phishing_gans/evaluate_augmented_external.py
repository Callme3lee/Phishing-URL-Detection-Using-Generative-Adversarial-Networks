from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import require_cuda
from .data import EncodedURLDataset
from .models import create_character_model
from .train_classifier import metrics as neural_metrics
from .train_traditional import evaluate as traditional_metrics
from .train_traditional import score_model


MODELS = ("cnn", "lstm", "rf", "svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every augmented-data checkpoint on GAN-unseen external sources."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/augmented_experiments"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional result directory for an independent random seed.",
    )
    parser.add_argument(
        "--external-root", type=Path, default=Path("artifacts/external_heldout")
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def evaluate_neural(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[str, float, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint["model_type"]
    model = create_character_model(model_type, **checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probabilities = []
    labels = []
    for tokens, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities.append(torch.sigmoid(model(tokens)).cpu().numpy())
        labels.append(targets.numpy())
    threshold = float(checkpoint["threshold"])
    result = neural_metrics(
        np.concatenate(labels).astype(np.uint8),
        np.concatenate(probabilities),
        threshold,
    )
    del model
    torch.cuda.empty_cache()
    return model_type, threshold, result


def main() -> None:
    args = parse_args()
    device = require_cuda()
    results_root = args.results_root or (args.experiment_root / "results")
    conditions = sorted(
        path.name
        for path in results_root.iterdir()
        if path.is_dir()
    )
    sources = sorted(
        path.name for path in (args.external_root / "data").iterdir() if path.is_dir()
    )
    total = len(conditions) * len(MODELS) * len(sources)
    position = 0

    for source in sources:
        data_dir = args.external_root / "data" / source
        feature_dir = args.external_root / "features" / source
        dataset = EncodedURLDataset(data_dir, "test")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        external_features = np.load(feature_dir / "test_features.npy", mmap_mode="r")
        external_labels = np.load(feature_dir / "test_labels.npy", mmap_mode="r")

        for condition in conditions:
            for model_name in MODELS:
                position += 1
                result_dir = results_root / condition / model_name
                output = result_dir / f"external_heldout_{source}.json"
                if output.exists():
                    print(
                        f"[{position}/{total}] SKIP {source} {condition} {model_name}",
                        flush=True,
                    )
                    continue
                if model_name in {"cnn", "lstm"}:
                    model_type, threshold, result = evaluate_neural(
                        result_dir / "best.pt", loader, device
                    )
                    payload = {
                        "source": source,
                        "condition": condition,
                        "model": model_type,
                        "frozen_threshold": threshold,
                        "metrics": result,
                    }
                else:
                    model = joblib.load(result_dir / "model.joblib")
                    training_metrics = json.loads(
                        (result_dir / "metrics.json").read_text(encoding="utf-8")
                    )
                    threshold = float(training_metrics["validation"]["threshold"])
                    scores = score_model(model, model_name, external_features)
                    result = traditional_metrics(external_labels, scores, threshold)
                    payload = {
                        "source": source,
                        "condition": condition,
                        "model": model_name,
                        "frozen_threshold": threshold,
                        "metrics": result,
                    }
                output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(
                    f"[{position}/{total}] DONE {source} {condition} {model_name} "
                    f"F1={100 * result['f1']:.2f}% ROC={100 * result['roc_auc']:.2f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main()
