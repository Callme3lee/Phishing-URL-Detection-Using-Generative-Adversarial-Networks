from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .common import load_json, require_cuda
from .data import EncodedURLDataset
from .models import create_character_model
from .summarize_robust_augmentation import CONDITIONS, MODELS
from .train_classifier import best_f1_threshold, metrics, predict
from .train_traditional import best_threshold, evaluate, score_model


SOURCE_ORDER = ("base_2m", "url_dataset", "new_data_urls")
EXTERNAL_SOURCES = ("url_dataset", "new_data_urls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate robust-experiment checkpoints using a threshold selected from "
            "the matching source's validation hostnames."
        )
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/robust_augmentation")
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="Condition directory to evaluate; repeat for multiple conditions.",
    )
    parser.add_argument(
        "--model", action="append", choices=MODELS, default=None
    )
    parser.add_argument(
        "--external-root", type=Path, default=Path("artifacts/external_heldout")
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def loader(dataset, args) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_root or (args.experiment_root / "results")
    conditions = args.condition or list(CONDITIONS)
    models = args.model or list(MODELS)
    manifest = load_json(args.experiment_root / "manifest.json")
    per_source_val = 2 * int(manifest["per_source_class"]["val"])
    real_val_data = EncodedURLDataset(
        args.experiment_root / "data" / "real_only", "val"
    )
    real_val_features = np.load(
        args.experiment_root / "features" / "real_only" / "val_features.npy",
        mmap_mode="r",
    )
    real_val_labels = np.load(
        args.experiment_root / "features" / "real_only" / "val_labels.npy",
        mmap_mode="r",
    )
    device = require_cuda()

    for source in EXTERNAL_SOURCES:
        source_index = SOURCE_ORDER.index(source)
        start = source_index * per_source_val
        stop = start + per_source_val
        validation_loader = loader(
            Subset(real_val_data, range(start, stop)), args
        )
        test_data = EncodedURLDataset(args.external_root / "data" / source, "test")
        test_loader = loader(test_data, args)
        test_features = np.load(
            args.external_root / "features" / source / "test_features.npy",
            mmap_mode="r",
        )
        test_labels = np.load(
            args.external_root / "features" / source / "test_labels.npy",
            mmap_mode="r",
        )
        for condition in conditions:
            for model_name in models:
                result_dir = results_root / condition / model_name
                output = result_dir / f"calibrated_external_{source}.json"
                if output.exists():
                    print(f"SKIP {source} {condition} {model_name}")
                    continue
                if model_name in {"cnn", "lstm"}:
                    checkpoint = torch.load(
                        result_dir / "best.pt", map_location=device, weights_only=False
                    )
                    model = create_character_model(
                        checkpoint["model_type"], **checkpoint["model_args"]
                    ).to(device)
                    model.load_state_dict(checkpoint["model_state"])
                    val_labels, val_scores = predict(
                        model, validation_loader, device
                    )
                    threshold = best_f1_threshold(val_labels, val_scores)
                    heldout_labels, heldout_scores = predict(model, test_loader, device)
                    result = metrics(heldout_labels, heldout_scores, threshold)
                    del model
                    torch.cuda.empty_cache()
                else:
                    model = joblib.load(result_dir / "model.joblib")
                    val_scores = score_model(
                        model, model_name, real_val_features[start:stop]
                    )
                    threshold = best_threshold(real_val_labels[start:stop], val_scores)
                    heldout_scores = score_model(model, model_name, test_features)
                    result = evaluate(test_labels, heldout_scores, threshold)
                output.write_text(
                    json.dumps(
                        {
                            "source": source,
                            "condition": condition,
                            "model": model_name,
                            "calibration": (
                                "F1 threshold selected on matching-source validation hostnames"
                            ),
                            "validation_slice": [start, stop],
                            "threshold": threshold,
                            "metrics": result,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"{source} {condition} {model_name}: "
                    f"F1={100 * result['f1']:.2f}% ROC={100 * result['roc_auc']:.2f}%"
                )


if __name__ == "__main__":
    main()
