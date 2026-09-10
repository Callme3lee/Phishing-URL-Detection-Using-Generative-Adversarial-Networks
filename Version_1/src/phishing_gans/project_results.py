from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASELINES = {
    "CNN": Path("artifacts/classifier/metrics.json"),
    "LSTM": Path("artifacts/classifier_lstm/metrics.json"),
    "Random Forest": Path("artifacts/classifier_rf/metrics.json"),
    "Linear SVM": Path("artifacts/classifier_svm/metrics.json"),
}
GANS = {
    "LSTM-LSGAN": {
        "corpus": Path("artifacts/corpus_lsgan_multisource"),
        "checkpoint": Path(
            "artifacts/autoregressive_lsgan_multisource/checkpoint_epoch_002.pt"
        ),
    },
    "LSTM-WGAN-GP": {
        "corpus": Path("artifacts/corpus_wgan_gp_multisource"),
        "checkpoint": Path(
            "artifacts/autoregressive_wgan_gp_multisource/checkpoint_epoch_002.pt"
        ),
    },
}
CLASSIFIER_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate final baseline metrics and GAN uniqueness results."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final_project_results.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(path: Path) -> dict[str, float]:
    payload = load_json(path)
    metrics = payload.get("test", payload.get("metrics"))
    if metrics is None:
        raise ValueError(f"No test metrics found in {path}")
    return {name: float(metrics[name]) for name in CLASSIFIER_METRICS}


def uniqueness(tokens: np.ndarray) -> tuple[int, float]:
    unique_count = len({row.tobytes() for row in tokens})
    return unique_count, unique_count / max(len(tokens), 1)


def load_gan(directory: Path, checkpoint: Path) -> dict[str, int | float | str | dict]:
    raw = np.load(directory / "raw_tokens.npy", mmap_mode="r")
    filtered = np.load(directory / "filtered_tokens.npy", mmap_mode="r")
    filter_report = load_json(directory / "filter_report.json")
    raw_unique_count, raw_unique_rate = uniqueness(raw)
    filtered_unique_count, filtered_unique_rate = uniqueness(filtered)
    return {
        "selected_checkpoint": str(checkpoint),
        "raw_count": len(raw),
        "raw_unique_count": raw_unique_count,
        "raw_uniqueness": raw_unique_rate,
        "accepted_count": len(filtered),
        "acceptance_rate": len(filtered) / max(len(raw), 1),
        "filtered_unique_count": filtered_unique_count,
        "filtered_uniqueness": filtered_unique_rate,
        "rejections": filter_report["rejections"],
    }


def print_results(results: dict) -> None:
    print("BASELINE CLASSIFIERS (untouched hostname-separated real test set)")
    print("Model\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC")
    for model, metrics in results["baselines"].items():
        values = "\t".join(f"{100 * metrics[name]:.4f}%" for name in CLASSIFIER_METRICS)
        print(f"{model}\t{values}")

    print("\nRETRAINED GAN UNIQUENESS")
    print("GAN\tRaw count\tRaw unique\tRaw uniqueness\tAccepted\tFiltered uniqueness")
    for model, metrics in results["gans"].items():
        print(
            f"{model}\t{metrics['raw_count']}\t{metrics['raw_unique_count']}\t"
            f"{100 * metrics['raw_uniqueness']:.4f}%\t{metrics['accepted_count']}\t"
            f"{100 * metrics['filtered_uniqueness']:.4f}%"
        )


def main() -> None:
    args = parse_args()
    results = {
        "metric_definitions": {
            "positive_class": "malicious/phishing",
            "classifier_metrics": list(CLASSIFIER_METRICS),
            "gan_uniqueness": "unique encoded generated URLs divided by generated URL count",
        },
        "gan_training_data": "artifacts/gan_data_multisource",
        "baselines": {
            model: load_baseline(path) for model, path in BASELINES.items()
        },
        "gans": {
            model: load_gan(specification["corpus"], specification["checkpoint"])
            for model, specification in GANS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_results(results)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
