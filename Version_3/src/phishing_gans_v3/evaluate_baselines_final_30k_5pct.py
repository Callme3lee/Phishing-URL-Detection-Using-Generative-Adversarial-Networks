from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import METRICS, neural_score, read_csv
from .evaluate_final_prospective_pair import score_condition
from .metrics import classification_metrics
from .models import create_classifier
from .url_data import encode_url, lexical_features, psl_aware_lexical_features


SEEDS = (42, 43, 44)
MODEL_SPECS = {
    "cnn_ms": ("cnn", "neural"),
    "lstm_ms": ("lstm", "neural"),
    "rnn_ms": ("rnn", "neural"),
    "rf_ms": ("random_forest", "traditional"),
    "svm_ms": ("linear_svm", "traditional"),
}
RESULT_IDS = {
    "cnn_ms": "experiments__multisource__baselines__v1__cnn",
    "lstm_ms": "experiments__multisource__baselines__v1__lstm",
    "rnn_ms": "experiments__multisource__baselines__v1__rnn",
    "rf_ms": "experiments__multisource__baselines__v1__random__forest",
    "svm_ms": "experiments__multisource__baselines__v1__linear__svm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen real-only MS baselines on the final 30K 5% dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/external/2026-08-22-final-confirmation-5pct/external_urls.csv"),
    )
    parser.add_argument(
        "--collection-report",
        type=Path,
        default=Path("data/external/2026-08-22-final-confirmation-5pct/collection_report.json"),
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("artifacts/results/all_models_equal_calibration_v1/per_condition"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/all_classifiers_final_30k_5pct_v2"),
    )
    parser.add_argument(
        "--hybrid-protocol",
        type=Path,
        default=Path("artifacts/final_confirmation_protocol_2026-08-22/manifest.json"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    collection = load_json(args.collection_report)
    observed_hash = sha256_file(args.dataset)
    if observed_hash != collection["dataset_sha256"]:
        raise RuntimeError("Frozen 30K dataset hash does not match its collection report")
    if not collection.get("selection_locked_before_inference", False):
        raise RuntimeError("The evaluation dataset was not locked before inference")

    urls, labels = read_csv(args.dataset)
    raw_lexical = np.stack([lexical_features(url) for url in urls]).astype(np.float32)
    raw_psl = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    device = require_cuda()
    results: dict[str, dict] = {}
    prediction_arrays: dict[str, np.ndarray] = {"labels": labels}

    for display_name, (folder, model_format) in MODEL_SPECS.items():
        calibration = load_json(args.calibration_results / f"{RESULT_IDS[display_name]}.json")
        threshold = float(calibration["threshold"])
        seed_scores = []
        checkpoints = []

        for seed in SEEDS:
            seed_dir = Path("artifacts/experiments/multisource_baselines_v1") / folder / f"seed_{seed}"
            if model_format == "neural":
                checkpoint_path = seed_dir / "checkpoint.pt"
                saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
                model = create_classifier(saved["model"], len(saved["vocabulary"]), saved["model_config"]).to(device)
                model.load_state_dict(saved["state_dict"])
                scores = neural_score(model, tokens, None, args.batch_size, device)
                del model
                torch.cuda.empty_cache()
            else:
                checkpoint_path = seed_dir / "model.joblib"
                saved = joblib.load(checkpoint_path)
                model = saved["model"]
                if folder == "random_forest":
                    scores = model.predict_proba(raw_lexical)[:, 1].astype(np.float32)
                else:
                    scores = model.decision_function(raw_lexical).astype(np.float32)
            seed_scores.append(scores)
            checkpoints.append(
                {"seed": seed, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
            )

        ensemble_scores = np.mean(np.stack(seed_scores), axis=0)
        predictions = (ensemble_scores >= threshold).astype(np.uint8)
        metrics = classification_metrics(labels, ensemble_scores, threshold)
        prediction_arrays[f"{display_name}_scores"] = ensemble_scores
        prediction_arrays[f"{display_name}_predictions"] = predictions
        results[display_name] = {
            "model": calibration["model"],
            "training_condition": "real_only_multisource",
            "threshold": threshold,
            "threshold_source": str(args.calibration_results / f"{RESULT_IDS[display_name]}.json"),
            "checkpoints": checkpoints,
            "metrics": metrics,
        }
        print(json.dumps({"model": display_name, "metrics": {key: metrics[key] for key in METRICS}}), flush=True)

    hybrid_protocol = load_json(args.hybrid_protocol)
    for protocol_key, display_name in (
        ("real_only", "hyb_ms_real_only"),
        ("gan_10pct", "hyb_ms_gan_10pct"),
    ):
        condition = hybrid_protocol["conditions"][protocol_key]
        ensemble_scores, checkpoints = score_condition(
            condition, urls, raw_psl, args.batch_size, device
        )
        threshold = float(condition["threshold"])
        predictions = (ensemble_scores >= threshold).astype(np.uint8)
        metrics = classification_metrics(labels, ensemble_scores, threshold)
        prediction_arrays[f"{display_name}_scores"] = ensemble_scores
        prediction_arrays[f"{display_name}_predictions"] = predictions
        results[display_name] = {
            "model": "HYB-MS",
            "training_condition": "real_only_multisource" if protocol_key == "real_only" else "multisource_plus_10pct_ms_hardgan_c",
            "threshold": threshold,
            "threshold_source": str(args.hybrid_protocol),
            "checkpoints": checkpoints,
            "metrics": metrics,
        }
        print(json.dumps({"model": display_name, "metrics": {key: metrics[key] for key in METRICS}}), flush=True)

    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "predictions.npz", **prediction_arrays)
    payload = {
        "project_version": 3,
        "evaluation_name": "All frozen MS classifiers on final 30K low-prevalence confirmation set",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dataset": {
            "path": str(args.dataset),
            "sha256": observed_hash,
            "rows": len(urls),
            "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "phishing_prevalence": float(np.mean(labels)),
        },
        "protocol": {
            "models_retrained": False,
            "thresholds_changed": False,
            "ensemble": "mean score across frozen seeds 42, 43 and 44",
            "thresholds": "registered equal-calibration thresholds",
            "report_updated": False,
        },
        "conditions": results,
    }
    payload["predictions_sha256"] = sha256_file(args.output / "predictions.npz")
    write_json(args.output / "results.json", payload)

    with (args.output / "all_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "threshold", *METRICS, "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            metrics = result["metrics"]
            matrix = metrics["confusion_matrix"]
            writer.writerow(
                {
                    "model": name,
                    "threshold": result["threshold"],
                    **{key: metrics[key] for key in METRICS},
                    **{key: matrix[key] for key in ("tn", "fp", "fn", "tp")},
                }
            )
    print(json.dumps({"output": str(args.output), "report_updated": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
