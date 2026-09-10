from __future__ import annotations

import argparse
import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_all_equal_calibration import neural_score
from .evaluate_frozen_cascade_fresh_snapshot import verifier_score
from .evaluate_hybrid_external import create_model
from .metrics import classification_metrics
from .train_lexical_verifier import LexicalVerifier
from .url_data import encode_url, psl_aware_lexical_features


METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "false_positive_rate", "false_negative_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private four-condition cascade ablation.")
    parser.add_argument(
        "--contract", type=Path,
        default=Path("artifacts/private_diagnostics/cascade_ablation_2026-08-28/calibration_contract.json"),
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/external/2026-08-28-fresh-raw-5pct/external_urls.csv"),
    )
    parser.add_argument(
        "--collection-report", type=Path,
        default=Path("data/external/2026-08-28-fresh-raw-5pct/collection_report.json"),
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("artifacts/experiments/recent_update_gan10_v1"),
    )
    parser.add_argument(
        "--lexical-scaler", type=Path,
        default=Path("data/experiments/recent_update_gan10_v1/real_only/processed/psl_lexical_scaler.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/private_diagnostics/cascade_ablation_2026-08-28/results.json"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def score_primary(
    root: Path, urls: list[str], raw_features: np.ndarray,
    batch_size: int, device: torch.device,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    outputs = []
    verified = []
    for seed in (42, 43, 44):
        path = root / f"seed_{seed}/checkpoint.pt"
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        scale = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        lexical = (
            (raw_features - np.asarray(saved["lexical_scaler_center"], dtype=np.float32))
            / scale
        ).astype(np.float32)
        model = create_model(saved).to(device)
        model.load_state_dict(saved["state_dict"])
        outputs.append(neural_score(model, tokens, lexical, batch_size, device))
        verified.append({"seed": seed, "path": str(path), "sha256": sha256_file(path)})
        del model, tokens, lexical
        gc.collect()
        torch.cuda.empty_cache()
    return np.mean(np.stack(outputs), axis=0), verified


def paired_correctness(labels: np.ndarray, real: np.ndarray, gan: np.ndarray) -> dict:
    real_correct = real == labels
    gan_correct = gan == labels
    return {
        "both_correct": int(np.sum(real_correct & gan_correct)),
        "both_wrong": int(np.sum(~real_correct & ~gan_correct)),
        "real_correct_gan_wrong": int(np.sum(real_correct & ~gan_correct)),
        "real_wrong_gan_correct": int(np.sum(~real_correct & gan_correct)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Private ablation result already exists: {args.output}")
    contract = load_json(args.contract)
    if contract["status"] != "PRIVATE POST-HOC DIAGNOSTIC; EXCLUDED FROM GRADUATION REPORT":
        raise RuntimeError("The private-only experiment contract is missing")
    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen snapshot hash mismatch")

    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    device = require_cuda()
    raw_features = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)

    primary_scores = {}
    primary_checkpoints = {}
    for key in ("real_only", "gan_10pct"):
        primary_scores[key], primary_checkpoints[key] = score_primary(
            args.model_root / key / "hyb_ms", urls, raw_features, args.batch_size, device
        )

    scaler = np.load(args.lexical_scaler)
    scale = np.where(scaler["scale"] == 0, 1.0, scaler["scale"])
    verifier_features = ((raw_features - scaler["center"]) / scale).astype(np.float32)
    verifier_outputs = []
    verifier_checkpoints = []
    for item in contract["stage2_checkpoints"]:
        path = Path(item["path"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Verifier checkpoint hash mismatch: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        model = LexicalVerifier(int(saved["feature_count"])).to(device)
        model.load_state_dict(saved["state_dict"])
        verifier_outputs.append(verifier_score(model, verifier_features, 2048, device))
        verifier_checkpoints.append({**item, "verified_sha256": actual})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    stage2_scores = np.mean(np.stack(verifier_outputs), axis=0)

    evaluations = {}
    prediction_arrays = {"labels": labels, "stage2_scores": stage2_scores}
    for key in ("real_only", "gan_10pct"):
        config = contract["conditions"][key]
        stage1 = primary_scores[key]
        screen = float(config["screening_rule"]["threshold"])
        hard = float(config["immediate_block_rule"]["threshold"])
        routed = stage1 >= screen
        immediate = stage1 >= hard
        cascade_score = np.where(immediate, 2.0, np.where(routed, stage2_scores, -1.0))
        points = []
        for cascade_config, single_config in zip(
            config["cascade_operating_points"], config["single_stage_operating_points"]
        ):
            cap = float(cascade_config["maximum_fpr"])
            single_threshold = float(single_config["threshold"])
            verifier_threshold = float(cascade_config["verifier_threshold"])
            single_metrics = classification_metrics(labels, stage1, single_threshold)
            cascade_metrics = classification_metrics(labels, cascade_score, verifier_threshold)
            points.append({
                "maximum_calibration_fpr": cap,
                "single_stage_threshold": single_threshold,
                "verifier_threshold": verifier_threshold,
                "single_stage": single_metrics,
                "cascade": cascade_metrics,
                "cascade_minus_single": {
                    metric: cascade_metrics[metric] - single_metrics[metric]
                    for metric in METRIC_KEYS
                },
            })
            cap_name = str(cap).replace(".", "p")
            prediction_arrays[f"{key}_{cap_name}_single"] = (stage1 >= single_threshold).astype(np.uint8)
            prediction_arrays[f"{key}_{cap_name}_cascade"] = (cascade_score >= verifier_threshold).astype(np.uint8)
        evaluations[key] = {
            "display_name": "Real-only HYB-MS" if key == "real_only" else "HYB-MS +10% GAN",
            "checkpoints": primary_checkpoints[key],
            "screening_threshold": screen,
            "immediate_threshold": hard,
            "routed_rows": int(routed.sum()),
            "routed_fraction": float(routed.mean()),
            "operating_points": points,
        }
        prediction_arrays[f"{key}_scores"] = stage1

    comparisons = []
    for index, cap in enumerate(contract["same_fpr_budgets"]):
        real = evaluations["real_only"]["operating_points"][index]
        gan = evaluations["gan_10pct"]["operating_points"][index]
        item = {"maximum_calibration_fpr": cap}
        for branch in ("single_stage", "cascade"):
            item[f"gan_minus_real_{branch}"] = {
                metric: gan[branch][metric] - real[branch][metric] for metric in METRIC_KEYS
            }
            suffix = str(cap).replace(".", "p")
            item[f"paired_{branch}"] = paired_correctness(
                labels,
                prediction_arrays[f"real_only_{suffix}_{'single' if branch == 'single_stage' else 'cascade'}"],
                prediction_arrays[f"gan_10pct_{suffix}_{'single' if branch == 'single_stage' else 'cascade'}"],
            )
        comparisons.append(item)

    payload = {
        "project_version": 3,
        "experiment": "private_four_condition_cascade_ablation_2026-08-28",
        "status": "PRIVATE DIAGNOSTIC COMPLETE; DO NOT ADD TO GRADUATION REPORT",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "dataset": {
            "path": str(args.dataset), "sha256": collection["dataset_sha256"],
            "rows": len(rows), "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
        },
        "same_stage2": True,
        "stage2": contract["stage2"],
        "stage2_checkpoints": verifier_checkpoints,
        "evaluations": evaluations,
        "gan_incremental_effect": comparisons,
        "methodological_limit": contract["methodological_limit"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.parent / "predictions.npz", **prediction_arrays)
    payload["predictions_sha256"] = sha256_file(args.output.parent / "predictions.npz")
    write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "evaluations": evaluations,
        "gan_incremental_effect": comparisons,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
