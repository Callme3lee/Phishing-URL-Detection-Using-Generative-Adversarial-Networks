from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .calibrate_two_stage_cascade import CAPS, optimize_cascade, threshold_for_recall
from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the private real-vs-GAN cascade ablation without test labels."
    )
    parser.add_argument(
        "--calibration-root", type=Path,
        default=Path("artifacts/results/updated_models_calibration_v1"),
    )
    parser.add_argument(
        "--gan-cascade-calibration", type=Path,
        default=Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/calibration_predictions.npz"),
    )
    parser.add_argument(
        "--gan-cascade-freeze", type=Path,
        default=Path("artifacts/experiments/two_stage_url_cascade_v1/calibration/frozen_cascade.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/private_diagnostics/cascade_ablation_2026-08-28/calibration_contract.json"),
    )
    return parser.parse_args()


def condition_contract(labels: np.ndarray, stage1: np.ndarray, stage2: np.ndarray) -> dict:
    screen = threshold_for_recall(labels, stage1, .95)
    hard = float(select_threshold(labels, stage1, .001)["threshold"])
    cascades = [optimize_cascade(labels, stage1, stage2, screen, hard, cap) for cap in CAPS]
    singles = []
    for cap in CAPS:
        selected = select_threshold(labels, stage1, cap)
        singles.append({
            "maximum_fpr": cap,
            "threshold": float(selected["threshold"]),
            "metrics": classification_metrics(labels, stage1, float(selected["threshold"])),
        })
    return {
        "screening_rule": {"minimum_calibration_recall": .95, "threshold": screen},
        "immediate_block_rule": {"maximum_calibration_fpr": .001, "threshold": hard},
        "cascade_operating_points": cascades,
        "single_stage_operating_points": singles,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Private calibration contract already exists: {args.output}")

    condition_files = {
        "real_only": args.calibration_root / "real_only/calibration_predictions.npz",
        "gan_10pct": args.calibration_root / "gan_10pct/calibration_predictions.npz",
    }
    loaded = {key: np.load(path) for key, path in condition_files.items()}
    labels = loaded["real_only"]["labels"].astype(np.uint8)
    if not np.array_equal(labels, loaded["gan_10pct"]["labels"]):
        raise RuntimeError("Real and GAN calibration labels differ")

    verifier = np.load(args.gan_cascade_calibration)
    if not np.array_equal(labels, verifier["labels"].astype(np.uint8)):
        raise RuntimeError("Verifier and Stage-1 calibration labels differ")
    stage2 = verifier["stage2_scores"].astype(np.float32)
    frozen = load_json(args.gan_cascade_freeze)

    conditions = {}
    for key in ("real_only", "gan_10pct"):
        stage1 = loaded[key]["scores"].astype(np.float32)
        conditions[key] = condition_contract(labels, stage1, stage2)
        conditions[key]["stage1_predictions"] = {
            "path": str(condition_files[key]),
            "sha256": sha256_file(condition_files[key]),
        }

    payload = {
        "project_version": 3,
        "experiment": "private_cascade_ablation_2026-08-28",
        "status": "PRIVATE POST-HOC DIAGNOSTIC; EXCLUDED FROM GRADUATION REPORT",
        "calibration_only_before_evaluation": True,
        "fresh_snapshot_labels_used_for_thresholds": False,
        "calibration_dataset": frozen["dataset"],
        "calibration_dataset_sha256": frozen["dataset_sha256"],
        "same_calibration_rows": True,
        "same_fpr_budgets": list(CAPS),
        "same_threshold_rules": True,
        "same_stage2_scores": True,
        "stage2": frozen["stage2"],
        "stage2_training_rows": frozen["stage2_training_rows"],
        "stage2_gan_training_rows": frozen["stage2_gan_training_rows"],
        "stage2_checkpoints": frozen["checkpoints"],
        "stage2_calibration_predictions": {
            "path": str(args.gan_cascade_calibration),
            "sha256": sha256_file(args.gan_cascade_calibration),
        },
        "conditions": conditions,
        "methodological_limit": (
            "The numerical thresholds use only the earlier calibration cohort, but this ablation "
            "was designed after the August 28 labels were viewed. It is diagnostic, not a new "
            "prospective confirmation result."
        ),
    }
    args.output.parent.mkdir(parents=True)
    write_json(args.output, payload)
    print({
        key: {
            "screen": conditions[key]["screening_rule"]["threshold"],
            "hard": conditions[key]["immediate_block_rule"]["threshold"],
            "single_thresholds": [p["threshold"] for p in conditions[key]["single_stage_operating_points"]],
            "cascade_thresholds": [p["verifier_threshold"] for p in conditions[key]["cascade_operating_points"]],
        }
        for key in conditions
    })


if __name__ == "__main__":
    main()
