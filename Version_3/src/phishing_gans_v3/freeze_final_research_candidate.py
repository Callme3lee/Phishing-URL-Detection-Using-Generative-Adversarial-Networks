from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json


OUTPUT = Path("artifacts/final_candidate_2026-08-22/manifest.json")
REGISTRY = Path("artifacts/results/all_models_equal_calibration_v1/per_condition")


def frozen_condition(path: Path, role: str) -> dict:
    record = load_json(path)
    checkpoints = []
    for item in record["checkpoints"]:
        checkpoint = Path(item["path"])
        actual = sha256_file(checkpoint)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        checkpoints.append({
            "seed": int(item["seed"]),
            "path": str(checkpoint),
            "sha256": actual,
        })
    return {
        "role": role,
        "id": record["id"],
        "display_name": (
            "Weighted HYB-MS +10% MS-HardGAN-C"
            if role == "final_candidate"
            else "Weighted HYB-MS real-only"
        ),
        "stored_real_training_rows": 363_692,
        "synthetic_training_rows": 40_410 if role == "final_candidate" else 0,
        "sampling": "source-balanced weighted sampling",
        "ensemble": "arithmetic mean of seeds 42, 43 and 44",
        "threshold": float(record["threshold"]),
        "threshold_policy": record["threshold_policy"],
        "checkpoints": checkpoints,
        "source_registry": str(path),
        "source_registry_sha256": sha256_file(path),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Final candidate is already frozen at {OUTPUT}")
    control_path = REGISTRY / "experiments__hybrid__cnn__coral__v1.json"
    candidate_path = REGISTRY / "experiments__fair__hybrid__gan__v1__gan__10pct__hyb__ms.json"
    calibration = Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv")
    payload = {
        "project_version": 3,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "final controlled real-only versus 10% GAN augmentation research comparison",
        "status": "FROZEN_RESEARCH_CANDIDATE_NOT_DEPLOYMENT_READY",
        "selection_basis": (
            "Selected before the prospective test from the equal-calibration 10K and 30K results; "
            "no prospective labels or scores were available."
        ),
        "positive_class": "phishing",
        "conditions": {
            "real_only": frozen_condition(control_path, "control"),
            "gan_10pct": frozen_condition(candidate_path, "final_candidate"),
        },
        "calibration": {
            "path": str(calibration),
            "sha256": sha256_file(calibration),
            "used_for_prospective_tuning": False,
        },
        "prospective_contract": {
            "evaluate_both_conditions_once": True,
            "no_retraining": True,
            "no_threshold_changes": True,
            "no_checkpoint_selection": True,
            "no_test_based_model_selection": True,
            "practical_gate": {"maximum_fpr": 0.02, "minimum_recall": 0.80},
            "augmentation_support_rule": (
                "gan_10pct must exceed real_only on recall, F1 and ROC-AUC without a higher FPR"
            ),
        },
    }
    write_json(OUTPUT, payload)
    readme = OUTPUT.parent / "README.md"
    readme.write_text(
        "# Frozen final research candidate\n\n"
        "The final candidate is **weighted HYB-MS +10% MS-HardGAN-C**. "
        "The paired control is **weighted HYB-MS real-only**.\n\n"
        "Both ensembles, all six checkpoint hashes, their separate frozen thresholds, "
        "and the one-shot prospective-test rules are stored in `manifest.json`.\n\n"
        "This freeze is for the scientific augmentation comparison. It does not claim deployment readiness.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
