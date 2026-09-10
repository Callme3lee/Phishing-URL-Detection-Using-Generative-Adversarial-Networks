from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json


def main() -> None:
    candidate = Path("artifacts/final_candidate_2026-08-22/manifest.json")
    output = Path("artifacts/final_confirmation_protocol_2026-08-22/manifest.json")
    if output.exists():
        raise FileExistsError(f"Protocol already frozen at {output}")
    source = load_json(candidate)
    payload = {
        "project_version": 3,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate_manifest": str(candidate),
        "source_candidate_manifest_sha256": sha256_file(candidate),
        "conditions": source["conditions"],
        "immutability": {
            "retraining_allowed": False,
            "recalibration_allowed": False,
            "threshold_changes_allowed": False,
            "checkpoint_selection_allowed": False,
            "dataset_selection_after_inference_allowed": False,
        },
        "primary_confirmation": {
            "rows": 30_000,
            "legitimate_rows": 28_500,
            "phishing_rows": 1_500,
            "phishing_prevalence": 0.05,
            "identity_contract": "zero URL, hostname, and 128-token identity overlap with every earlier project cohort",
            "augmentation_support_rule": (
                "gan_10pct must exceed real_only on recall, F1, and ROC-AUC "
                "without a higher false-positive rate"
            ),
            "practical_gate": {"maximum_false_positive_rate": 0.02, "minimum_recall": 0.80},
        },
        "long_url_stress": {
            "legitimate_rows": 2_000,
            "phishing_rows": 200,
            "purpose": "separate non-regression test for long legitimate paths and queries",
            "minimum_legitimate_url_length": 80,
            "gan_non_regression_rule": "gan_10pct false-positive rate must be no higher than real_only",
            "not_used_as_population_prevalence_estimate": True,
        },
        "statistics": {
            "paired_exact_test": "two-sided exact McNemar test on paired correctness",
            "bootstrap": "2,000 stratified paired bootstrap replicates with seed 20260822",
            "confidence_level": 0.95,
            "significance_level": 0.05,
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"],
        },
        "final_choice_rule": (
            "Choose gan_10pct as the final candidate if the primary augmentation support rule passes "
            "and its long-URL false-positive rate is no higher than real_only. Otherwise retain "
            "real_only or report the choice as inconclusive. Statistical significance and practical "
            "gate results are reported separately and cannot be manufactured by threshold changes."
        ),
        "deployment_claim": "The selected condition remains a deployment candidate until operational browser testing is completed.",
    }
    output.parent.mkdir(parents=True)
    write_json(output, payload)
    (output.parent / "README.md").write_text(
        "# Frozen final confirmation protocol\n\n"
        "This protocol locks the final model pair, thresholds, cohort sizes, statistical tests, "
        "and decision rule before either new cohort is scored.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
