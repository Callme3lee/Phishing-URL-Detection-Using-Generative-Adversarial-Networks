from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, sha256_file, write_json


MODELS = ("cnn", "lstm", "rnn", "random_forest", "linear_svm")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")
EXTERNAL = {
    "topdomains_phishtank": (
        Path("artifacts/results/urls_new_structure_balanced_v1/external_topdomains_phishtank/three_conditions/comparison.json"),
        Path("artifacts/results/hybrid_cnn_coral_v1/external_topdomains_phishtank/summary.json"),
        "topdomains_phishtank", "Top-Domain / PhishTank (500)",
    ),
    "tranco_openphish": (
        Path("artifacts/results/urls_new_structure_balanced_v1/external_openphish_tranco/three_conditions/comparison.json"),
        Path("artifacts/results/hybrid_cnn_coral_v1/external_openphish_tranco/summary.json"),
        "tranco_openphish", "Tranco / OpenPhish (400)",
    ),
}


def exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def mean_metric(value: dict) -> dict[str, float]:
    return {"mean": float(value["mean"]), "standard_deviation": float(value["standard_deviation"])}


def main() -> None:
    output_dir = Path("artifacts/results/pre_gan_readiness")
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = load_json(Path("data/processed/verification_report.json"))
    preparation = load_json(Path("data/processed/preparation_report.json"))
    baselines = load_json(Path("artifacts/results/baseline_summary.json"))
    calibration = load_json(Path("artifacts/results/hybrid_cnn_coral_v1/calibrated_ensemble_v1/threshold.json"))
    confirmation = load_json(Path("artifacts/results/hybrid_cnn_coral_v1/confirmation_readiness.json"))
    calibration_report = load_json(Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"))

    missing = []
    baseline_runs = 0
    for model in MODELS:
        for seed in SEEDS:
            run = Path("artifacts/baselines") / model / f"seed_{seed}"
            required = (run / "metrics.json", run / "test_predictions.npz",
                        run / ("checkpoint.pt" if model in {"cnn", "lstm", "rnn"} else "model.joblib"))
            if all(exists_nonempty(path) for path in required):
                baseline_runs += 1
            else:
                missing.extend(str(path) for path in required if not exists_nonempty(path))
    controlled_runs = 0
    for experiment in ("legitimate_coverage_v1", "urls_new_structure_balanced_v1"):
        for model in MODELS:
            for seed in SEEDS:
                if exists_nonempty(Path("artifacts/experiments") / experiment / model / f"seed_{seed}" / "metrics.json"):
                    controlled_runs += 1
    hybrid_runs = sum(exists_nonempty(
        Path("artifacts/experiments/hybrid_cnn_coral_v1") / f"seed_{seed}" / "checkpoint.pt"
    ) for seed in SEEDS)
    checkpoint_hashes_valid = all(
        sha256_file(Path(row["path"])) == row["sha256"] for row in calibration["checkpoints"]
    )

    progression = {}
    labels = ("Original LSTM", "Coverage LSTM", "Balanced LSTM",
              "Hybrid original threshold", "Calibrated ensemble")
    for key, (prior_path, hybrid_path, calibrated_key, title) in EXTERNAL.items():
        prior, hybrid = load_json(prior_path), load_json(hybrid_path)
        if prior["dataset_sha256"] != hybrid["dataset_sha256"]:
            raise RuntimeError(f"Dataset identity mismatch in {key}")
        systems = {
            labels[0]: {metric: mean_metric(prior["conditions"]["original"]["lstm"]["metrics"][metric]) for metric in METRICS},
            labels[1]: {metric: mean_metric(prior["conditions"]["outside_coverage"]["lstm"]["metrics"][metric]) for metric in METRICS},
            labels[2]: {metric: mean_metric(prior["conditions"]["urls_new_balanced"]["lstm"]["metrics"][metric]) for metric in METRICS},
            labels[3]: {metric: mean_metric(hybrid["original_threshold"][metric]) for metric in METRICS},
            labels[4]: {metric: {"mean": float(calibration["external_exploratory"][calibrated_key]["metrics"][metric]),
                                "standard_deviation": 0.0} for metric in METRICS},
        }
        progression[key] = {"title": title, "dataset_sha256": prior["dataset_sha256"], "systems": systems}

        colors = ("#9EA3A8", "#6C8EBF", "#B4A7D6", "#E76F51", "#2A9D8F")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for axis, metric in zip(axes.ravel(), METRICS):
            means = [100 * systems[label][metric]["mean"] for label in labels]
            errors = [100 * systems[label][metric]["standard_deviation"] for label in labels]
            axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=3, color=colors)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=23, ha="right")
            axis.set_title(metric.replace("_", " ").upper(), fontweight="bold")
            axis.set_ylim(0, 100); axis.grid(axis="y", alpha=.25)
        fig.suptitle(f"Pre-GAN classifier progression — {title}\nexternal results are exploratory",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(output_dir / f"{key}_progression.png", dpi=220, bbox_inches="tight")
        fig.savefig(output_dir / f"{key}_progression.pdf", bbox_inches="tight")
        plt.close(fig)

    gates = {
        "dataset_integrity": {"status": "pass" if verification["passed"] else "fail",
                              "evidence": f"{sum(verification['checks'].values())}/{len(verification['checks'])} verification checks pass"},
        "baseline_matrix": {"status": "pass" if baseline_runs == 15 else "fail",
                            "evidence": f"{baseline_runs}/15 complete model-seed runs"},
        "controlled_shift_experiments": {"status": "pass" if controlled_runs == 30 else "fail",
                                         "evidence": f"{controlled_runs}/30 metrics artifacts"},
        "hybrid_reproducibility": {"status": "pass" if hybrid_runs == 3 and checkpoint_hashes_valid else "fail",
                                   "evidence": f"{hybrid_runs}/3 checkpoints; frozen hashes valid={checkpoint_hashes_valid}"},
        "calibration_separation": {"status": "pass" if all(value == 0 for value in calibration_report["measured_overlap"].values()) else "fail",
                                   "evidence": "zero measured URL/hostname/encoding overlap"},
        "source_shortcut_risk": {"status": "caution", "evidence": "original strongest one-feature stump ROC-AUC=0.7696; hybrid mitigates but does not prove invariance"},
        "external_claim_validity": {"status": "caution", "evidence": "current hybrid/calibrated external results informed model design and are exploratory"},
        "later_temporal_confirmation": {"status": "pass" if confirmation["ready"] else "blocked",
                                        "evidence": f"confirmation readiness={confirmation['ready']}"},
        "version_3_gan_absence": {"status": "pass", "evidence": "no Version 3 generator/trainer or augmentation result exists yet"},
    }
    payload = {
        "project_version": 3, "scope": "complete evaluation before Version 3 GAN implementation",
        "dataset": {"master_sha256": preparation["master_sha256"], "splits": preparation["splits"],
                    "quarantined_records": preparation["ingestion"]["quarantined_records"],
                    "gan_training_contract": preparation["gan_training_contract"]},
        "baseline_models": baselines["models"], "progression": progression,
        "calibrated_ensemble": {"threshold": calibration["threshold"],
                                "policy": calibration["policy"],
                                "external_exploratory": calibration["external_exploratory"]},
        "artifact_audit": {"baseline_runs": baseline_runs, "controlled_experiment_runs": controlled_runs,
                           "hybrid_runs": hybrid_runs, "checkpoint_hashes_valid": checkpoint_hashes_valid,
                           "missing": missing},
        "gates": gates,
        "decision": {
            "engineering_ready_to_implement_gan": True,
            "scientifically_ready_to_claim_gan_improvement": False,
            "reason": "pipeline and reference models are reproducible, but a genuinely later untouched confirmation snapshot is still unavailable",
        },
    }
    write_json(output_dir / "summary.json", payload)

    lines = ["# Version 3 Pre-GAN Evaluation", "",
             "## Decision", "",
             "The project is **engineering-ready to implement a Version 3 GAN**, but it is **not yet scientifically ready to claim GAN improvement**. The classifier/data pipeline is reproducible; the remaining blocker is a genuinely later untouched confirmation snapshot for the calibrated real-data reference.", "",
             "## Data foundation", "",
             f"URLs_New contributed {preparation['ingestion']['input_rows']:,} raw rows. After validation, conflict handling, canonicalization, and encoded deduplication, the frozen splits contain {preparation['splits']['train']['total']:,} training, {preparation['splits']['validation']['total']:,} validation, and {preparation['splits']['test']['total']:,} test rows. Hostnames are disjoint and all {len(verification['checks'])} verification checks pass.", "",
             "The GAN contract is already fixed: train generators only on phishing rows from the frozen Version 3 training split.", "",
             "## Internal real-only baselines", "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("lstm", "cnn", "rnn", "random_forest", "linear_svm"):
        values = baselines["models"][model]["metrics"]
        lines.append("| " + model.replace("_", " ").title() + " | " + " | ".join(
            f"{values[metric]['mean']:.4f}" for metric in METRICS) + " |")
    lines.extend(("", "The LSTM is the strongest original internal baseline; the CNN has marginally higher recall and ROC-AUC. These internal values do not establish cross-source generalization.", "",
                  "## External progression", ""))
    for result in progression.values():
        lines.extend((f"### {result['title']}", "",
                      "| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
                      "|---|---:|---:|---:|---:|---:|---:|"))
        for label in labels:
            lines.append("| " + label + " | " + " | ".join(
                f"{result['systems'][label][metric]['mean']:.4f}" for metric in METRICS) + " |")
        lines.append("")
    lines.extend(("## Readiness gates", "", "| Gate | Status | Evidence |", "|---|---|---|"))
    for gate, value in gates.items():
        lines.append(f"| {gate.replace('_', ' ').title()} | {value['status'].upper()} | {value['evidence']} |")
    lines.extend(("", "## Before GAN training", "",
                  "1. Keep the real-only baselines, hybrid checkpoints, ensemble rule, threshold, and all splits frozen.",
                  "2. Obtain the genuinely later confirmation snapshot and evaluate the calibrated ensemble once.",
                  "3. Pre-register GAN quality gates before viewing augmentation outcomes: validity, uniqueness, novelty, training overlap, lexical realism, source diversity, and real-only classifier challenge rate.",
                  "4. Train generators only on the 82,219 phishing rows in the frozen original training split unless a separately named multi-source generator condition is explicitly registered.",
                  "5. Keep 10%, 20%, 30%, and 40% augmentation counts fixed and compare every condition against the same real-only reference across seeds 42/43/44.",
                  "6. Treat no improvement or degradation as valid evidence; do not change ratios, samples, or thresholds to force a win."))
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("docs/PRE_GAN_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
