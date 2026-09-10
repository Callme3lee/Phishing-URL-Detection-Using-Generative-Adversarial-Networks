from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model, score
from .metrics import classification_metrics, select_threshold
from .url_data import encode_url, psl_aware_lexical_features


CONDITIONS = ("real_only", "gan_10pct", "gan_20pct", "gan_30pct", "gan_40pct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate each HYB-MS ensemble identically, then evaluate frozen sets.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--real-root", type=Path, default=Path("artifacts/experiments/hybrid_cnn_coral_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments/fair_hybrid_gan_v1"))
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv"))
    parser.add_argument("--calibration-report", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"))
    parser.add_argument("--external", type=Path, default=Path("data/external/2026-08-12-large-frozen/external_urls.csv"))
    parser.add_argument("--external-report", type=Path, default=Path("data/external/2026-08-12-large-frozen/collection_report.json"))
    parser.add_argument("--challenge", type=Path, default=Path("data/generated/ms_hardgan_v2_final/challenge_holdout.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/fair_hybrid_gan_v1/calibrated_ensembles.json"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def frozen_csv(path: Path, report_path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    report = load_json(report_path)
    if sha256_file(path) != report["dataset_sha256"]: raise RuntimeError(f"Hash mismatch: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != int(report["rows"]): raise RuntimeError(f"Row mismatch: {path}")
    return rows, labels, report


def checkpoint_root(args: argparse.Namespace, condition: str) -> Path:
    return args.real_root if condition == "real_only" else args.checkpoint_root / condition / "hyb_ms"


def ensemble(urls: list[str], root: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, list[dict]]:
    raw = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    outputs = []; identities = []
    for seed in (42, 43, 44):
        path = root / f"seed_{seed}" / "checkpoint.pt"
        saved = torch.load(path, map_location=device, weights_only=False)
        tokens = np.stack([encode_url(url, saved["vocabulary"], 128) for url in urls])
        center = np.asarray(saved["lexical_scaler_center"], dtype=np.float32)
        scale_values = np.asarray(saved["lexical_scaler_scale"], dtype=np.float32)
        scale_values = np.where(scale_values == 0, 1.0, scale_values)
        lexical = ((raw - center) / scale_values).astype(np.float32)
        model = create_model(saved).to(device); model.load_state_dict(saved["state_dict"])
        outputs.append(score(model, tokens, lexical, batch_size, device))
        identities.append({"seed": seed, "checkpoint": str(path), "sha256": sha256_file(path)})
        del model; gc.collect(); torch.cuda.empty_cache()
    return np.mean(np.stack(outputs), axis=0), identities


def main() -> None:
    args = parse_args(); device = require_cuda()
    calibration_rows, calibration_labels, calibration_report = frozen_csv(args.calibration, args.calibration_report)
    external_rows, external_labels, external_report = frozen_csv(args.external, args.external_report)
    with args.challenge.open("r", encoding="utf-8", newline="") as handle: challenge_rows = list(csv.DictReader(handle))
    internal_labels = np.load(args.data_root / "real_only" / "processed" / "test_labels.npy")
    with (args.data_root / "real_only" / "processed" / "test_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        internal_rows = list(csv.DictReader(handle))
    datasets = {
        "calibration": ([row["url"] for row in calibration_rows], calibration_labels),
        "internal_test": ([row["url"] for row in internal_rows], internal_labels),
        "external_10k": ([row["url"] for row in external_rows], external_labels),
        "synthetic_challenge": ([row["url"] for row in challenge_rows], None),
    }
    results = {}
    for condition in CONDITIONS:
        root = checkpoint_root(args, condition); scores = {}; identities = None
        for name, (urls, _) in datasets.items():
            scores[name], current = ensemble(urls, root, device, args.batch_size)
            if identities is None: identities = current
            elif identities != current: raise RuntimeError("Checkpoint identity changed")
        selected = select_threshold(calibration_labels, scores["calibration"], maximum_fpr=0.01)
        threshold = float(selected["threshold"])
        challenge_recall = float((scores["synthetic_challenge"] >= threshold).mean())
        results[condition] = {
            "threshold": threshold, "threshold_selection": selected,
            "threshold_policy": "three-seed mean score; maximum calibration F1 subject to calibration FPR <= 1%",
            "calibration_resubstitution": classification_metrics(calibration_labels, scores["calibration"], threshold),
            "internal_test": classification_metrics(internal_labels, scores["internal_test"], threshold),
            "external_10k": classification_metrics(external_labels, scores["external_10k"], threshold),
            "synthetic_challenge": {"rows": len(challenge_rows), "recall": challenge_recall, "false_negative_rate": 1.0 - challenge_recall},
            "checkpoints": identities,
        }
        print(json.dumps({"condition": condition, "threshold": threshold,
                          "internal_f1": results[condition]["internal_test"]["f1"],
                          "external_f1": results[condition]["external_10k"]["f1"],
                          "external_fpr": results[condition]["external_10k"]["false_positive_rate"],
                          "challenge_recall": challenge_recall}), flush=True)
    report = {"project_version": 3, "protocol": "identical separately calibrated three-seed HYB-MS ensembles",
              "calibration_dataset": str(args.calibration), "calibration_sha256": calibration_report["dataset_sha256"],
              "calibration_warning": "resubstitution metrics are optimistic and not test results",
              "external_dataset": str(args.external), "external_sha256": external_report["dataset_sha256"],
              "external_rows": len(external_rows), "challenge_dataset": str(args.challenge),
              "challenge_sha256": sha256_file(args.challenge), "results": results,
              "deployment_guard": "Strong balanced external metrics make a deployment candidate only; temporal, prevalence, latency and operational gates remain."}
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "conditions": len(results)}, indent=2))


if __name__ == "__main__":
    main()
