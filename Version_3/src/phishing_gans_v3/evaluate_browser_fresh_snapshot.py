from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the exact packaged browser ONNX ensemble and frozen tier policy."
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
        "--browser-root", type=Path, default=Path("browser_extension_user_pilot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/browser_extension_fresh_raw_2026-08-28_5pct"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def score_ensemble(
    urls: list[str], browser_root: Path, batch_size: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    model_root = browser_root / "models"
    vocabulary = load_json(model_root / "vocabulary.json")
    scaler = load_json(model_root / "scaler.json")
    manifest = load_json(model_root / "bundle_manifest.json")

    normalized = [canonicalize_url(url)[0] for url in urls]
    tokens = np.stack([encode_url(url, vocabulary, 128) for url in normalized]).astype(np.int64)
    raw_features = np.stack(
        [psl_aware_lexical_features(url) for url in normalized]
    ).astype(np.float32)
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    lexical = ((raw_features - center) / np.where(scale == 0, 1.0, scale)).astype(np.float32)

    seed_scores: list[np.ndarray] = []
    verified_models: list[dict[str, object]] = []
    for item in manifest["models"]:
        model_path = model_root / item["filename"]
        actual_hash = sha256_file(model_path)
        if actual_hash != item["onnx_sha256"]:
            raise RuntimeError(f"Browser model hash mismatch: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        chunks = []
        for start in range(0, len(urls), batch_size):
            chunks.append(session.run(
                ["probability"],
                {
                    "tokens": tokens[start:start + batch_size],
                    "lexical": lexical[start:start + batch_size],
                },
            )[0].astype(np.float32))
        seed_scores.append(np.concatenate(chunks))
        verified_models.append({
            "seed": item["seed"],
            "filename": item["filename"],
            "sha256": actual_hash,
            "provider": "CPUExecutionProvider",
        })
    return np.mean(np.stack(seed_scores), axis=0).astype(np.float32), verified_models


def tier_for_score(score: float, tiers: list[dict[str, object]]) -> str:
    for tier in tiers:
        lower = tier["lower_inclusive"]
        upper = tier["upper_exclusive"]
        if (lower is None or score >= float(lower)) and (upper is None or score < float(upper)):
            return str(tier["name"])
    raise RuntimeError(f"No policy tier for score {score}")


def evaluate_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    matrix = metrics["confusion_matrix"]
    metrics["per_10000_urls"] = {
        "false_alerts": 10_000 * matrix["fp"] / len(labels),
        "detected_phishing": 10_000 * matrix["tp"] / len(labels),
        "missed_phishing": 10_000 * matrix["fn"] / len(labels),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Browser snapshot result already exists at {args.output}")

    collection = load_json(args.collection_report)
    if sha256_file(args.dataset) != collection["dataset_sha256"]:
        raise RuntimeError("Frozen snapshot hash does not match the collection report")
    if not collection.get("selection_locked_before_inference"):
        raise RuntimeError("Snapshot was not declared frozen before inference")

    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    urls = [row["url"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if len(rows) != collection["rows"] or int(labels.sum()) != collection["phishing_rows"]:
        raise RuntimeError("Snapshot row counts differ from the collection report")

    scores, verified_models = score_ensemble(urls, args.browser_root, args.batch_size)
    model_root = args.browser_root / "models"
    policy_path = model_root / "policy.json"
    bundle_path = model_root / "bundle_manifest.json"
    policy = load_json(policy_path)
    tiers = policy["tiers"]
    tier_names = [str(tier["name"]) for tier in tiers]
    assigned_tiers = np.asarray([tier_for_score(float(score), tiers) for score in scores])

    tier_rows = []
    for tier_name in tier_names:
        mask = assigned_tiers == tier_name
        legitimate = int(np.sum(mask & (labels == 0)))
        phishing = int(np.sum(mask & (labels == 1)))
        tier_rows.append({
            "tier": tier_name,
            "legitimate": legitimate,
            "phishing": phishing,
            "total": legitimate + phishing,
        })

    threshold_results = {}
    for tier in tiers:
        if tier["lower_inclusive"] is None:
            continue
        threshold_results[f"{tier['name']}_or_higher"] = {
            "threshold": float(tier["lower_inclusive"]),
            "metrics": evaluate_at_threshold(labels, scores, float(tier["lower_inclusive"])),
        }

    primary_threshold = float(policy["primary_binary_rule"]["threshold"])
    payload = {
        "project_version": 3,
        "evaluation": "exact packaged browser ONNX ensemble and frozen tier policy",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": collection["dataset_sha256"],
            "rows": len(rows),
            "legitimate_rows": int(np.sum(labels == 0)),
            "phishing_rows": int(np.sum(labels == 1)),
            "phishing_prevalence": float(labels.mean()),
            "measured_overlap": collection.get(
                "measured_overlap", collection.get("identity_contract")
            ),
        },
        "browser_bundle": {
            "root": str(args.browser_root),
            "manifest": str(bundle_path),
            "manifest_sha256": sha256_file(bundle_path),
            "policy": str(policy_path),
            "policy_sha256": sha256_file(policy_path),
            "policy_version": policy["policy_version"],
            "model": policy["model"],
            "verified_models": verified_models,
            "inference_runtime": "ONNX Runtime CPU, matching the packaged ONNX graphs; browser WebAssembly parity is separately tested",
        },
        "primary_warning_or_block": {
            "threshold": primary_threshold,
            "metrics": evaluate_at_threshold(labels, scores, primary_threshold),
        },
        "tier_counts": tier_rows,
        "threshold_results": threshold_results,
        "score_summary": {
            "minimum": float(np.min(scores)),
            "median": float(np.median(scores)),
            "maximum": float(np.max(scores)),
        },
        "interpretation": {
            "warning_or_block_is_the_binary_phishing_decision": True,
            "caution_is_not_a_strict_false_positive": True,
            "block_is_the_only_normal_mode_interruption": True,
            "no_training_or_threshold_change": True,
            "limitation": "Feed labels and Tranco root proxies are not manual page-level adjudication; this is an offline external audit, not production certification.",
        },
    }

    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "browser_predictions.npz",
        labels=labels,
        scores=scores,
        tiers=assigned_tiers,
    )
    payload["predictions_sha256"] = sha256_file(args.output / "browser_predictions.npz")
    write_json(args.output / "browser_evaluation.json", payload)
    with (args.output / "tier_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tier", "legitimate", "phishing", "total"])
        writer.writeheader()
        writer.writerows(tier_rows)

    primary = payload["primary_warning_or_block"]["metrics"]
    print(json.dumps({
        "primary_warning_or_block": primary,
        "tier_counts": tier_rows,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
