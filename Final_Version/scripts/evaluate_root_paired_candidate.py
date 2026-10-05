#!/usr/bin/env python3
"""Compare the isolated root-paired candidate with the frozen v0.6.0 model."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_final_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Version_3/src"))

from phishing_gans_v3.analyze_threshold_sets import recall_threshold  # noqa: E402
from phishing_gans_v3.common import sha256_file, write_json  # noqa: E402
from phishing_gans_v3.evaluate_hybrid_external import create_model, score  # noqa: E402
from phishing_gans_v3.metrics import classification_metrics  # noqa: E402
from phishing_gans_v3.url_data import canonicalize_url, encode_url, psl_aware_lexical_features  # noqa: E402


V3_CHECKPOINTS = REPOSITORY / "Version_3/artifacts/experiments/recent_update_gan10_v1/gan_10pct/hyb_ms"
CANDIDATE_CHECKPOINTS = REPOSITORY / "Final_Version/artifacts/root_paired_hyb_ms_gan10_v1/gan_10pct/hyb_ms"
CALIBRATION = REPOSITORY / "Version_3/data/calibration/updated_models_2026-08-15_v1/calibration_urls.csv"
INTERNAL = REPOSITORY / "Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/processed"
EXTERNAL = REPOSITORY / "Version_3/data/evaluation/million_url_final_2026-09-15_v1/combined_urls.csv.gz"
V3_EXTERNAL_PREDICTIONS = REPOSITORY / "Version_3/artifacts/results/million_url_final_v0_6_0_2026-09-15_v1/raw_predictions.jsonl.gz"
V3_ROOT_RESULTS = REPOSITORY / "Version_3/artifacts/results/root_slash_invariance_v0_6_0_2026-09-22_v1/results.json"
OUTPUT = REPOSITORY / "Final_Version/artifacts/results/root_paired_candidate_v1"
SEEDS = (42, 43, 44)
BATCH_SIZE = 4096


def root_pair(raw_url: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    return (
        urlunsplit((parsed.scheme, parsed.netloc, "", "", "")),
        urlunsplit((parsed.scheme, parsed.netloc, "/", "", "")),
    )


class TorchEnsemble:
    def __init__(self, checkpoint_root: Path, device: torch.device):
        self.device = device
        self.models = []
        self.checkpoints = []
        for seed in SEEDS:
            path = checkpoint_root / f"seed_{seed}/checkpoint.pt"
            saved = torch.load(path, map_location=device, weights_only=False)
            model = create_model(saved).to(device)
            model.load_state_dict(saved["state_dict"]); model.eval()
            self.models.append(model); self.checkpoints.append(saved)
        first = self.checkpoints[0]
        self.vocabulary = first["vocabulary"]
        self.center = np.asarray(first["lexical_scaler_center"], dtype=np.float32)
        self.scale = np.where(np.asarray(first["lexical_scaler_scale"], dtype=np.float32) == 0, 1.0,
                              np.asarray(first["lexical_scaler_scale"], dtype=np.float32))
        for saved in self.checkpoints[1:]:
            if saved["vocabulary"] != self.vocabulary:
                raise RuntimeError("Candidate seeds use different vocabularies")

    def prepare(self, raw_urls: list[str]) -> tuple[np.ndarray, np.ndarray]:
        normalized = [canonicalize_url(value)[0] for value in raw_urls]
        tokens = np.stack([encode_url(value, self.vocabulary, 128) for value in normalized])
        raw = np.stack([psl_aware_lexical_features(value) for value in normalized]).astype(np.float32)
        lexical = ((raw - self.center) / self.scale).astype(np.float32)
        return tokens, lexical

    def score_prepared(self, tokens: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        values = [score(model, tokens, lexical, BATCH_SIZE, self.device) for model in self.models]
        return np.mean(np.stack(values), axis=0).astype(np.float32)

    def score_urls(self, raw_urls: list[str]) -> np.ndarray:
        tokens, lexical = self.prepare(raw_urls)
        return self.score_prepared(tokens, lexical)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def score_array_dataset(engine: TorchEnsemble, split: str) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.load(INTERNAL / f"{split}_tokens.npy", mmap_mode="r")
    lexical = np.load(INTERNAL / f"{split}_psl_lexical.npy", mmap_mode="r")
    labels = np.load(INTERNAL / f"{split}_labels.npy").astype(np.uint8)
    return labels, engine.score_prepared(tokens, lexical)


def percentile(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "maximum": float(np.max(values))}


def paired_summary(labels: np.ndarray, left: np.ndarray, right: np.ndarray, threshold: float) -> dict:
    delta = np.abs(left - right)
    a = left >= threshold; b = right >= threshold
    result = {"pairs": int(len(labels)), "absolute_score_delta": percentile(delta),
              "class_flips": int(np.sum(a != b)), "class_flip_rate": float(np.mean(a != b)),
              "empty_root_metrics": classification_metrics(labels, left, threshold),
              "slash_root_metrics": classification_metrics(labels, right, threshold)}
    for name, mask in (("legitimate", labels == 0), ("phishing", labels == 1)):
        result[name] = {"pairs": int(np.sum(mask)), "absolute_score_delta": percentile(delta[mask]),
                        "class_flips": int(np.sum(a[mask] != b[mask])),
                        "class_flip_rate": float(np.mean(a[mask] != b[mask]))}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoints", type=Path, default=CANDIDATE_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def plot_comparison(payload: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = ["#4C78A8", "#1C9A8B"]
    labels = ["V3 / v0.6.0", "Final candidate"]
    internal = [payload["internal"]["v3"], payload["internal"]["candidate"]]
    for key, title, axis in (("recall", "Internal recall", axes[0]),
                             ("f1", "Internal F1", axes[1])):
        axis.bar(labels, [100 * row[key] for row in internal], color=colors)
        axis.set_ylim(0, 100); axis.set_ylabel("Percent"); axis.set_title(title, weight="bold")
        axis.grid(axis="y", alpha=.2)
        for i, row in enumerate(internal): axis.text(i, 100 * row[key] + 1, f"{100*row[key]:.2f}%", ha="center")
    root = payload["root_robustness"]
    values = [100 * root["v3"]["class_flip_rate"], 100 * root["candidate"]["class_flip_rate"]]
    axes[2].bar(labels, values, color=colors); axes[2].set_ylabel("Percent")
    axes[2].set_title("Root slash class-flip rate", weight="bold"); axes[2].grid(axis="y", alpha=.2)
    for i, value in enumerate(values): axes[2].text(i, value + 1, f"{value:.2f}%", ha="center")
    fig.suptitle("Version 3 vs root-paired Final candidate", fontsize=17, weight="bold")
    fig.tight_layout(); fig.savefig(output / "internal_and_root_robustness.png", dpi=200); plt.close(fig)

    ext = payload["external_million"]
    metrics = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]
    display = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    x = np.arange(len(metrics)); width = .36
    fig, axis = plt.subplots(figsize=(13, 6.2))
    axis.bar(x-width/2, [100*ext["v3"][m] for m in metrics], width, label=labels[0], color=colors[0])
    axis.bar(x+width/2, [100*ext["candidate"][m] for m in metrics], width, label=labels[1], color=colors[1])
    axis.set_xticks(x, display); axis.set_ylabel("Percent"); axis.set_title("Million-URL external evaluation", weight="bold")
    axis.legend(); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(output / "external_million_comparison.png", dpi=200); plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Frozen evaluation output exists: {output}")
    output.mkdir(parents=True)
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Final Version evaluation")
    device = torch.device("cuda")
    candidate = TorchEnsemble(args.candidate_checkpoints, device)

    calibration_rows = read_csv_rows(CALIBRATION)
    calibration_labels = np.asarray([int(row["label"]) for row in calibration_rows], dtype=np.uint8)
    candidate_calibration_scores = candidate.score_urls([row["url"] for row in calibration_rows])
    candidate_operating_point = recall_threshold(calibration_labels, candidate_calibration_scores, .01)
    candidate_threshold = float(candidate_operating_point["threshold"])
    v3_calibration = np.load(REPOSITORY / "Version_3/browser_extension_explainable_v0_6_0/tests/browser_onnx_calibration_predictions.npz")
    v3_operating_point = recall_threshold(v3_calibration["labels"].astype(np.uint8),
                                          v3_calibration["scores"].astype(np.float32), .01)
    v3_threshold = float(v3_operating_point["threshold"])

    internal_labels, candidate_internal_scores = score_array_dataset(candidate, "test")
    v3_internal_seed_scores = []
    for seed in SEEDS:
        values = np.load(V3_CHECKPOINTS / f"seed_{seed}/test_predictions.npz")
        v3_internal_seed_scores.append(values["scores"])
    v3_internal_scores = np.mean(np.stack(v3_internal_seed_scores), axis=0).astype(np.float32)

    external_labels = np.empty(1_000_000, dtype=np.uint8)
    candidate_external_scores = np.empty(1_000_000, dtype=np.float32)
    candidate_root_empty: list[np.ndarray] = []
    candidate_root_slash: list[np.ndarray] = []
    root_labels: list[np.ndarray] = []
    with gzip.open(EXTERNAL, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle); batch_urls: list[str] = []; batch_labels: list[int] = []; offset = 0
        for row in reader:
            batch_urls.append(row["url"]); batch_labels.append(int(row["label"]))
            if len(batch_urls) < BATCH_SIZE:
                continue
            scores = candidate.score_urls(batch_urls); end = offset + len(batch_urls)
            candidate_external_scores[offset:end] = scores; external_labels[offset:end] = batch_labels
            pairs = [(i, root_pair(url)) for i, url in enumerate(batch_urls)]
            pairs = [(i, pair) for i, pair in pairs if pair is not None]
            if pairs:
                candidate_root_slash.append(scores[[i for i, _ in pairs]])
                candidate_root_empty.append(candidate.score_urls([pair[0] for _, pair in pairs]))
                root_labels.append(np.asarray([batch_labels[i] for i, _ in pairs], dtype=np.uint8))
            offset = end; batch_urls = []; batch_labels = []
            if offset % 40960 == 0:
                print(json.dumps({"stage": "external", "scored": offset, "total": 1_000_000}), flush=True)
        if batch_urls:
            scores = candidate.score_urls(batch_urls); end = offset + len(batch_urls)
            candidate_external_scores[offset:end] = scores; external_labels[offset:end] = batch_labels
            pairs = [(i, root_pair(url)) for i, url in enumerate(batch_urls)]
            pairs = [(i, pair) for i, pair in pairs if pair is not None]
            if pairs:
                candidate_root_slash.append(scores[[i for i, _ in pairs]])
                candidate_root_empty.append(candidate.score_urls([pair[0] for _, pair in pairs]))
                root_labels.append(np.asarray([batch_labels[i] for i, _ in pairs], dtype=np.uint8))
            offset = end
    if offset != 1_000_000:
        raise RuntimeError(f"Expected 1,000,000 external rows, found {offset}")

    v3_external_scores = np.empty(1_000_000, dtype=np.float32)
    with gzip.open(V3_EXTERNAL_PREDICTIONS, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line); v3_external_scores[index] = float(row["score"])

    root_labels_array = np.concatenate(root_labels)
    candidate_empty = np.concatenate(candidate_root_empty)
    candidate_slash = np.concatenate(candidate_root_slash)
    v3_root = json.loads(V3_ROOT_RESULTS.read_text())["raw_model_class_flips"]["all"]
    payload = {
        "project_version": "Final Version",
        "experiment": "root_paired_hyb_ms_gan10_v1",
        "controlled_change": "paired empty-path and slash representations for every root-only training row",
        "unchanged": ["source URL identities", "labels", "non-root training rows", "validation", "domain validation",
                      "internal test", "external million-URL set", "HYB-MS architecture", "MS-HardGAN-C pool",
                      "10% weighted GAN exposure", "seeds 42/43/44", "threshold-selection cohort and rule"],
        "thresholds": {
            "selection": "same 23,100-row calibration cohort; maximize recall subject to FPR <= 1%",
            "v3": v3_operating_point, "candidate": candidate_operating_point,
        },
        "internal": {
            "rows": int(len(internal_labels)),
            "v3": classification_metrics(internal_labels, v3_internal_scores, v3_threshold),
            "candidate": classification_metrics(internal_labels, candidate_internal_scores, candidate_threshold),
        },
        "external_million": {
            "rows": int(len(external_labels)), "legitimate": int(np.sum(external_labels == 0)),
            "phishing": int(np.sum(external_labels == 1)),
            "v3": classification_metrics(external_labels, v3_external_scores, v3_threshold),
            "candidate": classification_metrics(external_labels, candidate_external_scores, candidate_threshold),
        },
        "root_robustness": {
            "pairs": int(len(root_labels_array)),
            "v3": {"pairs": int(v3_root["pairs"]), "class_flips": int(v3_root["flips"]),
                   "class_flip_rate": float(v3_root["flip_rate"])},
            "candidate": paired_summary(root_labels_array, candidate_empty, candidate_slash, candidate_threshold),
        },
        "artifacts": {
            "candidate_checkpoints": {str(seed): sha256_file(args.candidate_checkpoints / f"seed_{seed}/checkpoint.pt") for seed in SEEDS},
            "calibration_dataset_sha256": sha256_file(CALIBRATION), "external_dataset_sha256": sha256_file(EXTERNAL),
        },
        "safety": {"test_urls_opened": False, "network_requests_to_test_urls": False,
                   "version_3_modified": False, "extension_modified": False, "test_threshold_tuning": False},
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(output / "candidate_scores.npz", calibration_labels=calibration_labels,
                        calibration_scores=candidate_calibration_scores, internal_labels=internal_labels,
                        internal_scores=candidate_internal_scores, external_labels=external_labels,
                        external_scores=candidate_external_scores, root_labels=root_labels_array,
                        root_empty_scores=candidate_empty, root_slash_scores=candidate_slash)
    plot_comparison(payload, output)
    payload["artifacts"]["candidate_scores_sha256"] = sha256_file(output / "candidate_scores.npz")
    payload["artifacts"]["internal_root_plot_sha256"] = sha256_file(output / "internal_and_root_robustness.png")
    payload["artifacts"]["external_plot_sha256"] = sha256_file(output / "external_million_comparison.png")
    write_json(output / "results.json", payload)
    print(json.dumps({"thresholds": payload["thresholds"], "internal": payload["internal"],
                      "external_million": payload["external_million"],
                      "root_robustness": payload["root_robustness"]}, indent=2))
    del candidate; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
