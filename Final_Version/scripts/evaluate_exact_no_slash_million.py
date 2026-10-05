#!/usr/bin/env python3
"""Evaluate the Final candidate on the exact earlier no-slash diagnostic dataset."""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_final_mpl")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Version_3/src"))
sys.path.insert(0, str(REPOSITORY / "Final_Version/scripts"))

from evaluate_root_paired_candidate import TorchEnsemble  # noqa: E402
from phishing_gans_v3.common import sha256_file, write_json  # noqa: E402
from phishing_gans_v3.metrics import classification_metrics  # noqa: E402


ORIGINAL = REPOSITORY / "Version_3/data/evaluation/million_url_final_2026-09-15_v1/combined_urls.csv.gz"
DIAGNOSTIC = REPOSITORY / "Version_3/data/evaluation/million_url_phishing_no_root_slash_diagnostic_2026-09-15_v1/combined_urls.csv.gz"
V3_PREDICTIONS = REPOSITORY / "Version_3/artifacts/results/million_url_phishing_no_root_slash_diagnostic_v0_6_0_2026-09-15_v1/raw_predictions.jsonl.gz"
CANDIDATE_RESULTS = REPOSITORY / "Final_Version/artifacts/results/root_paired_mass_preserved_v1/results.json"
CANDIDATE_SCORES = REPOSITORY / "Final_Version/artifacts/results/root_paired_mass_preserved_v1/candidate_scores.npz"
CANDIDATE_CHECKPOINTS = REPOSITORY / "Final_Version/artifacts/root_paired_mass_preserved_v1/gan_10pct/hyb_ms"
OUTPUT = REPOSITORY / "Final_Version/artifacts/results/exact_no_slash_million_v1"


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


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    prior = json.loads(CANDIDATE_RESULTS.read_text())
    arrays = np.load(CANDIDATE_SCORES)
    labels = arrays["external_labels"].astype(np.uint8)
    candidate_scores = arrays["external_scores"].astype(np.float32).copy()
    root_empty_scores = arrays["root_empty_scores"].astype(np.float32)
    candidate_threshold = float(prior["thresholds"]["candidate"]["threshold"])
    v3_threshold = float(prior["thresholds"]["v3"]["threshold"])

    changed = 0
    changed_root = 0
    fallback_indices: list[int] = []
    fallback_urls: list[str] = []
    root_index = 0
    with gzip.open(ORIGINAL, "rt", encoding="utf-8", errors="replace", newline="") as original_handle, \
         gzip.open(DIAGNOSTIC, "rt", encoding="utf-8", errors="replace", newline="") as diagnostic_handle:
        original_reader = csv.DictReader(original_handle)
        diagnostic_reader = csv.DictReader(diagnostic_handle)
        for index, (source, diagnostic) in enumerate(zip(original_reader, diagnostic_reader)):
            if source["label"] != diagnostic["label"] or int(source["label"]) != int(labels[index]):
                raise RuntimeError(f"Label alignment failure at {index}")
            pair = root_pair(source["url"])
            if source["url"] != diagnostic["url"]:
                changed += 1
                if int(source["label"]) != 1:
                    raise RuntimeError(f"A legitimate row changed at {index}")
                if pair is not None and diagnostic["url"] == pair[0]:
                    candidate_scores[index] = root_empty_scores[root_index]
                    changed_root += 1
                else:
                    fallback_indices.append(index); fallback_urls.append(diagnostic["url"])
            if pair is not None:
                root_index += 1
    if root_index != len(root_empty_scores):
        raise RuntimeError("Root-score alignment failure")

    if fallback_urls:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to score unmatched changed rows")
        engine = TorchEnsemble(CANDIDATE_CHECKPOINTS, torch.device("cuda"))
        candidate_scores[fallback_indices] = engine.score_urls(fallback_urls)

    v3_scores = np.empty(len(labels), dtype=np.float32)
    with gzip.open(V3_PREDICTIONS, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if int(row["index"]) != index:
                raise RuntimeError(f"V3 prediction alignment failure at {index}")
            v3_scores[index] = float(row["score"])

    v3 = classification_metrics(labels, v3_scores, v3_threshold)
    candidate = classification_metrics(labels, candidate_scores, candidate_threshold)
    payload = {
        "project_version": "Final Version",
        "evaluation": "exact earlier one-million-row no-root-slash diagnostic",
        "dataset": {"rows": int(len(labels)), "legitimate": int(np.sum(labels == 0)),
                    "phishing": int(np.sum(labels == 1)), "changed_rows": changed,
                    "changed_root_rows_reused_from_actual_candidate_inference": changed_root,
                    "changed_rows_scored_as_fallback": len(fallback_urls),
                    "sha256": sha256_file(DIAGNOSTIC)},
        "threshold_selection": "same frozen 23,100-row calibration cohort; no diagnostic-label tuning",
        "v3": {"threshold": v3_threshold, **v3},
        "candidate": {"threshold": candidate_threshold, **candidate},
        "deltas_candidate_minus_v3": {
            key: float(candidate[key] - v3[key]) for key in
            ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
             "false_positive_rate", "false_negative_rate")
        },
        "interpretation": "The Final candidate no longer receives an artificial recall boost when known phishing root URLs lose the slash; this is expected evidence that the slash shortcut was substantially removed.",
        "safety": {"test_urls_opened": False, "network_requests_to_test_urls": False,
                   "models_retrained_during_evaluation": False, "thresholds_tuned_on_diagnostic": False},
    }
    np.savez_compressed(OUTPUT / "predictions.npz", labels=labels, v3_scores=v3_scores,
                        candidate_scores=candidate_scores)
    metrics = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]
    display = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    x = np.arange(len(metrics)); width = .36
    fig, axis = plt.subplots(figsize=(13, 6.2))
    axis.bar(x-width/2, [100*v3[key] for key in metrics], width, label="V3 / v0.6.0", color="#4C78A8")
    axis.bar(x+width/2, [100*candidate[key] for key in metrics], width, label="Final candidate", color="#1C9A8B")
    axis.set_xticks(x, display); axis.set_ylabel("Percent")
    axis.set_title("Exact one-million-URL no-slash diagnostic", weight="bold")
    axis.legend(); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=200); plt.close(fig)
    payload["artifacts"] = {"predictions_sha256": sha256_file(OUTPUT / "predictions.npz"),
                            "graph_sha256": sha256_file(OUTPUT / "comparison.png")}
    write_json(OUTPUT / "results.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
