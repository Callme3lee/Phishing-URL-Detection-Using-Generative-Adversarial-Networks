from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json
from .metrics import classification_metrics
from .url_data import canonicalize_url, encode_url, psl_aware_lexical_features


BROWSER = Path("browser_extension")
OUTPUT = Path("artifacts/results/browser_v0_2_policy_diagnostic")
DATASETS = {
    "primary_30k": Path("data/external/2026-08-22-final-confirmation-5pct/external_urls.csv"),
    "long_url_stress": Path("data/external/2026-08-22-final-long-url-stress/evaluation_urls.csv"),
}


def score_dataset(path: Path, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    urls = [canonicalize_url(row["url"])[0] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    vocabulary = load_json(BROWSER / "models/vocabulary.json")
    scaler = load_json(BROWSER / "models/scaler.json")
    tokens = np.stack([encode_url(url, vocabulary, 128) for url in urls]).astype(np.int64)
    raw = np.stack([psl_aware_lexical_features(url) for url in urls]).astype(np.float32)
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    lexical = ((raw - center) / np.where(scale == 0, 1, scale)).astype(np.float32)
    manifest = load_json(BROWSER / "models/bundle_manifest.json")
    seed_scores = []
    for model in manifest["models"]:
        model_path = BROWSER / "models" / model["filename"]
        if sha256_file(model_path) != model["onnx_sha256"]:
            raise RuntimeError(f"Browser model hash mismatch: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        chunks = []
        for start in range(0, len(rows), batch_size):
            chunks.append(session.run(["probability"], {
                "tokens": tokens[start:start + batch_size],
                "lexical": lexical[start:start + batch_size],
            })[0].astype(np.float32))
        seed_scores.append(np.concatenate(chunks))
    return labels, np.mean(np.stack(seed_scores), axis=0).astype(np.float32)


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    predictions = scores >= threshold
    return {
        "threshold": threshold,
        "metrics": metrics,
        "legitimate_interruptions": int(np.sum((labels == 0) & predictions)),
        "phishing_interruptions": int(np.sum((labels == 1) & predictions)),
        "legitimate_rows": int(np.sum(labels == 0)),
        "phishing_rows": int(np.sum(labels == 1)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic already exists at {OUTPUT}")
    policy = load_json(BROWSER / "models/policy.json")
    warning_threshold = float(policy["primary_binary_rule"]["threshold"])
    block = next(tier for tier in policy["tiers"] if tier["name"] == "block")
    block_threshold = float(block["lower_inclusive"])
    results = {}
    for name, path in DATASETS.items():
        labels, scores = score_dataset(path)
        old = evaluate(labels, scores, warning_threshold)
        new = evaluate(labels, scores, block_threshold)
        results[name] = {
            "dataset": str(path), "dataset_sha256": sha256_file(path),
            "old_v0_1_behavior_warning_and_block_interrupt": old,
            "new_v0_2_behavior_only_block_interrupts": new,
            "change": {
                "legitimate_interruptions": new["legitimate_interruptions"] - old["legitimate_interruptions"],
                "phishing_interruptions": new["phishing_interruptions"] - old["phishing_interruptions"],
            },
        }
        print(json.dumps({"dataset": name, "old": old, "new": new}, indent=2), flush=True)
    payload = {
        "project_version": 3,
        "extension_version": "0.2.0",
        "model_bundle": str(BROWSER / "models/bundle_manifest.json"),
        "model_bundle_sha256": sha256_file(BROWSER / "models/bundle_manifest.json"),
        "policy": str(BROWSER / "models/policy.json"),
        "policy_sha256": sha256_file(BROWSER / "models/policy.json"),
        "model_and_score_thresholds_unchanged": True,
        "change_under_test": "warning becomes passive; only the existing strict block tier interrupts",
        "results": results,
        "interpretation_limit": (
            "This is a post-hoc deployment-policy diagnostic on already evaluated datasets, not a new model-validation claim. "
            "It measures interruption burden, not whether passive warnings disappear."
        ),
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "results.json", payload)
    with PdfPages(OUTPUT / "browser_v0_2_policy_diagnostic.pdf") as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "Browser v0.2 false-interruption policy", fontsize=24, weight="bold", color="#17324D")
        fig.text(.055, .85, "Same URL model and thresholds • warning becomes passive • only strict block interrupts", fontsize=12, color="#3B6EA8")
        columns = ["Cohort", "Legitimate", "v0.1 false blocks", "v0.2 false blocks", "Avoided", "v0.1 phish blocks", "v0.2 phish blocks"]
        rows = []
        labels = {"primary_30k": "Primary 30K", "long_url_stress": "Long-URL stress"}
        for name, result in results.items():
            old = result["old_v0_1_behavior_warning_and_block_interrupt"]
            new = result["new_v0_2_behavior_only_block_interrupts"]
            rows.append([labels[name], f"{old['legitimate_rows']:,}", f"{old['legitimate_interruptions']:,}",
                         f"{new['legitimate_interruptions']:,}", f"{old['legitimate_interruptions']-new['legitimate_interruptions']:,}",
                         f"{old['phishing_interruptions']:,}", f"{new['phishing_interruptions']:,}"])
        axis = fig.add_axes([.04, .53, .92, .23]); axis.axis("off")
        table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                           colWidths=[.18, .11, .15, .15, .10, .15, .15])
        table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#CAD5E1")
            if row == 0: cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
        fig.text(.055, .37, "What changed", fontsize=17, weight="bold", color="#17324D")
        fig.text(.07, .30, "• High-risk warning badges remain visible, but they no longer replace the requested page.", fontsize=11.5, color="#526678")
        fig.text(.07, .24, "• Only the already calibrated strictest tier opens an interstitial.", fontsize=11.5, color="#526678")
        fig.text(.07, .18, "• Exact-host trust is explicit, local, reversible, and never applies automatically to subdomains.", fontsize=11.5, color="#526678")
        fig.text(.055, .09, "Diagnostic only: these previously evaluated datasets were not used to change model scores or thresholds.", fontsize=10.5, color="#B45309", weight="bold")
        pdf.savefig(fig); plt.close(fig)
    print(json.dumps({"output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
