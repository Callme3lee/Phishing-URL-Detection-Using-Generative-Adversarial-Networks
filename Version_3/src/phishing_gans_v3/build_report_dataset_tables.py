from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .common import sha256_file, write_json


OUTPUT = Path("artifacts/results/report_dataset_tables_2026-09-01_v1")
DATASETS = (
    ("v3_original_train", "Original V3 training", Path("data/processed/train_manifest.csv"), 276_400, 82_219),
    ("v3_original_validation", "Original V3 validation", Path("data/processed/validation_manifest.csv"), 29_535, 9_948),
    ("v3_original_test", "Original V3 internal test", Path("data/processed/test_manifest.csv"), 38_908, 10_321),
    ("multisource_real_train", "Multi-source real training", Path("data/experiments/hybrid_cnn_coral_v1/processed/train_manifest.csv"), 279_412, 84_280),
    ("final_real_only_train", "Final updated real-only training", Path("data/experiments/recent_update_gan10_v1/real_only/processed/train_manifest.csv"), 281_158, 86_026),
    ("final_gan10_train", "Final 10% GAN training", Path("data/experiments/recent_update_gan10_v1/gan_10pct/processed/train_manifest.csv"), 281_158, 126_436),
    ("policy_development", "Browser policy development", Path("Final Evaluation Dataset/policy_development.csv"), 3_800, 200),
    ("frozen_final_6k", "Frozen final 6K evaluation", Path("Final Evaluation Dataset/frozen_final_test.csv"), 5_700, 300),
    ("postfreeze_10k", "Post-freeze 10K evaluation", Path("data/evaluation/postfreeze_10k_3pct_2026-09-01/evaluation_urls.csv"), 9_700, 300),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int32)
    return {
        "rows": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": int(np.max(array)),
        "over_128": int(np.sum(array > 128)),
        "over_128_pct": float(np.mean(array > 128)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    dataset_rows = []
    length_rows = []
    payload = {"datasets": {}}
    for key, name, path, expected_legitimate, expected_phishing in DATASETS:
        rows = read_rows(path)
        legitimate = [row for row in rows if int(row["label"]) == 0]
        phishing = [row for row in rows if int(row["label"]) == 1]
        if len(legitimate) != expected_legitimate or len(phishing) != expected_phishing:
            raise RuntimeError(f"Class-count mismatch for {path}")
        dataset_rows.append({
            "key": key, "dataset": name, "role": (
                "training" if "train" in key else
                "validation/development" if "validation" in key or "development" in key else
                "evaluation"
            ),
            "legitimate": len(legitimate), "phishing": len(phishing),
            "total": len(rows), "phishing_pct": 100 * len(phishing) / len(rows),
            "sha256": sha256_file(path), "path": str(path),
        })
        class_summaries = {}
        for label, label_name, selected in ((0, "Legitimate", legitimate), (1, "Phishing", phishing)):
            summary = summarize_lengths([len(row["url"]) for row in selected])
            class_summaries[label_name.lower()] = summary
            length_rows.append({"key": key, "dataset": name, "class": label_name, **summary})
        payload["datasets"][key] = {
            **dataset_rows[-1], "lengths": class_summaries,
        }

    with (OUTPUT / "dataset_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0]))
        writer.writeheader(); writer.writerows(dataset_rows)
    with (OUTPUT / "url_length_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_rows[0]))
        writer.writeheader(); writer.writerows(length_rows)
    write_json(OUTPUT / "results.json", payload)
    print(json.dumps({"output": str(OUTPUT), "datasets": dataset_rows}, indent=2))


if __name__ == "__main__":
    main()
