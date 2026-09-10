from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt

from .common import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/generators"))
    args = parser.parse_args()
    results = [load_json(path) for path in sorted(args.input_dir.glob("*/metrics.json"))]
    if not results:
        raise RuntimeError("No completed generator corpus metrics found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"project_version": 2, "generators": {item["model"]: item for item in results}}
    write_json(args.output_dir / "summary.json", summary)
    fields = ("model", "raw_uniqueness", "accepted_uniqueness", "raw_validity", "acceptance_rate", "memorization_rate", "repeated_template_rate")
    rows = [{
        "model": item["model"], "raw_uniqueness": item["raw"]["uniqueness"],
        "accepted_uniqueness": item["accepted"]["uniqueness"],
        "raw_validity": item["raw"]["validity"], "acceptance_rate": item["acceptance_rate"],
        "memorization_rate": item["raw"]["memorization_rate"],
        "repeated_template_rate": item["raw"]["repeated_template_rate"],
    } for item in results]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    metrics = ("raw_uniqueness", "raw_validity", "acceptance_rate", "memorization_rate", "repeated_template_rate")
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5))
    names = [row["model"].replace("_", " ").title() for row in rows]
    for axis, metric in zip(axes, metrics):
        values = [100 * row[metric] for row in rows]
        axis.bar(names, values, color=("#2b6cb0", "#2f855a", "#dd6b20")[:len(rows)])
        axis.set_title(metric.replace("_", " ").title()); axis.set_ylabel("Percent")
        axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Version 2 generator quality — same malicious training split")
    fig.tight_layout(); fig.savefig(args.output_dir / "comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.output_dir / "comparison.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
