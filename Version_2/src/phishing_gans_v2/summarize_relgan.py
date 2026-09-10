from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


def best_record(path: Path, adversarial_only: bool = False) -> dict:
    records = load_json(path)
    eligible = [row for row in records if not adversarial_only or row.get("stage") == "adversarial"]
    return max(eligible, key=lambda row: row["selection_score"])


def main() -> None:
    generator_root = Path("artifacts/generators")
    output = Path("artifacts/results/generators/checkpoint_audits"); output.mkdir(parents=True, exist_ok=True)
    records = {
        "gumbel_lsgan": best_record(generator_root / "gumbel_lsgan" / "history.json"),
        "seqgan": best_record(generator_root / "seqgan" / "history.json"),
        "relgan": best_record(generator_root / "relgan" / "history.json", adversarial_only=True),
    }
    rows = []
    for model, record in records.items():
        quality = record["quality"]
        rows.append({"model": model, "epoch": record["epoch"], "selection_score": record["selection_score"],
                     "uniqueness": quality["uniqueness"], "validity": quality["validity"],
                     "memorization_rate": quality["memorization_rate"],
                     "repeated_template_rate": quality["repeated_template_rate"],
                     "samples": quality["samples"]})
    payload = {"project_version": 2, "protocol": "selected-checkpoint 2000-sample audits",
               "warning": "RelGAN has no full two-million-sample filtered corpus yet", "generators": rows}
    write_json(output / "summary.json", payload)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)

    names = [row["model"].replace("_", " ").title() for row in rows]
    metrics = ("uniqueness", "validity", "memorization_rate", "repeated_template_rate", "selection_score")
    colors = ("#2b6cb0", "#dd6b20", "#2f855a")
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.5, 4.5))
    for axis, metric in zip(axes, metrics):
        values = [100 * row[metric] for row in rows]
        axis.bar(names, values, color=colors); axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("Percent"); axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=.25)
        if metric in ("uniqueness", "validity", "selection_score"):
            axis.set_ylim(max(0, min(values) - 1), min(100, max(values) + .25))
    fig.suptitle("Selected generator checkpoint audits — 2,000 samples each", fontweight="bold")
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "comparison.pdf", bbox_inches="tight"); plt.close(fig)

    history = load_json(generator_root / "relgan" / "history.json")
    pretrain = [row for row in history if row["stage"] == "pretraining"]
    adversarial = [row for row in history if row["stage"] == "adversarial"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot([row["epoch"] for row in pretrain], [row["mle_loss_mean"] for row in pretrain],
                 marker="o", color="#2b6cb0")
    axes[0].set_title("RelGAN MLE warm-up"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean MLE loss")
    axes[1].plot([row["epoch"] for row in adversarial], [100 * row["selection_score"] for row in adversarial],
                 marker="o", color="#2f855a", label="Selection score")
    axes[1].plot([row["epoch"] for row in adversarial], [100 * row["quality"]["validity"] for row in adversarial],
                 marker="s", color="#805ad5", label="Validity")
    axes[1].set_title("RelGAN adversarial quality"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Percent")
    axes[1].legend();
    for axis in axes: axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "relgan_training.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "relgan_training.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
