from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


MODELS = ("seqgan", "urlgan", "relgan")
LABELS = {"seqgan": "SG", "urlgan": "URL-GAN", "relgan": "RelGAN"}


def main() -> None:
    root = Path("artifacts/results/urlgan_experiment")
    root.mkdir(parents=True, exist_ok=True)
    payload = {"project_version": 3, "plain_names": {
        "SG": "original one-layer LSTM SeqGAN",
        "URL-GAN": "new two-layer LSTM + Gumbel-Softmax generator with character-CNN discriminator",
        "RelGAN": "relational-memory text GAN",
        "G": "generator", "D": "discriminator", "Char-CNN-D": "character-CNN discriminator",
    }, "models": {}}
    for model in MODELS:
        audit = load_json(Path("data/generated") / model / "audit.json")
        quality = load_json(Path("artifacts/results/gan_quality") / model / "quality.json")
        payload["models"][model] = {
            "label": LABELS[model], "accepted_rows": audit["accepted_rows"],
            "acceptance_rate": audit["accepted_fraction"], "raw_validity": audit["raw_quality"]["validity"],
            "raw_uniqueness": audit["raw_quality"]["uniqueness"],
            "source_auc": quality["source_classifier_roc_auc"]["mean"],
            "lexical_smd": quality["lexical_standardized_mean_difference"]["mean_absolute"],
            "cluster_coverage": quality["real_cluster_coverage"], "cluster_js": quality["cluster_js_distance"],
            "majority_challenge": quality["real_only_classifier_challenge"]["majority_challenging_rate"],
        }
    payload["conclusion"] = (
        "URL-GAN is stable and usable, clearly better than RelGAN, but the original compact SeqGAN remains the best "
        "augmentation generator because it has higher filter yield and closer real-phishing distributions."
    )
    write_json(root / "comparison.json", payload)

    metrics = ("acceptance_rate", "raw_validity", "raw_uniqueness", "cluster_coverage", "majority_challenge")
    titles = ("Filter acceptance ↑", "Raw validity ↑", "Raw uniqueness ↑", "Real cluster coverage ↑", "Classifier challenge ↑")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))
    colors = ("#4C78A8", "#2A9D8F", "#E15759")
    for axis, metric, title in zip(axes, metrics, titles):
        values = [100 * payload["models"][model][metric] for model in MODELS]
        bars = axis.bar([LABELS[model] for model in MODELS], values, color=colors)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
        axis.set_title(title, fontweight="bold"); axis.set_ylim(0, 105); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Phishing URL generator comparison — 500,000-sample audits", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(root / "gan_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(root / "gan_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# URL-GAN Experiment", "", "## Simple names", "",
             "- `URL-GAN`: the complete new GAN.", "- `2L-LSTM-G`: its two-layer LSTM generator.",
             "- `Char-CNN-D`: its character-CNN discriminator.", "- `GS`: Gumbel-Softmax relaxation.",
             "- `SG`: the original compact SeqGAN used as the reference.", "", "## Fixed training contract", "",
             "All generators used the same 82,219 phishing URLs from the original frozen training split, the same 168-token "
             "character vocabulary, maximum length 128, batch size 64, seed 42, and a 500,000-sample final generation budget. "
             "No legitimate, validation, test, calibration, or external URL was used to train the generators.", "",
             "## Results", "", "| GAN | Accepted | Accept. | Validity | Uniqueness | Source AUC ↓ | Lex. SMD ↓ | Clust. cov. ↑ | Challenge ↑ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        v = payload["models"][model]
        lines.append(f"| {v['label']} | {v['accepted_rows']:,} | {v['acceptance_rate']:.4f} | {v['raw_validity']:.4f} | "
                     f"{v['raw_uniqueness']:.4f} | {v['source_auc']:.4f} | {v['lexical_smd']:.4f} | "
                     f"{v['cluster_coverage']:.4f} | {v['majority_challenge']:.4f} |")
    lines.extend(("", "Source AUC closer to 0.5 is better; lower lexical SMD is better. Challenge is neither automatically good "
                  "nor bad—it measures how often most frozen real-only classifiers miss a generated phishing URL.", "", "## Conclusion", "", payload["conclusion"]))
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
