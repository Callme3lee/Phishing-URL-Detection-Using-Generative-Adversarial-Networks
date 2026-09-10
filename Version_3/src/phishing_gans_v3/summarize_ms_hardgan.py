from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


def main() -> None:
    v1_audit = load_json(Path("data/generated/ms_hardgan_v1/audit.json"))
    v2_audit = load_json(Path("data/generated/ms_hardgan_v2_final/audit.json"))
    v1_quality = load_json(Path("artifacts/results/ms_hardgan_v1/quality/quality.json"))
    v2_quality = load_json(Path("artifacts/results/ms_hardgan_v2/quality/quality.json"))
    output = Path("artifacts/results/ms_hardgan_comparison")
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "strict_acceptance": (v1_audit["accepted_fraction"], v2_audit["accepted_fraction"]),
        "pattern_agreement": (
            v1_audit["category_agreement"]["requested_pattern_agreement"],
            v2_audit["category_agreement"]["requested_pattern_agreement"],
        ),
        "source_agreement": (
            v1_audit["category_agreement"]["requested_source_agreement"],
            v2_audit["category_agreement"]["requested_source_agreement"],
        ),
        "cluster_coverage": (v1_quality["real_cluster_coverage"], v2_quality["real_cluster_coverage"]),
        "majority_challenge": (
            v1_quality["frozen_ms_classifier_challenge"]["majority_challenging_rate"],
            v2_quality["frozen_ms_classifier_challenge"]["majority_challenging_rate"],
        ),
        "at_least_one_challenge": (
            v1_quality["frozen_ms_classifier_challenge"]["challenging_to_at_least_one_rate"],
            v2_quality["frozen_ms_classifier_challenge"]["challenging_to_at_least_one_rate"],
        ),
    }
    report = {
        "conditions": ["MS-HardGAN v1", "MS-HardGAN-C"], "metrics": metrics,
        "selection": "MS-HardGAN-C is preferred because categorical control and challenge improve without overlap or collapse",
        "limitation": "source agreement is dominated by URLs_New; source-classifier AUC shows residual synthetic shift",
    }
    write_json(output / "comparison.json", report)
    labels = ("Strict\naccept", "Pattern\nagree", "Source\nagree", "Cluster\ncoverage",
              "Majority\nchallenge", "Any-model\nchallenge")
    v1 = [values[0] for values in metrics.values()]
    v2 = [values[1] for values in metrics.values()]
    x = np.arange(len(labels)); width = .36
    fig, axis = plt.subplots(figsize=(12, 6))
    first = axis.bar(x - width / 2, v1, width, label="MS-HardGAN v1", color="#A0A0A0")
    second = axis.bar(x + width / 2, v2, width, label="MS-HardGAN-C", color="#4C78A8")
    axis.bar_label(first, fmt="%.3f", padding=2, fontsize=8)
    axis.bar_label(second, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(x, labels); axis.set_ylim(0, 1.08); axis.set_ylabel("Fraction")
    axis.set_title("Multi-source defensive URL GAN: category-control correction", fontweight="bold")
    axis.grid(axis="y", alpha=.25); axis.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=220)
    fig.savefig(output / "comparison.pdf"); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
