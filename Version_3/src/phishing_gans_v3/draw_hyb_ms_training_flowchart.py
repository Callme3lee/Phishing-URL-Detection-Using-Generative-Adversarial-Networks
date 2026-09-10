from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#1D4ED8"
NAVY = "#15396C"
CYAN = "#0891B2"
GREEN = "#15803D"
AMBER = "#B45309"
PURPLE = "#7E22CE"
RED = "#B91C1C"
INK = "#172033"
LIGHT = "#F8FAFC"


def box(axis, xy, width, height, title, body, color, title_size=12, body_size=9.4):
    x, y = xy
    shadow = FancyBboxPatch(
        (x + 0.10, y - 0.10), width, height,
        boxstyle="round,pad=0.16,rounding_size=0.18", linewidth=0,
        facecolor="#CBD5E1", alpha=0.55, zorder=1,
    )
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.16,rounding_size=0.18", linewidth=2.1,
        edgecolor=color, facecolor="white", zorder=2,
    )
    axis.add_patch(shadow)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height - 0.30, title, ha="center", va="top",
              fontsize=title_size, fontweight="bold", color=color, zorder=3)
    axis.text(x + width / 2, y + height - 0.80, body, ha="center", va="top",
              fontsize=body_size, color=INK, linespacing=1.32, zorder=3)
    return patch


def arrow(axis, start, end, color="#64748B", width=2.2, label=None, bend=0.0):
    item = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16, linewidth=width,
        color=color, connectionstyle=f"arc3,rad={bend}", shrinkA=3, shrinkB=3, zorder=4,
    )
    axis.add_patch(item)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        axis.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=8.4,
                  fontweight="bold", color=color,
                  bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"}, zorder=5)


def main() -> None:
    output = Path("artifacts/reports/hyb_ms_training_flowchart")
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(21, 12), facecolor=LIGHT)
    axis.set_xlim(0, 21)
    axis.set_ylim(0, 12)
    axis.axis("off")

    fig.suptitle("HYB-MS Training and Prediction Flow", fontsize=25, fontweight="bold", color=NAVY, y=0.975)
    axis.text(
        10.5, 11.38,
        "Character CNN + 35 PSL-aware lexical features + source balancing + class-conditional CORAL",
        ha="center", fontsize=13, color="#475569",
    )

    # Stage 1: data and sampling.
    box(axis, (0.35, 7.55), 3.45, 3.15, "1  FROZEN REAL DATA",
        "363,692 unique URLs\n\nLegitimate\n276,400 URLs_New\n3,012 outside coverage\n\nPhishing\n82,219 URLs_New\n2,000 PhishTank\n61 OpenPhish\n\nGAN rows: 0", BLUE, body_size=8.7)
    box(axis, (4.45, 8.05), 3.35, 2.25, "2  SOURCE-BALANCED SAMPLING",
        "One 363,692-draw epoch\nwith replacement\n\nTarget exposure:\n35% / 15% legitimate\n30% / 15% / 5% phishing", CYAN, body_size=9.0)
    arrow(axis, (3.82, 9.12), (4.42, 9.12), BLUE)

    # Stage 2: dual inputs.
    box(axis, (8.55, 9.05), 3.45, 1.60, "3A  CHARACTER INPUT",
        "Normalize and remove scheme\nEncode up to 128 characters", PURPLE)
    box(axis, (8.55, 6.95), 3.45, 1.60, "3B  LEXICAL INPUT",
        "Calculate 35 numeric PSL-aware\nfeatures; robust-scale them", AMBER)
    arrow(axis, (7.82, 9.32), (8.52, 9.72), CYAN, bend=-0.10)
    arrow(axis, (7.82, 8.88), (8.52, 7.75), CYAN, bend=0.10)

    # Stage 3: branches and fusion.
    box(axis, (12.75, 9.05), 3.55, 1.60, "4A  CHARACTER CNN",
        "48-D embedding → Conv1D 3/4/5\n128 channels each → max pool\nOutput: 384 values", PURPLE)
    box(axis, (12.75, 6.95), 3.55, 1.60, "4B  LEXICAL NETWORK",
        "35 → 64 → ReLU + dropout\n64 → 32 → ReLU\nOutput: 32 values", AMBER)
    arrow(axis, (12.02, 9.85), (12.72, 9.85), PURPLE)
    arrow(axis, (12.02, 7.75), (12.72, 7.75), AMBER)
    box(axis, (17.15, 7.85), 3.45, 2.30, "5  FEATURE FUSION",
        "Concatenate: 384 + 32 = 416\n\n416 → 128 → ReLU\nDropout 0.4 → 1 logit\nSigmoid → phishing score", GREEN)
    arrow(axis, (16.32, 9.85), (17.12, 9.45), PURPLE, bend=0.10)
    arrow(axis, (16.32, 7.75), (17.12, 8.48), AMBER, bend=-0.10)

    # Losses and update loop.
    box(axis, (12.15, 3.75), 4.10, 2.15, "6  TRAINING OBJECTIVE",
        "Binary cross-entropy classification\n+\n0.01 × class-conditional CORAL\nAlign sources within legitimate class\nand within phishing class", RED)
    box(axis, (7.25, 3.75), 3.75, 2.15, "7  GPU OPTIMIZATION",
        "AdamW + mixed precision\nBackpropagate through both branches\nBatch 512; gradient clip 1.0\nMaximum 8 epochs; patience: 2", BLUE)
    arrow(axis, (18.87, 7.82), (16.28, 5.15), GREEN, label="scores + 128-D representation", bend=0.05)
    arrow(axis, (12.12, 4.82), (11.03, 4.82), RED, label="total loss")

    # Validation and selection.
    box(axis, (2.10, 3.60), 3.85, 2.45, "8  MODEL SELECTION",
        "Original validation: 39,483 URLs\nDomain validation: 709 URLs\n\nScore = 50% original PR-AUC\n+ 50% domain PR-AUC\n\nBest epoch retained", GREEN, body_size=8.9)
    arrow(axis, (7.22, 4.82), (5.98, 4.82), BLUE)

    # Ensemble and calibration.
    box(axis, (1.00, 0.55), 4.15, 1.95, "9  THREE TRAINING SEEDS",
        "Train identical architecture independently\nSeed 42  |  Seed 43  |  Seed 44\nEach selected checkpoint: epoch 1", PURPLE)
    arrow(axis, (3.90, 3.57), (3.15, 2.53), GREEN)
    box(axis, (6.15, 0.55), 3.75, 1.95, "10  ENSEMBLE",
        "For every URL, average the three\nseed-specific phishing probabilities", CYAN)
    arrow(axis, (5.18, 1.52), (6.12, 1.52), PURPLE)
    box(axis, (10.90, 0.55), 4.05, 1.95, "11  THRESHOLD CALIBRATION",
        "Separate 400-URL calibration set\nMaximize F1 subject to FPR ≤ 1%\nFrozen threshold: 0.987955749", AMBER)
    arrow(axis, (9.93, 1.52), (10.87, 1.52), CYAN)
    box(axis, (15.95, 0.55), 4.25, 1.95, "12  FINAL DECISION",
        "Mean score ≥ threshold → PHISHING\nMean score < threshold → LEGITIMATE\n\nExternal 10K: F1 91.20% | FPR 1.70%", GREEN)
    arrow(axis, (14.98, 1.52), (15.92, 1.52), AMBER)

    # Phase labels.
    axis.text(0.35, 11.05, "TRAINING PHASE", color=BLUE, fontsize=10.5, fontweight="bold")
    axis.plot([0.35, 20.55], [6.55, 6.55], color="#CBD5E1", linewidth=1.5, linestyle="--")
    axis.text(0.35, 6.72, "LEARNING AND VALIDATION", color=RED, fontsize=10.5, fontweight="bold")
    axis.plot([0.35, 20.55], [3.08, 3.08], color="#CBD5E1", linewidth=1.5, linestyle="--")
    axis.text(0.35, 3.23, "FINAL ENSEMBLE AND PREDICTION", color=GREEN, fontsize=10.5, fontweight="bold")

    axis.text(
        20.55, 0.12,
        "Important: HYB-MS is the 0% GAN control. Future GAN samples enter as a separate phishing source group.",
        ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=NAVY,
    )
    fig.subplots_adjust(left=0.018, right=0.982, top=0.92, bottom=0.035)
    fig.savefig(output / "hyb_ms_training_flowchart.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output / "hyb_ms_training_flowchart.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
