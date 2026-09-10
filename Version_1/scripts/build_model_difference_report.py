"""Build a landscape PDF summarizing why two GAN-augmentation results differ."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np


NAVY = "#123047"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#C94C4C"
GRAY = "#566573"
LIGHT = "#F3F6F8"
WHITE = "#FFFFFF"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--other-root", type=Path,
        required=True,
        help="Path to the independently reproduced comparison project",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/model_result_difference_analysis.pdf"),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canvas(title: str, subtitle: str, page: int):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.055, 0.945, title, fontsize=22, fontweight="bold", color=NAVY, va="top")
    ax.text(0.055, 0.902, subtitle, fontsize=10.5, color=GRAY, va="top")
    ax.plot([0.055, 0.945], [0.875, 0.875], color=BLUE, lw=2)
    ax.text(0.055, 0.025, "Phishing URL Detection with GAN Augmentation", fontsize=8.5, color=GRAY)
    ax.text(0.945, 0.025, f"Page {page} of 8", fontsize=8.5, color=GRAY, ha="right")
    return fig, ax


def box(ax, x, y, w, h, title, body, color=BLUE, body_size=10.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                           facecolor=LIGHT, edgecolor="#D5DEE3", linewidth=1)
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch((x, y+h-0.052), w, 0.052,
                                boxstyle="round,pad=0.012,rounding_size=0.012",
                                facecolor=color, edgecolor=color))
    ax.text(x+0.018, y+h-0.026, title, color=WHITE, fontsize=11.5,
            fontweight="bold", va="center")
    wrapped = "\n".join(textwrap.fill(p, width=max(28, int(w*105))) for p in body.split("\n"))
    ax.text(x+0.018, y+h-0.075, wrapped, color=NAVY, fontsize=body_size,
            va="top", linespacing=1.35)


def table(ax, x, y, w, h, columns, rows, widths=None, font=8.8, header_color=NAVY):
    tax = ax.inset_axes([x, y, w, h]); tax.axis("off")
    tbl = tax.table(cellText=rows, colLabels=columns, colWidths=widths,
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8E0E5"); cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(header_color); cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F9FA")
    return tbl


def pct(value: float) -> str:
    return f"{100*value:.4f}%"


def main() -> None:
    cfg = args()
    other = cfg.other_root
    project = cfg.project_root
    prep = load(other / "data/processed/preparation_report.json")
    cross = load(other / "artifacts/augmentation_experiments/cross_model_comparison.json")
    modern = load(project / "artifacts/immediate_external_evaluation_2026-08-01/summary/summary.json")
    cnn = next(row for row in cross["comparisons"] if row["model"] == "cnn")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(cfg.output) as pdf:
        fig, ax = canvas("Why the 40% Result Looks Different", "Evidence-based comparison of the two GAN-augmentation projects • 2 August 2026", 1)
        box(ax, .055, .61, .42, .22, "Main answer",
            "The results are not contradictory. The newer report measures an easier same-source test; this project’s final comparison measures current-feed distribution shift. Absolute scores should not be compared as if the test sets were equivalent.", TEAL, 12)
        box(ax, .525, .61, .42, .22, "What the 40% result proves",
            "WGAN-GP 40% is the highest observed in-domain CNN condition. Its F1 gain is only +0.0187 percentage points and its paired 95% interval [-0.0089, +0.0464] crosses zero. It is not yet a supported augmentation improvement.", ORANGE, 12)
        box(ax, .055, .32, .42, .22, "New project",
            "52,946 in-domain test URLs\nReal-only CNN F1: 99.4188%\nWGAN-GP 40% CNN F1: 99.4375%\nExternal dataset: unopened", BLUE, 11.5)
        box(ax, .525, .32, .42, .22, "This project’s final test",
            "600 independently assembled current-feed URLs\nReal-only CNN F1: 62.28%\nModern LSGAN 10% CNN F1: 68.05%\nFrozen external evaluation: completed", TEAL, 11.5)
        ax.text(.055, .23, "Bottom line", fontsize=14, fontweight="bold", color=NAVY)
        ax.text(.055, .19, textwrap.fill(
            "The 99.4% score comes mainly from test-set familiarity and an already saturated real-only baseline—not from a large demonstrated GAN effect.",
            120,
        ), fontsize=12, color=NAVY, va="top", linespacing=1.3)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("Dataset and Sample-Count Differences", "The two training/evaluation systems are not dose-only variants", 2)
        rows = [
            ["Retained development URLs", "543,384", "Modern final corpus: 49,358", "Different sources and scope"],
            ["Real training rows", "432,662", "39,402", "About 11× more in new run"],
            ["Generated rows", "288,441", "4,378", "66× more synthetic rows"],
            ["Final training rows", "721,103", "43,780", "Not directly comparable"],
            ["Evaluation rows", "52,946", "600", "Different precision/uncertainty"],
            ["Evaluation type", "Same-source hostname holdout", "Independent current feed", "Major difficulty difference"],
            ["External status", "Not loaded", "Evaluated once", "Different evidence level"],
        ]
        table(ax, .055, .44, .89, .36, ["Item", "New WGAN 40%", "Selected modern LSGAN 10%", "Interpretation"], rows, [.24,.21,.25,.30], 9)
        box(ax, .055, .16, .89, .20, "Meaning of the percentages",
            "In the new experiment, 40% means 288,441 generated malicious URLs added to all 432,662 real training rows. In the modern experiment, 10% means 4,378 generated URLs added to a balanced 39,402-row real corpus. Percentage alone cannot explain which condition generalizes better.", BLUE, 11)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("Evaluation Protocol Is the Main Difference", "Same-source generalization and external/temporal generalization answer different questions", 3)
        box(ax, .055, .57, .42, .26, "New report: in-domain",
            "Both development sources are divided into train, validation, and test by hostname. This prevents exact-host leakage but preserves source style, collection artifacts, label conventions, and time period across all partitions. The test asks whether the model generalizes to new hostnames from familiar sources.", BLUE, 11)
        box(ax, .525, .57, .42, .26, "This project: shifted distribution",
            "The final test uses current PhishTank/PhishStats phishing URLs and unused Tranco controls after excluding 2,149,042 protected hostname hashes. Thresholds and checkpoints were frozen. The test asks whether the model survives a new feed and harder benign domains.", TEAL, 11)
        ax.annotate("Higher absolute metrics expected", xy=(.27,.49), xytext=(.27,.40), ha="center", fontsize=12, color=BLUE,
                    arrowprops=dict(arrowstyle="->", lw=2, color=BLUE))
        ax.annotate("Lower metrics reveal deployment gap", xy=(.73,.49), xytext=(.73,.40), ha="center", fontsize=12, color=TEAL,
                    arrowprops=dict(arrowstyle="->", lw=2, color=TEAL))
        box(ax, .14, .13, .72, .18, "Critical reused source",
            "The new project’s URL dataset.csv is byte-for-byte identical to the file this project previously used as an external source (matching SHA-256). Promoting it into training naturally removes much of the earlier cross-source failure.", ORANGE, 11.5)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("Source and Label Structure in the New Test", "Hostname separation is correct, but source imbalance can still simplify pooled evaluation", 4)
        counts = np.array([[31671,10451],[41,10783]])
        bax = ax.inset_axes([.07,.42,.5,.37])
        x=np.arange(2); bax.bar(x, counts[:,0], color=BLUE, label="Legitimate"); bax.bar(x, counts[:,1], bottom=counts[:,0], color=ORANGE, label="Malicious")
        bax.set_xticks(x, ["Main source", "Additional source"]); bax.set_ylabel("Frozen test URLs"); bax.legend(frameon=False); bax.grid(axis="y",alpha=.2)
        box(ax, .62, .50, .32, .29, "Key audit finding",
            "The additional test source contains 10,783 malicious URLs but only 41 legitimate URLs: 99.62% malicious. A model can exploit source-specific cues even when no hostname repeats.", RED, 11)
        box(ax, .055, .13, .89, .20, "Pooled ROC-AUC diagnostic",
            "The saved real-only CNN predictions give 99.9255% pooled ROC-AUC, but approximately 94.23% when ROC-AUC is computed separately per source and averaged. The additional source averages only about 88.52% ROC-AUC and has just 41 negative examples. The pooled number is valid for this exact mixture, but it is not evidence of unseen-source robustness.", ORANGE, 11)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("What ‘40% Works’ Means Numerically", "Three-seed means on the new project’s frozen in-domain test", 5)
        metrics=["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","FPR"]
        real=[cnn[f"real_only_{m}_mean"] for m in ["accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate"]]
        aug=[cnn[f"augmented_{m}_mean"] for m in ["accuracy","precision","recall","f1","roc_auc","pr_auc","false_positive_rate"]]
        changes=(np.array(aug)-np.array(real))*100
        rows=[[m,pct(r),pct(a),f"{d:+.4f} pp"] for m,r,a,d in zip(metrics,real,aug,changes)]
        table(ax,.055,.43,.56,.37,["Metric","Real only","WGAN-GP 40%","Change"],rows,[.28,.23,.26,.23],9.5)
        cax=ax.inset_axes([.67,.43,.27,.37]); colors=[TEAL if v>0 else RED for v in changes[:-1]]
        positions = np.arange(len(metrics) - 1)
        cax.barh(positions, changes[:-1], color=colors)
        cax.set_yticks(positions, metrics[:-1]); cax.invert_yaxis()
        cax.axvline(0,color=NAVY,lw=1); cax.set_xlabel("Change (percentage points)"); cax.grid(axis="x",alpha=.2); cax.tick_params(labelsize=8)
        box(ax,.055,.14,.89,.20,"Statistical reading",
            "The F1 gain is +0.0187 pp, while recall and ROC-AUC are slightly lower. The F1 confidence interval crosses zero, as do the other paired CNN intervals. The accuracy change is roughly eight fewer errors per 52,946-URL run. This is a near-tie, not a large augmentation effect.", ORANGE,11.2)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("Model and Training Differences", "Architecture contributes, but it is secondary to data distribution", 6)
        rows=[
            ["Vocabulary", "191", "115", "New sources contain more characters"],
            ["Embedding", "48 dimensions", "32 dimensions", "New CNN is larger"],
            ["Convolution channels", "128 per kernel", "64 per kernel", "About double width"],
            ["Kernel sizes", "3, 4, 5", "3, 5, 7", "Different receptive fields"],
            ["Dropout", "0.50", "0.30", "Different regularization"],
            ["Training", "Up to 8 epochs", "Normally 5 epochs", "More optimization in new run"],
            ["Loss balancing", "Dynamic positive weight", "Dataset-dependent", "Controls synthetic class skew"],
        ]
        table(ax,.055,.42,.89,.38,["Component","New project CNN","Original project CNN","Effect"],rows,[.23,.22,.22,.33],9.3)
        box(ax,.055,.14,.89,.20,"Important control",
            "At 40% augmentation the new training set is 61.3% malicious, versus 40.1% malicious in its test set. The implementation compensates with BCE positive weight = negatives/positives, so the result is not simply caused by predicting everything as malicious. The model and training code are reasonable.", TEAL,11.2)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("How the Augmentation Findings Relate", "Different GANs win under different source and evaluation conditions", 7)
        rows=[
            ["New in-domain", "WGAN-GP 40%", "+0.0187 pp", "3", "CI crosses zero"],
            ["Original limited-real", "LSGAN 40%", "+0.1285 pp", "1", "Exploratory"],
            ["Original limited-real", "WGAN-GP 40%", "-0.1545 pp", "1", "Harmful in that setup"],
            ["Source-balanced", "Quality LSGAN 30%", "+0.1493 pp", "3", "Positive paired CI"],
            ["Modern temporal", "Quality LSGAN 10%", "+3.15 pp", "3", "Positive paired CI"],
            ["Current-feed external", "Quality LSGAN 10%", "+5.78 pp", "3", "All pairs positive; CI crosses zero"],
        ]
        table(ax,.055,.43,.89,.37,["Experiment","Best/relevant condition","F1 change","Seeds","Evidence"],rows,[.23,.24,.16,.10,.27],9.2)
        box(ax,.055,.14,.89,.20,"Scientific interpretation",
            "The percentage is not the main causal variable. Synthetic relevance to the missed phishing distribution, real-source balance, rejection and quality filtering, class weighting, and the evaluation domain matter more. The new 40% outcome is therefore plausible without invalidating the selected modern 10% result.", BLUE,11.3)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = canvas("Required Next Test", "How to decide whether the 40% candidate truly generalizes", 8)
        steps=[
            "Freeze the three real-only and three WGAN-GP 40% CNN checkpoints and stored thresholds.",
            "Verify zero exact URL, canonical-hostname, and encoded-row overlap with PhiUSIIL.",
            "Evaluate all six checkpoints exactly once; do not recalibrate on the external labels.",
            "Report pooled, per-source, and source-macro metrics, including uncertainty.",
            "Run the planned quality-filtered WGAN-GP 10% condition as a predeclared comparison.",
            "Choose the browser candidate using external recall/F1, ROC-AUC, precision/FPR, size, and latency.",
        ]
        for i,s in enumerate(steps,1):
            y=.80-(i-1)*.105
            ax.add_patch(plt.Circle((.085,y),.023,color=TEAL)); ax.text(.085,y,str(i),ha="center",va="center",color=WHITE,fontweight="bold")
            ax.text(.125,y,textwrap.fill(s,95),va="center",fontsize=11,color=NAVY)
        box(ax,.055,.08,.89,.12,"Final finding",
            "No implementation defect is needed to explain the score difference. The new 40% model is a valid candidate, but its current in-domain gain is statistically unresolved and does not supersede the externally evaluated modern LSGAN result.", ORANGE,11.2)
        pdf.savefig(fig); plt.close(fig)

    print(cfg.output.resolve())


if __name__ == "__main__":
    main()
