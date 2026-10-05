#!/usr/bin/env python3
"""Generate the report figure for the consented browser-extension survey."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


REPORT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPORT_DIR.parents[1]
SURVEY = PROJECT_DIR / "Documentation" / "Screenshots" / "تقييم استخدام اضافة المتصفح لمشروع التخرج.csv"
OUTPUT_PDF = REPORT_DIR / "assets" / "graphs" / "user_survey_results.pdf"
OUTPUT_PNG = REPORT_DIR / "assets" / "graphs" / "user_survey_results.png"


def main() -> None:
    with SURVEY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 survey responses, found {len(rows)}")

    values = [list(row.values()) for row in rows]
    clarity = sum(float(row[2]) for row in values) / len(values)
    rating = sum(float(row[10]) for row in values) / len(values)
    understood = sum(row[3] == "نعم" for row in values) / len(values) * 100
    helpful = sum(row[9] in {"نعم", "إلى حد ما"} for row in values) / len(values) * 100
    no_speed_effect = sum(row[7] == "لم ألاحظ أي تأثير" for row in values) / len(values) * 100
    trusted_warning = sum(row[4] == "نعم" for row in values) / len(values) * 100
    safe_interruption = sum(row[8] != "لا" for row in values) / len(values) * 100

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [0.75, 1.55]})
    fig.suptitle("Browser Extension User Survey (n = 10)", fontsize=16, fontweight="bold", y=0.98)

    score_names = ["Warning clarity", "Overall rating"]
    score_values = [clarity, rating]
    bars = axes[0].barh(score_names, score_values, color=["#2A9D8F", "#4C78A8"], height=0.52)
    axes[0].set_xlim(0, 5)
    axes[0].set_xlabel("Mean score out of 5")
    axes[0].set_title("Ratings", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.22)
    axes[0].invert_yaxis()
    for bar, value in zip(bars, score_values):
        axes[0].text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.1f}/5", va="center", fontweight="bold")

    item_names = [
        "Understood warning tiers",
        "Helpful or partly helpful",
        "No noticeable speed effect",
        "Warnings on trusted sites",
        "Safe-site interruption reported",
    ]
    item_values = [understood, helpful, no_speed_effect, trusted_warning, safe_interruption]
    colors = ["#2A9D8F", "#2A9D8F", "#2A9D8F", "#E07A5F", "#E07A5F"]
    bars = axes[1].barh(item_names, item_values, color=colors, height=0.58)
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Participants (%)")
    axes[1].set_title("Responses", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].invert_yaxis()
    for bar, value in zip(bars, item_values):
        axes[1].text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_axisbelow(True)

    fig.text(0.5, 0.015, "Green/blue: favourable response  •  Orange: reported usability problem", ha="center", color="#4B5563")
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.92), w_pad=3.0)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
