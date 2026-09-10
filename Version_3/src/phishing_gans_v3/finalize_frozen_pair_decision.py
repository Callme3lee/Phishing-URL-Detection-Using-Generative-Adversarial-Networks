from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .common import load_json, sha256_file, write_json


ROOT = Path("artifacts/results/final_model_decision_2026-08-22")
PRIMARY = Path("artifacts/results/final_confirmation_2026-08-22_30k_5pct/results.json")
STRESS = Path("artifacts/results/final_long_url_stress_2026-08-22/results.json")
PROTOCOL = Path("artifacts/final_confirmation_protocol_2026-08-22/manifest.json")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def table_page(pdf: PdfPages, title: str, subtitle: str, result: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(.055, .91, title, fontsize=23, weight="bold", color="#17324D")
    fig.text(.055, .855, subtitle, fontsize=12.5, color="#3B6EA8")
    columns = ["Condition", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR"]
    rows = []
    for key, label in (("real_only", "HYB-MS real-only"), ("gan_10pct", "HYB-MS +10% GAN")):
        values = result["conditions"][key]["metrics"]
        rows.append([label] + [pct(values[metric]) for metric in METRICS])
    axis = fig.add_axes([.035, .57, .93, .20]); axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                       colWidths=[.20, .10, .105, .09, .09, .10, .10, .085, .085])
    table.auto_set_font_size(False); table.set_fontsize(8.7); table.scale(1, 2.1)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CAD5E1")
        if row == 0: cell.set_facecolor("#3B6EA8"); cell.set_text_props(color="white", weight="bold")
        elif row == 2: cell.set_facecolor("#E7F3F1")
    paired = result["paired_exact_comparison"]
    fig.text(.06, .43, result["verdict"]["heading"], fontsize=17, weight="bold",
             color="#2A9D8F" if result["verdict"]["passed"] else "#D9534F")
    fig.text(.06, .365, result["verdict"]["explanation"], fontsize=11.5, color="#526678")
    fig.text(.06, .275, f"All rows: McNemar p = {paired['all_rows']['two_sided_exact_mcnemar_p_value']:.4g}", fontsize=11, color="#17324D")
    fig.text(.06, .225, f"Legitimate rows: p = {paired['legitimate_rows']['two_sided_exact_mcnemar_p_value']:.4g}    Phishing rows: p = {paired['phishing_rows']['two_sided_exact_mcnemar_p_value']:.4g}", fontsize=11, color="#17324D")
    fig.text(.06, .13, "Bootstrap intervals and all checkpoint/dataset hashes are stored in the machine-readable result.", fontsize=10.5, color="#526678")
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"Final decision already exists at {ROOT}")
    protocol = load_json(PROTOCOL); primary = load_json(PRIMARY); stress = load_json(STRESS)
    if primary["protocol_sha256"] != sha256_file(PROTOCOL) or stress["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("An evaluation does not match the frozen protocol")
    primary_pass = bool(primary["verdict"]["passed"])
    stress_pass = bool(stress["verdict"]["passed"])
    gan_selected = primary_pass and stress_pass
    selected = "gan_10pct" if gan_selected else "real_only"
    decision = {
        "formal_outcome": (
            "Select weighted HYB-MS +10% MS-HardGAN-C"
            if gan_selected else
            "Retain weighted HYB-MS real-only as the conservative final reference"
        ),
        "selected_condition": selected,
        "primary_rule_passed": primary_pass,
        "long_url_non_regression_passed": stress_pass,
        "reason": (
            "Both the primary augmentation rule and long-URL FPR non-regression rule passed."
            if gan_selected else
            "The +10% model passed the 30K primary rule but increased FPR on the separate long-URL stress set, so the frozen two-part selection rule does not permit selecting it as the unconditional final model."
        ),
        "important_context": (
            "The +10% model remains the best low-prevalence candidate and its 30K overall paired gain is significant. "
            "The long-URL FPR increase is not conventionally significant, but the predeclared non-regression rule was directional and is not changed after seeing results."
        ),
        "deployment_ready": False,
        "deployment_limit": "Operational extension latency, memory, live false-alert burden, failure handling, and drift monitoring remain unvalidated.",
    }
    payload = {
        "project_version": 3, "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "primary_result": str(PRIMARY), "primary_result_sha256": sha256_file(PRIMARY),
        "stress_result": str(STRESS), "stress_result_sha256": sha256_file(STRESS),
        "decision": decision,
        "primary_summary": {key: primary["conditions"][key]["metrics"] for key in ("real_only", "gan_10pct")},
        "stress_summary": {key: stress["conditions"][key]["metrics"] for key in ("real_only", "gan_10pct")},
        "statistical_summary": {
            "primary_all_rows_mcnemar_p": primary["paired_exact_comparison"]["all_rows"]["two_sided_exact_mcnemar_p_value"],
            "primary_phishing_rows_mcnemar_p": primary["paired_exact_comparison"]["phishing_rows"]["two_sided_exact_mcnemar_p_value"],
            "primary_f1_delta_ci95": primary["paired_stratified_bootstrap"]["metrics"]["f1"]["confidence_interval_95"],
            "primary_fpr_delta_ci95": primary["paired_stratified_bootstrap"]["metrics"]["false_positive_rate"]["confidence_interval_95"],
            "stress_legitimate_rows_mcnemar_p": stress["paired_exact_comparison"]["legitimate_rows"]["two_sided_exact_mcnemar_p_value"],
            "stress_fpr_delta_ci95": stress["paired_stratified_bootstrap"]["metrics"]["false_positive_rate"]["confidence_interval_95"],
        },
    }
    ROOT.mkdir(parents=True)
    write_json(ROOT / "decision.json", payload)
    readme = f"""# Final frozen-model decision

## Decision

**{decision['formal_outcome']}.**

The +10% model clearly won the 30K/5% primary confirmation, but its long-legitimate-URL FPR was 1.45% versus 1.05% for real-only. The strict predeclared rule required no FPR increase on that stress set, so it did not pass the complete two-part selection gate.

This does not erase the GAN result: on the primary set, +10% improved F1 from 79.06% to 81.27%, reduced FPR from 1.26% to 1.01%, and produced a significant paired overall gain (p = 0.0000743). It remains the best ordinary low-prevalence candidate, while real-only is retained as the conservative unconditional reference.

Neither model is declared deployment-ready. No checkpoint, threshold, or dataset was changed after scoring began.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    with PdfPages(ROOT / "final_model_decision.pdf") as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .91, "Final frozen-model decision", fontsize=26, weight="bold", color="#17324D")
        fig.text(.055, .84, "Two locked cohorts • unchanged models and thresholds • paired statistics", fontsize=13, color="#3B6EA8")
        fig.text(.055, .70, decision["formal_outcome"], fontsize=18, weight="bold", color="#D97706", wrap=True)
        fig.text(.055, .57, decision["reason"], fontsize=12, color="#526678", wrap=True)
        fig.text(.055, .43, "What the evidence says", fontsize=16, weight="bold", color="#17324D")
        fig.text(.07, .355, "• +10% GAN is better on the large ordinary low-prevalence cohort.", fontsize=12, color="#17324D")
        fig.text(.07, .295, "• Real-only produces fewer false alerts on deliberately long legitimate URLs.", fontsize=12, color="#17324D")
        fig.text(.07, .235, "• The complete strict selection rule therefore does not yield an unconditional GAN winner.", fontsize=12, color="#17324D")
        fig.text(.055, .12, "Deployment status: NOT READY — operational browser validation is still required.", fontsize=12, weight="bold", color="#B42318")
        pdf.savefig(fig); plt.close(fig)
        table_page(pdf, "Primary confirmation", "30,000 URLs • 5% phishing • zero URL/host/encoding overlap", primary)
        table_page(pdf, "Long-URL stress test", "2,000 long legitimate paths/queries + 200 phishing URLs", stress)
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
