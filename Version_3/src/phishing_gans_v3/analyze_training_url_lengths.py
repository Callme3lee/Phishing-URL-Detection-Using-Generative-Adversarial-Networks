from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import load_json, write_json
from .metrics import classification_metrics


REAL_MANIFEST = Path("data/experiments/recent_update_gan10_v1/real_only/processed/train_manifest.csv")
GAN_MANIFEST = Path("data/experiments/recent_update_gan10_v1/gan_10pct/processed/train_manifest.csv")
POLICY_MANIFEST = Path("artifacts/final_browser_policy_v0_5_2_2026-09-01/manifest.json")
POLICY_PREDICTIONS = Path("artifacts/final_browser_policy_v0_5_2_2026-09-01/policy_development_predictions.npz")
INSTALLED_POLICY = Path("browser_extension_user_pilot/models/policy.json")
OUTPUT = Path("artifacts/results/training_data_length_audit_2026-09-01_v2")

BUCKETS = [-1, 30, 60, 100, 128, np.inf]
BUCKET_LABELS = ["1-30", "31-60", "61-100", "101-128", ">128"]


def percentile(series: pd.Series, value: float) -> float:
    return float(series.quantile(value, interpolation="linear"))


def summary_row(condition: str, origin: str, class_name: str, frame: pd.DataFrame) -> dict:
    lengths = frame["url_length"]
    return {
        "condition": condition,
        "origin": origin,
        "class": class_name,
        "rows": int(len(frame)),
        "unique_urls": int(frame["url"].nunique()),
        "duplicate_rows": int(len(frame) - frame["url"].nunique()),
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
        "p90_length": percentile(lengths, 0.90),
        "p95_length": percentile(lengths, 0.95),
        "maximum_length": int(lengths.max()),
        "over_100": int((lengths > 100).sum()),
        "over_100_pct": float((lengths > 100).mean()),
        "over_128": int((lengths > 128).sum()),
        "over_128_pct": float((lengths > 128).mean()),
    }


def metric_row(name: str, threshold: float, labels: np.ndarray, scores: np.ndarray, source: str) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    return {
        "option": name,
        "threshold": threshold,
        "development_accuracy": metrics["accuracy"],
        "development_precision": metrics["precision"],
        "development_recall": metrics["recall"],
        "development_f1": metrics["f1"],
        "development_fpr": metrics["false_positive_rate"],
        "development_fnr": metrics["false_negative_rate"],
        "selection_source": source,
    }


def threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    phishing_scores = np.sort(scores[labels == 1])[::-1]
    required = int(np.ceil(target * len(phishing_scores)))
    return float(phishing_scores[required - 1])


def markdown_table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if column == "threshold":
                values.append(f"{float(value):.6f}")
            elif column in percent_columns:
                values.append(f"{100 * float(value):.2f}%")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.2f}")
            elif isinstance(value, (int, np.integer)):
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit at {OUTPUT}")
    usecols = ["url", "label", "canonical_hostname", "source_group"]
    real = pd.read_csv(REAL_MANIFEST, usecols=usecols, keep_default_na=False)
    gan = pd.read_csv(GAN_MANIFEST, usecols=usecols, keep_default_na=False)
    for frame in (real, gan):
        frame["label"] = frame["label"].astype(int)
        frame["url_length"] = frame["url"].astype(str).str.len()
        frame["origin"] = np.where(frame["source_group"].eq("synthetic_phishing"), "Synthetic", "Real")
        frame["class"] = np.where(frame["label"].eq(1), "Phishing", "Legitimate")
        frame["length_bucket"] = pd.cut(
            frame["url_length"], bins=BUCKETS, labels=BUCKET_LABELS, include_lowest=True, right=True
        )

    real_urls = set(real["url"])
    gan_urls = set(gan["url"])
    synthetic = gan[gan["origin"].eq("Synthetic")].copy()
    gan_real = gan[gan["origin"].eq("Real")].copy()
    real_legitimate = real[real["label"].eq(0)].copy()
    real_phishing = real[real["label"].eq(1)].copy()
    synthetic_phishing = synthetic[synthetic["label"].eq(1)].copy()

    overlap = {
        "real_only_rows": int(len(real)),
        "gan_condition_rows": int(len(gan)),
        "real_rows_in_gan_condition": int(len(gan_real)),
        "synthetic_rows_in_gan_condition": int(len(synthetic)),
        "real_manifest_unique_urls": int(len(real_urls)),
        "gan_manifest_unique_urls": int(len(gan_urls)),
        "exact_url_intersection": int(len(real_urls & gan_urls)),
        "real_urls_missing_from_gan_condition": int(len(real_urls - gan_urls)),
        "extra_urls_in_gan_condition": int(len(gan_urls - real_urls)),
        "unique_url_jaccard": float(len(real_urls & gan_urls) / len(real_urls | gan_urls)),
        "real_rows_identical_and_in_same_order": bool(
            len(gan_real) == len(real)
            and np.array_equal(real["url"].to_numpy(), gan_real["url"].to_numpy())
            and np.array_equal(real["label"].to_numpy(), gan_real["label"].to_numpy())
        ),
        "synthetic_exact_url_overlap_with_real": int(len(set(synthetic["url"]) & real_urls)),
        "synthetic_unique_hostnames": int(synthetic["canonical_hostname"].nunique()),
        "synthetic_hostname_overlap_with_any_real": int(
            len(set(synthetic["canonical_hostname"]) & set(real["canonical_hostname"]))
        ),
        "synthetic_hostname_overlap_with_real_legitimate": int(
            len(
                set(synthetic["canonical_hostname"])
                & set(real_legitimate["canonical_hostname"])
            )
        ),
        "synthetic_hostname_overlap_with_real_phishing": int(
            len(
                set(synthetic["canonical_hostname"])
                & set(real_phishing["canonical_hostname"])
            )
        ),
        "real_cross_class_exact_url_overlap": int(
            len(set(real_legitimate["url"]) & set(real_phishing["url"]))
        ),
    }

    summary_frames = [
        ("Real-only", "Real", "Legitimate", real_legitimate),
        ("Real-only", "Real", "Phishing", real_phishing),
        ("GAN condition", "Real", "Legitimate", gan_real[gan_real["label"].eq(0)]),
        ("GAN condition", "Real", "Phishing", gan_real[gan_real["label"].eq(1)]),
        ("GAN condition", "Synthetic", "Phishing", synthetic_phishing),
        ("GAN condition", "All", "Legitimate", gan[gan["label"].eq(0)]),
        ("GAN condition", "All", "Phishing", gan[gan["label"].eq(1)]),
    ]
    summary = pd.DataFrame(
        [summary_row(condition, origin, class_name, frame) for condition, origin, class_name, frame in summary_frames]
    )

    bucket_rows = []
    distribution_frames = [
        ("Real-only", "Real", "Legitimate", real_legitimate),
        ("Real-only", "Real", "Phishing", real_phishing),
        ("GAN condition", "Synthetic", "Phishing", synthetic_phishing),
        ("GAN condition", "All", "Legitimate", gan[gan["label"].eq(0)]),
        ("GAN condition", "All", "Phishing", gan[gan["label"].eq(1)]),
    ]
    for condition, origin, class_name, frame in distribution_frames:
        counts = frame["length_bucket"].value_counts(sort=False)
        for bucket in BUCKET_LABELS:
            count = int(counts.get(bucket, 0))
            bucket_rows.append({
                "condition": condition,
                "origin": origin,
                "class": class_name,
                "length_bucket": bucket,
                "count": count,
                "percentage_within_group": count / len(frame),
            })
    buckets = pd.DataFrame(bucket_rows)

    # Histogram-overlap coefficient is 1 for identical bucket proportions and 0
    # for disjoint distributions.
    real_phish_distribution = np.asarray([
        (real_phishing["length_bucket"] == bucket).mean() for bucket in BUCKET_LABELS
    ])
    synthetic_distribution = np.asarray([
        (synthetic_phishing["length_bucket"] == bucket).mean() for bucket in BUCKET_LABELS
    ])
    overlap["real_vs_synthetic_phishing_length_bucket_overlap_coefficient"] = float(
        np.minimum(real_phish_distribution, synthetic_distribution).sum()
    )

    policy_manifest = load_json(POLICY_MANIFEST)
    installed_policy = load_json(INSTALLED_POLICY)
    predictions = np.load(POLICY_PREDICTIONS)
    labels = predictions["labels"].astype(np.uint8)
    scores = predictions["browser_onnx_scores"].astype(np.float64)
    current_threshold = float(installed_policy["primary_binary_rule"]["threshold"])
    candidate_threshold = float(policy_manifest["browser_candidate"]["primary_threshold"])
    threshold_rows = [
        metric_row("Installed v0.5.1", current_threshold, labels, scores, "Existing installed policy"),
        metric_row("Frozen browser candidate", candidate_threshold, labels, scores, "Frozen policy-development FPR rule"),
        metric_row("Development recall 80%", threshold_for_recall(labels, scores, 0.80), labels, scores, "Policy-development recall target"),
        metric_row("Development recall 85%", threshold_for_recall(labels, scores, 0.85), labels, scores, "Policy-development recall target"),
    ]
    thresholds = pd.DataFrame(threshold_rows)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary.to_csv(OUTPUT / "length_summary.csv", index=False)
    buckets.to_csv(OUTPUT / "length_buckets.csv", index=False)
    thresholds.to_csv(OUTPUT / "threshold_options.csv", index=False)
    write_json(OUTPUT / "overlap_audit.json", overlap)
    collision_rows = synthetic.merge(
        real[["url", "label", "source_group"]], on="url", how="inner",
        suffixes=("_synthetic", "_real"),
    )
    collision_audit = pd.DataFrame({
        "url_sha256": collision_rows["url"].map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        ),
        "synthetic_label": collision_rows["label_synthetic"].astype(int),
        "real_label": collision_rows["label_real"].astype(int),
        "real_source_group": collision_rows["source_group_real"],
        "labels_conflict": (
            collision_rows["label_synthetic"].astype(int)
            != collision_rows["label_real"].astype(int)
        ),
    })
    collision_audit.to_csv(OUTPUT / "exact_overlap_conflicts_hashed.csv", index=False)

    plot_groups = [
        ("Real legitimate", real_legitimate),
        ("Real phishing", real_phishing),
        ("Synthetic phishing", synthetic_phishing),
    ]
    plot_values = np.asarray([
        [(frame["length_bucket"] == bucket).mean() * 100 for bucket in BUCKET_LABELS]
        for _, frame in plot_groups
    ])
    colors = ["#3B6EA8", "#6A93BD", "#E4A11B", "#D66B4D", "#B94A48"]
    fig, axis = plt.subplots(figsize=(12.5, 6.8), facecolor="white")
    left = np.zeros(len(plot_groups))
    for index, bucket in enumerate(BUCKET_LABELS):
        bars = axis.barh(
            [name for name, _ in plot_groups], plot_values[:, index], left=left,
            label=bucket, color=colors[index], height=0.55,
        )
        for bar, value, start in zip(bars, plot_values[:, index], left):
            if value >= 5:
                axis.text(start + value / 2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                          ha="center", va="center", fontsize=9, weight="bold", color="white")
        left += plot_values[:, index]
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percentage")
    axis.set_title("Training URL Length Distribution", fontsize=19, weight="bold", color="#17324D", pad=18)
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(title="Characters", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.22)
    fig.savefig(OUTPUT / "length_distribution.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    display_summary = summary[[
        "condition", "origin", "class", "rows", "mean_length", "median_length",
        "p95_length", "maximum_length", "over_100", "over_100_pct", "over_128", "over_128_pct",
    ]].copy()
    display_buckets = buckets.copy()
    display_thresholds = thresholds.copy()
    report = f"""# Training-set similarity and URL-length audit

## Exact training-set relationship

- Real-only contains **{overlap['real_only_rows']:,} real rows**.
- The GAN condition contains the same **{overlap['real_rows_in_gan_condition']:,} real rows** plus **{overlap['synthetic_rows_in_gan_condition']:,} synthetic phishing rows**.
- Real rows are identical and in the same order: **{overlap['real_rows_identical_and_in_same_order']}**.
- Synthetic URLs exactly overlapping real URLs: **{overlap['synthetic_exact_url_overlap_with_real']:,}**.
- Exact synthetic hostname overlap with real legitimate hostnames: **{overlap['synthetic_hostname_overlap_with_real_legitimate']:,}**.
- Exact synthetic hostname overlap with real phishing hostnames: **{overlap['synthetic_hostname_overlap_with_real_phishing']:,}**.
- Exact legitimate/phishing URL overlap inside the real set: **{overlap['real_cross_class_exact_url_overlap']:,}**.
- Real-versus-synthetic phishing length-bucket overlap coefficient: **{overlap['real_vs_synthetic_phishing_length_bucket_overlap_coefficient']:.3f}**.

The one exact synthetic/real URL collision has conflicting labels: synthetic
phishing versus real legitimate. It is recorded by SHA-256 in
`exact_overlap_conflicts_hashed.csv`. One row out of 407,594 cannot plausibly
explain the observed FNR, but future synthetic filtering should reject matches
against every real class, not only the phishing pool.

## Length summary

The character-CNN branch reads at most 128 characters. `over_128` therefore
counts URLs whose tail is truncated in that branch; the hybrid lexical branch
still receives full-URL summary features such as total length and character
counts.

{markdown_table(display_summary, {'over_100_pct', 'over_128_pct'})}

## Sorted length buckets

{markdown_table(display_buckets, {'percentage_within_group'})}

## Threshold choices measured on the separate policy-development set

{markdown_table(display_thresholds, {'development_accuracy', 'development_precision', 'development_recall', 'development_f1', 'development_fpr', 'development_fnr'})}

The 10K and original frozen final-test labels were not used to calculate the
recall-target thresholds. They must not be evaluated on those same files and
then presented as new final results.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "overlap": overlap,
        "length_summary": summary.to_dict(orient="records"),
        "threshold_options": thresholds.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
