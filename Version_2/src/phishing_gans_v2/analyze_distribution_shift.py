from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_json, write_json
from .url_data import LEXICAL_FEATURE_NAMES, lexical_features


SEED = 20260811
FEATURES = tuple(LEXICAL_FEATURE_NAMES)
KEY_FEATURES = (
    "url_length", "hostname_length", "path_length", "query_length",
    "https", "slash_count", "colon_count", "double_slash_count",
    "dot_count_host", "estimated_subdomains", "digit_ratio",
    "character_entropy", "suspicious_token_count", "question_count",
)
GENERATORS = ("gumbel_lsgan", "seqgan", "relgan")
CONDITIONS = (
    "real_only", "random_10_percent", "realism_filtered_10_percent",
    "realism_challenging_10_percent",
)
MODELS = ("cnn", "lstm", "random_forest", "linear_svm")


def sample_rows(values: np.ndarray, indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
    return np.asarray(values[np.sort(chosen)], dtype=np.float32)


def feature_summary(reference: np.ndarray, target: np.ndarray) -> dict:
    reference_mean = reference.mean(axis=0, dtype=np.float64)
    reference_std = reference.std(axis=0, dtype=np.float64)
    target_mean = target.mean(axis=0, dtype=np.float64)
    target_std = target.std(axis=0, dtype=np.float64)
    # A 0.1 floor prevents constant scheme-marker features from producing
    # meaningless million-SD values while their complete separation remains
    # visible in the KS statistic and raw means.
    pooled_std = np.sqrt((np.square(reference_std) + np.square(target_std)) / 2)
    standardized = (target_mean - reference_mean) / np.maximum(pooled_std, 0.1)
    ks = np.asarray([
        ks_2samp(reference[:, index], target[:, index], method="auto").statistic
        for index in range(reference.shape[1])
    ])
    rows = [{
        "feature": FEATURES[index],
        "reference_mean": float(reference_mean[index]),
        "target_mean": float(target_mean[index]),
        "standardized_mean_difference": float(standardized[index]),
        "ks_statistic": float(ks[index]),
    } for index in range(len(FEATURES))]
    return {
        "centroid_distance_in_reference_sd": float(np.linalg.norm(standardized)),
        "mean_ks_statistic": float(ks.mean()),
        "features": rows,
        "largest_shifts": sorted(
            rows, key=lambda row: abs(row["standardized_mean_difference"]), reverse=True
        )[:12],
        "largest_ks": sorted(rows, key=lambda row: row["ks_statistic"], reverse=True)[:12],
    }


def source_auc(internal: np.ndarray, external: np.ndarray, rng: np.random.Generator) -> dict:
    count = min(len(internal), len(external))
    left = internal[rng.choice(len(internal), count, replace=False)]
    right = external[rng.choice(len(external), count, replace=False)]
    values = np.concatenate((left, right))
    labels = np.concatenate((np.zeros(count, dtype=np.uint8), np.ones(count, dtype=np.uint8)))
    model = Pipeline((
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ))
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, values, labels, scoring="roc_auc", cv=folds)
    return {"mean": float(scores.mean()), "sd": float(scores.std(ddof=1)), "values": scores.tolist()}


def load_external() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = list(csv.DictReader(Path("data/external/2026-08-11/external_urls.csv").open(encoding="utf-8")))
    values = np.stack([lexical_features(row["url"]) for row in rows]).astype(np.float32)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    return values, labels, rows


def proportions(values: np.ndarray) -> dict:
    index = {name: FEATURES.index(name) for name in FEATURES}
    return {
        "root_or_empty_path_rate": float((values[:, index["path_length"]] <= 1).mean()),
        "query_present_rate": float((values[:, index["query_length"]] > 0).mean()),
        "https_rate": float((values[:, index["https"]] > 0).mean()),
        "scheme_marker_rate": float((values[:, index["double_slash_count"]] > 0).mean()),
        "longer_than_127_rate": float((values[:, index["url_length"]] > 127).mean()),
        "ip_hostname_rate": float((values[:, index["ip_hostname"]] > 0).mean()),
        "suspicious_token_present_rate": float((values[:, index["suspicious_token_count"]] > 0).mean()),
        "means": {name: float(values[:, index[name]].mean()) for name in KEY_FEATURES},
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    feature_dir = Path("data/features"); data_dir = Path("data/processed")
    train_features = np.load(feature_dir / "train_features.npy", mmap_mode="r")
    train_labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    test_features = np.load(feature_dir / "test_features.npy", mmap_mode="r")
    test_labels = np.load(data_dir / "test_labels.npy", mmap_mode="r")
    external_features, external_labels, external_rows = load_external()

    groups = {
        "train_benign": sample_rows(train_features, np.flatnonzero(train_labels == 0), 100_000, rng),
        "train_phishing": sample_rows(train_features, np.flatnonzero(train_labels == 1), 100_000, rng),
        "test_benign": sample_rows(test_features, np.flatnonzero(test_labels == 0), 50_000, rng),
        "test_phishing": sample_rows(test_features, np.flatnonzero(test_labels == 1), 50_000, rng),
        "external_benign": external_features[external_labels == 0],
        "external_phishing": external_features[external_labels == 1],
    }
    for generator in GENERATORS:
        pool = np.load(Path("data/augmented") / generator / "generated_pool_features.npy", mmap_mode="r")
        indices = np.arange(len(pool), dtype=np.int64)
        groups[f"gan_{generator}"] = sample_rows(pool, indices, 100_000, rng)
    for condition in ("realism_filtered_10_percent", "realism_challenging_10_percent"):
        pool = np.load(Path("data/augmented/seqgan_quality") / f"{condition}_features.npy", mmap_mode="r")
        groups[f"gan_{condition}"] = sample_rows(pool, np.arange(len(pool)), 100_000, rng)

    shifts = {
        "external_benign_vs_train_benign": feature_summary(groups["train_benign"], groups["external_benign"]),
        "external_phishing_vs_train_phishing": feature_summary(groups["train_phishing"], groups["external_phishing"]),
        "internal_test_benign_vs_train_benign": feature_summary(groups["train_benign"], groups["test_benign"]),
        "internal_test_phishing_vs_train_phishing": feature_summary(groups["train_phishing"], groups["test_phishing"]),
    }
    gan_shifts = {
        name: {
            "vs_train_phishing": feature_summary(groups["train_phishing"], values),
            "vs_external_phishing": feature_summary(groups["external_phishing"], values),
        }
        for name, values in groups.items() if name.startswith("gan_")
    }

    # A simple lexical classifier shows which acquisition artifacts learned on
    # the old data reverse direction on the later snapshot.
    x_train = np.concatenate((groups["train_benign"], groups["train_phishing"]))
    y_train = np.concatenate((np.zeros(len(groups["train_benign"])), np.ones(len(groups["train_phishing"]))))
    lexical_model = Pipeline((
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ))
    lexical_model.fit(x_train, y_train)
    internal_x = np.concatenate((groups["test_benign"], groups["test_phishing"]))
    internal_y = np.concatenate((np.zeros(len(groups["test_benign"])), np.ones(len(groups["test_phishing"]))))
    external_scores = lexical_model.predict_proba(external_features)[:, 1]
    internal_scores = lexical_model.predict_proba(internal_x)[:, 1]
    scaler = lexical_model.named_steps["scale"]
    coefficients = lexical_model.named_steps["classifier"].coef_[0]
    contributions = {}
    for label, source_name, target_name in (
        (0, "train_benign", "external_benign"),
        (1, "train_phishing", "external_phishing"),
    ):
        delta = (groups[target_name].mean(0) - groups[source_name].mean(0)) / scaler.scale_
        signed = delta * coefficients
        rows = [{
            "feature": FEATURES[index], "logit_shift_toward_phishing": float(signed[index]),
            "harmful_for_correct_classification": bool(signed[index] > 0 if label == 0 else signed[index] < 0),
        } for index in range(len(FEATURES))]
        contributions[target_name] = sorted(
            rows,
            key=lambda row: abs(row["logit_shift_toward_phishing"])
            if row["harmful_for_correct_classification"] else -1,
            reverse=True,
        )[:10]

    external_summary = load_json(Path("artifacts/results/external/2026-08-11_phishtank_topmillion/summary.json"))
    classifier_attribution = {}
    for condition in CONDITIONS:
        classifier_attribution[condition] = {
            metric: float(np.mean([
                external_summary["grouped"][model][condition]["native_metrics"][metric]["mean"]
                for model in MODELS
            ]))
            for metric in ("accuracy", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate")
        }
    classifier_attribution["real_only_models_below_auc_0_5"] = [
        model for model in MODELS
        if external_summary["grouped"][model]["real_only"]["native_metrics"]["roc_auc"]["mean"] < .5
    ]

    profile = {name: proportions(values) for name, values in groups.items()}
    payload = {
        "project_version": 2,
        "analysis_role": "diagnostic only; August 11 external snapshot is prohibited for future tuning claims",
        "sample_counts": {name: len(values) for name, values in groups.items()},
        "profiles": profile,
        "feature_shifts": shifts,
        "source_separability_auc": {
            "benign_internal_vs_external": source_auc(groups["train_benign"], groups["external_benign"], rng),
            "phishing_internal_vs_external": source_auc(groups["train_phishing"], groups["external_phishing"], rng),
        },
        "lexical_label_model": {
            "internal_test_roc_auc": float(roc_auc_score(internal_y, internal_scores)),
            "external_roc_auc": float(roc_auc_score(external_labels, external_scores)),
            "harmful_shift_contributions": contributions,
        },
        "gan_feature_shifts": gan_shifts,
        "classifier_condition_macro_means": classifier_attribution,
        "external_source_counts": {
            source: sum(row["source"] == source for row in external_rows)
            for source in sorted({row["source"] for row in external_rows})
        },
    }

    output_dir = Path("artifacts/results/distribution_shift")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", payload)

    # Heatmap of the most shifted features, selected independently of labels.
    benign_rows = {row["feature"]: row for row in shifts["external_benign_vs_train_benign"]["features"]}
    phishing_rows = {row["feature"]: row for row in shifts["external_phishing_vs_train_phishing"]["features"]}
    signed_ks = lambda row: np.sign(row["target_mean"] - row["reference_mean"]) * row["ks_statistic"]
    top = sorted(FEATURES, key=lambda name: max(abs(signed_ks(benign_rows[name])), abs(signed_ks(phishing_rows[name]))), reverse=True)[:14]
    heat = np.asarray([[signed_ks(benign_rows[name]) for name in top], [signed_ks(phishing_rows[name]) for name in top]])
    fig, axis = plt.subplots(figsize=(14, 3.8))
    image = axis.imshow(heat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(top)), [name.replace("_", " ") for name in top], rotation=32, ha="right")
    axis.set_yticks((0, 1), ("External legitimate vs train legitimate", "External phishing vs train phishing"))
    for row in range(2):
        for column in range(len(top)):
            axis.text(column, row, f"{heat[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("External lexical shifts: signed Kolmogorov–Smirnov statistic")
    fig.colorbar(image, ax=axis, label="Signed distribution shift (−1 to +1)")
    fig.tight_layout(); fig.savefig(output_dir / "feature_shift_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "feature_shift_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    condition_names = list(CONDITIONS)
    x = np.arange(len(MODELS)); width = .19
    colors = ("#2b6cb0", "#dd6b20", "#319795", "#805ad5")
    for index, condition in enumerate(condition_names):
        aucs = [100 * external_summary["grouped"][model][condition]["native_metrics"]["roc_auc"]["mean"] for model in MODELS]
        axes[0].bar(x + (index - 1.5) * width, aucs, width, label=condition.replace("_", " "), color=colors[index])
    axes[0].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, ("CNN", "LSTM", "RF", "SVM")); axes[0].set_ylim(25, 65)
    axes[0].set_ylabel("External ROC-AUC (%)"); axes[0].set_title("Real-only already shifts; augmentation is secondary")
    axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=.25)

    names = [f"gan_{g}" for g in GENERATORS] + ["gan_realism_filtered_10_percent", "gan_realism_challenging_10_percent"]
    train_dist = [gan_shifts[name]["vs_train_phishing"]["mean_ks_statistic"] for name in names]
    external_dist = [gan_shifts[name]["vs_external_phishing"]["mean_ks_statistic"] for name in names]
    xx = np.arange(len(names)); w = .36
    axes[1].bar(xx - w/2, train_dist, w, label="distance from train phishing", color="#319795")
    axes[1].bar(xx + w/2, external_dist, w, label="distance from external phishing", color="#c53030")
    axes[1].set_xticks(xx, ("Gumbel", "SeqGAN", "RelGAN", "Filtered", "Challenging"), rotation=20, ha="right")
    axes[1].set_ylabel("Mean lexical KS statistic"); axes[1].set_ylim(0, .4); axes[1].set_title("GANs inherit the training distribution")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Distribution-shift attribution")
    fig.tight_layout(); fig.savefig(output_dir / "attribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "attribution.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"output_dir": str(output_dir), "groups": payload["sample_counts"]}, indent=2))


if __name__ == "__main__":
    main()
