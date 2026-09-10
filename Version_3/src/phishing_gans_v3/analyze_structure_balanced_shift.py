from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import write_json
from .url_data import LEXICAL_FEATURE_NAMES, lexical_features


STRUCTURES = ("root", "shallow", "deep", "query")
KEY_FEATURES = ("url_length", "hostname_length", "path_length", "query_length", "dot_count_host", "estimated_subdomains")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify shift before and after URLs_New-only structure balancing.")
    parser.add_argument("--base-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-features", type=Path, default=Path("data/features"))
    parser.add_argument("--experiment-data", type=Path, default=Path("data/experiments/urls_new_structure_balanced_v1/processed"))
    parser.add_argument("--experiment-features", type=Path, default=Path("data/experiments/urls_new_structure_balanced_v1/features"))
    parser.add_argument("--old-external", type=Path, default=Path("data/external/2026-08-11/external_urls.csv"))
    parser.add_argument("--new-external", type=Path, default=Path("data/external/2026-08-11-openphish-tranco/external_urls.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/urls_new_structure_balanced_v1/distribution_shift"))
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def structure(url: str) -> str:
    parsed = urlsplit("http://" + url.lstrip("/"))
    if parsed.query:
        return "query"
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if not segments:
        return "root"
    if len(segments) == 1:
        return "shallow"
    return "deep"


def manifest_structures(path: Path) -> dict[str, dict[str, float | int]]:
    counts = Counter()
    totals = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            counts[(label, structure(row["url"]))] += 1
            totals[label] += 1
    return {
        name: {
            category: {
                "rows": counts[(label, category)],
                "proportion": counts[(label, category)] / totals[label],
            }
            for category in STRUCTURES
        }
        for label, name in ((0, "legitimate"), (1, "phishing"))
    }


def external_data(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    matrix = np.stack([lexical_features(row["url"]) for row in rows]).astype(np.float32)
    counts = Counter((int(row["label"]), structure(row["url"])) for row in rows)
    totals = Counter(int(row["label"]) for row in rows)
    profile = {
        name: {
            category: {
                "rows": counts[(label, category)],
                "proportion": counts[(label, category)] / totals[label],
            }
            for category in STRUCTURES
        }
        for label, name in ((0, "legitimate"), (1, "phishing"))
    }
    return matrix, labels, profile


def feature_means(matrix: np.ndarray) -> dict[str, float]:
    return {name: float(matrix[:, LEXICAL_FEATURE_NAMES.index(name)].mean()) for name in KEY_FEATURES}


def total_variation(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> float:
    return 0.5 * sum(abs(float(left[name]["proportion"]) - float(right[name]["proportion"])) for name in STRUCTURES)


def source_auc(internal: np.ndarray, external: np.ndarray, repeats: int) -> dict[str, object]:
    values = []
    take = len(external)
    for repeat in range(repeats):
        rng = np.random.default_rng(4200 + repeat)
        chosen = rng.choice(len(internal), size=take, replace=False)
        values_matrix = np.concatenate((internal[chosen], external), axis=0)
        source_labels = np.concatenate((np.zeros(take, dtype=np.uint8), np.ones(take, dtype=np.uint8)))
        train_x, test_x, train_y, test_y = train_test_split(
            values_matrix, source_labels, test_size=0.35, stratify=source_labels, random_state=8400 + repeat
        )
        model = Pipeline((
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=12600 + repeat)),
        ))
        model.fit(train_x, train_y)
        values.append(float(roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])))
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "values": values,
        "interpretation": "0.5 means lexically indistinguishable sources; 1.0 means perfectly separable sources",
    }


def main() -> None:
    args = parse_args()
    base_matrix = np.load(args.base_features / "train_features.npy", mmap_mode="r")
    base_labels = np.load(args.base_data / "train_labels.npy", mmap_mode="r")
    experiment_matrix = np.load(args.experiment_features / "train_features.npy", mmap_mode="r")
    experiment_labels = np.load(args.experiment_data / "train_labels.npy", mmap_mode="r")
    base_structure = manifest_structures(args.base_data / "train_manifest.csv")
    experiment_structure = manifest_structures(args.experiment_data / "train_manifest.csv")
    externals = {
        "topdomain_phishtank": external_data(args.old_external),
        "tranco_openphish": external_data(args.new_external),
    }
    payload: dict[str, object] = {
        "protocol": "lexical and structural shift; source classifier repeated holdout; no model-score tuning",
        "structures": {"original_train": base_structure, "balanced_train": experiment_structure},
        "external": {},
    }
    for external_name, (external_matrix, external_labels, external_structure) in externals.items():
        external_payload = {"structures": external_structure, "classes": {}}
        for label, class_name in ((0, "legitimate"), (1, "phishing")):
            base_class = np.asarray(base_matrix[base_labels == label])
            experiment_class = np.asarray(experiment_matrix[experiment_labels == label])
            external_class = external_matrix[external_labels == label]
            reference_std = np.std(base_class, axis=0)
            reference_std[reference_std < 1e-6] = 1.0
            original_smd = np.abs(np.mean(base_class, axis=0) - np.mean(external_class, axis=0)) / reference_std
            balanced_smd = np.abs(np.mean(experiment_class, axis=0) - np.mean(external_class, axis=0)) / reference_std
            external_payload["classes"][class_name] = {
                "feature_means": {
                    "original_train": feature_means(base_class),
                    "balanced_train": feature_means(experiment_class),
                    "external": feature_means(external_class),
                },
                "mean_absolute_standardized_mean_difference_27_features": {
                    "original_train_to_external": float(np.mean(original_smd)),
                    "balanced_train_to_external": float(np.mean(balanced_smd)),
                    "delta_balanced_minus_original": float(np.mean(balanced_smd) - np.mean(original_smd)),
                },
                "structure_total_variation_distance": {
                    "original_train_to_external": total_variation(base_structure[class_name], external_structure[class_name]),
                    "balanced_train_to_external": total_variation(experiment_structure[class_name], external_structure[class_name]),
                },
                "lexical_source_classifier_auc": {
                    "original_train_vs_external": source_auc(base_class, external_class, args.repeats),
                    "balanced_train_vs_external": source_auc(experiment_class, external_class, args.repeats),
                },
            }
        payload["external"][external_name] = external_payload
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "distribution_shift.json", payload)

    x = np.arange(len(STRUCTURES))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, external_name in zip(axes, ("topdomain_phishtank", "tranco_openphish")):
        profiles = (
            (base_structure["legitimate"], "#4C78A8", "Original train"),
            (experiment_structure["legitimate"], "#F58518", "Balanced train"),
            (payload["external"][external_name]["structures"]["legitimate"], "#54A24B", "External legitimate"),
        )
        for offset, (profile, color, label) in enumerate(profiles):
            axis.bar(x + (offset - 1) * width, [100 * profile[name]["proportion"] for name in STRUCTURES], width, color=color, label=label)
        axis.set_title(external_name.replace("_", " / ").title(), fontweight="bold")
        axis.set_xticks(x, [name.title() for name in STRUCTURES])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Legitimate structural share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("URLs_New-only balancing reduces—but cannot remove—external root shift", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(args.output_dir / "structural_shift.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "structural_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# URLs_New-Only Structure Balancing — Distribution Shift", "",
        "Structural total-variation distance (TVD) ranges from 0 to 1; lower is closer. Source-classifier ROC-AUC ranges from 0.5 (indistinguishable) to 1.0 (perfectly separable).", "",
        "| External set | Class | Structure TVD original | Structure TVD balanced | Source AUC original | Source AUC balanced | Mean abs. SMD original | Mean abs. SMD balanced |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for external_name in ("topdomain_phishtank", "tranco_openphish"):
        for class_name in ("legitimate", "phishing"):
            item = payload["external"][external_name]["classes"][class_name]
            tvd = item["structure_total_variation_distance"]
            auc = item["lexical_source_classifier_auc"]
            smd = item["mean_absolute_standardized_mean_difference_27_features"]
            lines.append(
                f"| {external_name} | {class_name} | {tvd['original_train_to_external']:.4f} | "
                f"{tvd['balanced_train_to_external']:.4f} | {auc['original_train_vs_external']['mean']:.4f} | "
                f"{auc['balanced_train_vs_external']['mean']:.4f} | {smd['original_train_to_external']:.4f} | "
                f"{smd['balanced_train_to_external']:.4f} |"
            )
    lines.extend((
        "", "Balancing changes only legitimate training proportions. All phishing training rows are retained, so phishing-side shift should remain essentially unchanged.",
        "", "A lower structural distance does not guarantee low external FPR because hostname vocabulary, collection source, TLDs, and other lexical relationships can remain separable.",
    ))
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
