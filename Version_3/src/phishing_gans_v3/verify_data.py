from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import load_json, resolve_from_config, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the immutable Version 3 prepared dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def manifest_hosts(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["canonical_hostname"] for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    project = load_json(args.config)
    config = project["data"]
    report = load_json(args.data_dir / "preparation_report.json")
    raw = resolve_from_config(args.config.resolve(), config["master_csv"])
    checks: dict[str, bool] = {
        "master_hash_matches": sha256_file(raw) == config["master_sha256"] == report["master_sha256"],
        "project_version": report["project_version"] == 3,
        "scheme_removed": report["scheme_removed_from_all_model_inputs"] is True,
        "scheme_feature_removed": report["scheme_feature_removed"] is True,
        "no_version_2_rows_imported": report["version_2_rows_imported"] == 0,
        "encoded_rows_match_splits": report["encoded_audit"]["retained_encoded_unique"] == sum(
            details["total"] for details in report["splits"].values()
        ),
        "quarantine_hash": sha256_file(args.data_dir / "quarantine.csv") == report["quarantine_sha256"],
    }
    hosts = {}
    for split in ("train", "validation", "test"):
        details = report["splits"][split]
        tokens = np.load(args.data_dir / f"{split}_tokens.npy", mmap_mode="r")
        labels = np.load(args.data_dir / f"{split}_labels.npy", mmap_mode="r")
        malicious = np.load(args.data_dir / f"{split}_malicious_indices.npy", mmap_mode="r")
        checks[f"{split}_token_shape"] = list(tokens.shape) == [details["total"], config["maximum_encoded_length"]]
        checks[f"{split}_label_shape"] = labels.shape == (details["total"],)
        checks[f"{split}_label_counts"] = int((labels == 0).sum()) == details["label_0"] and int((labels == 1).sum()) == details["label_1"]
        checks[f"{split}_malicious_indices"] = bool(np.all(labels[malicious] == 1)) and len(malicious) == details["label_1"]
        checks[f"{split}_manifest_hash"] = sha256_file(args.data_dir / f"{split}_manifest.csv") == details["manifest_sha256"]
        hosts[split] = manifest_hosts(args.data_dir / f"{split}_manifest.csv")
        checks[f"{split}_all_urls_scheme_free"] = True
        with (args.data_dir / f"{split}_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["url"].lower().startswith(("http://", "https://", "ftp://")):
                    checks[f"{split}_all_urls_scheme_free"] = False
                    break
    checks["train_validation_hosts_disjoint"] = hosts["train"].isdisjoint(hosts["validation"])
    checks["train_test_hosts_disjoint"] = hosts["train"].isdisjoint(hosts["test"])
    checks["validation_test_hosts_disjoint"] = hosts["validation"].isdisjoint(hosts["test"])
    payload = {"passed": all(checks.values()), "checks": checks}
    write_json(args.data_dir / "verification_report.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"Version 3 verification failed: {payload}")
    print(payload)


if __name__ == "__main__":
    main()
