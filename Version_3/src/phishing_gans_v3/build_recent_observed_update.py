from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


NAME = "recent_observed_update_2026-08-01_to_2026-08-15_v1"
ROOT = Path("data/updates") / NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a compact recent-observation update layer for the next training cycle.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--phishtank", type=Path, default=Path("data/deployment/2026-08-15-precision/raw/phishtank_online_valid.csv.gz"))
    parser.add_argument("--tranco", type=Path, default=Path("data/deployment/2026-08-15-confirmation/raw/tranco_top1m.csv.zip"))
    parser.add_argument("--paths", type=Path, default=Path("data/development/deployment_recovery_2026-08-15/benign_candidates.csv"))
    parser.add_argument("--window-start", default="2026-08-01T00:00:00+00:00")
    parser.add_argument("--window-end", default="2026-08-15T00:00:00+00:00")
    parser.add_argument("--path-fraction", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=2026081503)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def registered(host: str) -> str:
    value = PSL_EXTRACTOR(host.lower())
    return ".".join(part for part in (value.domain, value.suffix) if part) or host.lower()


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def structure(url: str) -> str:
    from urllib.parse import urlsplit
    parsed = urlsplit(url if "://" in url else "http://" + url.lstrip("/"))
    if parsed.query:
        return "query"
    depth = len([part for part in parsed.path.split("/") if part])
    return "root" if depth == 0 else ("shallow" if depth == 1 else "deep")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args(); dataset = args.output_root / "updated_samples.csv"
    if args.output_root.exists():
        raise FileExistsError(f"Recent update is already frozen at {args.output_root}")
    args.output_root.mkdir(parents=True)
    vocabulary = load_json(Path("data/processed/vocabulary.json")); maximum = 128

    base_urls: set[str] = set(); base_hosts: set[str] = set(); base_domains: set[str] = set(); base_hashes: set[bytes] = set()
    base_paths = [Path("data/experiments/hybrid_cnn_coral_v1/processed/train_manifest.csv"),
                  Path("data/experiments/hybrid_cnn_coral_v1/processed/domain_validation_manifest.csv")]
    for path in base_paths:
        for row in read_csv(path):
            url = row["url"]; host = row["canonical_hostname"]; base_urls.add(url); base_hosts.add(host)
            base_domains.add(registered(host)); base_hashes.add(encoded_digest(encode_url(url, vocabulary, maximum)))

    failed_confirmation = read_csv(Path("data/deployment/2026-08-15-confirmation/confirmation_urls.csv"))
    holdout_urls = {row["url"] for row in failed_confirmation}
    holdout_hosts = {row["canonical_hostname"] for row in failed_confirmation}
    holdout_domains = {row["registered_domain"] for row in failed_confirmation}
    holdout_hashes = {encoded_digest(encode_url(row["url"], vocabulary, maximum)) for row in failed_confirmation}

    retired_domains: set[str] = set()
    retired_paths = []
    for pattern in ("data/calibration/*/calibration_urls.csv", "data/deployment/2026-08-15-precision/calibration_urls.csv",
                    "data/external/*/external_urls.csv"):
        retired_paths.extend(Path().glob(pattern))
    for path in retired_paths:
        for row in read_csv(path):
            try: host = row.get("canonical_hostname") or canonicalize_url(row.get("raw_url") or row.get("url") or "")[1]
            except ValueError: continue
            retired_domains.add(registered(host))
    recovery_domains = {registered(row["canonical_hostname"]) for row in
                        read_csv(Path("data/experiments/deployment_recovery_v1/gan_10pct/processed/train_manifest.csv"))
                        if row.get("source_group") == "recovery_phishtank"}

    fields = ["url", "label", "source", "source_id", "source_record_time", "observation_window",
              "recency_basis", "structure_category", "canonical_hostname", "registered_domain",
              "prior_experiment_exposure", "raw_url"]
    phishing: list[dict[str, object]] = []; phish_seen: set[str] = set(); rejects: Counter = Counter()
    with gzip.open(args.phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        source_phish = list(csv.DictReader(handle))
    source_phish.sort(key=lambda row: stable(args.seed, row.get("url", "")))
    for row in source_phish:
        timestamp = row.get("verification_time", "")
        if not (args.window_start <= timestamp < args.window_end):
            continue
        if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
            rejects["phishing_not_verified_online"] += 1; continue
        raw = row.get("url", "")
        try: url, host, _ = canonicalize_url(raw)
        except ValueError: rejects["phishing_invalid"] += 1; continue
        domain = registered(host); digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if url in base_urls or host in base_hosts or domain in base_domains or digest in base_hashes:
            rejects["phishing_original_training_identity"] += 1; continue
        if url in holdout_urls or host in holdout_hosts or domain in holdout_domains or digest in holdout_hashes:
            rejects["phishing_failed_confirmation_identity"] += 1; continue
        if domain in phish_seen: rejects["phishing_duplicate_domain"] += 1; continue
        phish_seen.add(domain)
        exposures = []
        if domain in retired_domains: exposures.append("retired_pre_2026-08-15_evaluation")
        if domain in recovery_domains: exposures.append("deployment_recovery_v1_development")
        phishing.append({"url": url, "label": 1, "source": "PhishTank verified-online snapshot",
                         "source_id": row.get("phish_id", ""), "source_record_time": timestamp,
                         "observation_window": "2026-08-01 through 2026-08-14 UTC",
                         "recency_basis": "PhishTank verification timestamp", "structure_category": structure(url),
                         "canonical_hostname": host, "registered_domain": domain,
                         "prior_experiment_exposure": ";".join(exposures) or "none", "raw_url": raw})
    if len(phishing) < 1_000:
        raise RuntimeError(f"Only {len(phishing)} recent base-disjoint phishing domains")

    target = len(phishing); path_target = round(target * args.path_fraction); root_target = target - path_target
    current_phish_domains: set[str] = set()
    for row in source_phish:
        try:
            current_phish_domains.add(registered(canonicalize_url(row.get("url", ""))[1]))
        except ValueError:
            continue
    legitimate: list[dict[str, object]] = []; legit_seen: set[str] = set()

    def accept_legitimate(raw: str, source: str, source_id: str, source_time: str, basis: str) -> bool:
        try: url, host, _ = canonicalize_url(raw)
        except ValueError: rejects["legitimate_invalid"] += 1; return False
        domain = registered(host); digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if domain in current_phish_domains:
            rejects["legitimate_current_phishtank_domain"] += 1; return False
        if url in base_urls or host in base_hosts or domain in base_domains or digest in base_hashes:
            rejects["legitimate_original_training_identity"] += 1; return False
        if url in holdout_urls or host in holdout_hosts or domain in holdout_domains or digest in holdout_hashes:
            rejects["legitimate_failed_confirmation_identity"] += 1; return False
        if domain in legit_seen or domain in phish_seen:
            rejects["legitimate_duplicate_domain"] += 1; return False
        legit_seen.add(domain)
        legitimate.append({"url": url, "label": 0, "source": source, "source_id": source_id,
                           "source_record_time": source_time, "observation_window": "snapshot collected 2026-08-15 UTC",
                           "recency_basis": basis, "structure_category": structure(url),
                           "canonical_hostname": host, "registered_domain": domain,
                           "prior_experiment_exposure": "retired_pre_2026-08-15_evaluation" if domain in retired_domains else "none",
                           "raw_url": raw})
        return True

    path_rows = read_csv(args.paths); path_rows.sort(key=lambda row: stable(args.seed + 1, row["url"]))
    per_structure = Counter(); structure_cap = {"shallow": path_target // 3, "deep": path_target // 3,
                                               "query": path_target - 2*(path_target//3)}
    for row in path_rows:
        category = row["structure_category"]
        if category == "root" or per_structure[category] >= structure_cap[category]: continue
        before = len(legitimate)
        if accept_legitimate(row["raw_url"], "Common Crawl URL Index", row.get("common_crawl_digest", ""),
                             row.get("crawl_timestamp", ""), "collected 2026-08-15; archived crawl timestamp retained"):
            per_structure[category] += 1
        if sum(per_structure.values()) >= path_target: break
    actual_paths = sum(per_structure.values())
    root_needed = target - actual_paths
    with zipfile.ZipFile(args.tranco) as archive:
        member = archive.namelist()[0]
        for rank, domain in csv.reader((line.decode("utf-8", errors="replace") for line in archive.open(member))):
            if len(legitimate) >= target: break
            accept_legitimate(f"https://{domain.strip().lower()}/", "Tranco root proxy", rank,
                               "2026-08-15", "present in Tranco snapshot downloaded 2026-08-15")
    if len(legitimate) < target:
        raise RuntimeError(f"Only {len(legitimate)} legitimate rows for {target} phishing rows")
    legitimate = legitimate[:target]
    all_rows = legitimate + phishing; random.Random(args.seed).shuffle(all_rows)
    write_csv(dataset, all_rows, fields)
    write_csv(args.output_root / "recent_phishing.csv", phishing, fields)
    write_csv(args.output_root / "recent_legitimate.csv", legitimate, fields)

    report = {
        "dataset_name": NAME, "project_version": 3, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "small recent-observation update layer to append to the original real training foundation",
        "rows": len(all_rows), "phishing_rows": len(phishing), "legitimate_rows": len(legitimate),
        "label_mapping": {"legitimate": 0, "phishing": 1},
        "phishing_window": {"start_inclusive": args.window_start, "end_exclusive": args.window_end},
        "structure_counts": dict(Counter(str(row["structure_category"]) for row in all_rows)),
        "legitimate_structure_counts": dict(Counter(str(row["structure_category"]) for row in legitimate)),
        "unique_urls": len({str(row["url"]) for row in all_rows}),
        "unique_registered_domains": len({str(row["registered_domain"]) for row in all_rows}),
        "original_training_overlap": {"urls": 0, "hostnames": 0, "registered_domains": 0, "encoded_inputs": 0},
        "failed_confirmation_overlap": {"urls": 0, "hostnames": 0, "registered_domains": 0, "encoded_inputs": 0},
        "prior_exposure_counts": dict(Counter(str(row["prior_experiment_exposure"]) for row in all_rows)),
        "retirement_rule": "Any earlier calibration/external cohort represented here is retired and may not evaluate models trained with this update.",
        "protected_holdout_rule": "The failed 2026-08-15 confirmation cohort remains fully excluded and is not converted into training data.",
        "files": {name: {"path": str(args.output_root / name), "sha256": sha256_file(args.output_root / name)}
                  for name in ("updated_samples.csv", "recent_phishing.csv", "recent_legitimate.csv")},
        "source_hashes": {"phishtank": sha256_file(args.phishtank), "tranco": sha256_file(args.tranco), "common_crawl_candidates": sha256_file(args.paths)},
        "filter_rejections": dict(rejects),
        "limitations": [
            "PhishTank timestamps are genuine verification times, but the snapshot includes only URLs still online when downloaded.",
            "Tranco rows are recently observed ranking entries, not proof that domains were created during the two-week window.",
            "Common Crawl path rows retain their older crawl timestamps and are structural supplements, not claimed as August-created URLs.",
            "Several phishing identities occurred in retired exploratory evaluation cohorts; those old metrics cannot evaluate future models trained on this update.",
        ],
    }
    write_json(args.output_root / "collection_report.json", report)
    readme = f"""# {NAME}\n\nThis frozen update contains {len(all_rows):,} URLs: {len(phishing):,} PhishTank URLs verified from 2026-08-01 through 2026-08-14 UTC and {len(legitimate):,} matched legitimate proxies observed in the 2026-08-15 Tranco/Common Crawl collection.\n\nIt is an update layer, not a replacement dataset. Future fair experiments must append the exact same rows to both real-only HYB-MS and HYB-MS +10% GAN conditions. The original training and failed confirmation identities are excluded. Earlier exploratory cohorts represented in this update are retired and cannot be reused for evaluation.\n\nSee `collection_report.json` for hashes, provenance, overlap contracts, and limitations.\n"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
