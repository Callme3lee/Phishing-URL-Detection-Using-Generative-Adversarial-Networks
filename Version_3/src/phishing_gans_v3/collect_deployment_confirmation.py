from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


ROOT = Path("data/deployment/2026-08-15-confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock the post-threshold low-prevalence confirmation cohort.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--threshold-analysis", type=Path, default=Path("artifacts/results/deployment_recovery_v1/calibration/calibration_analysis.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/experiments/deployment_recovery_v1/gan_10pct/hyb_ms"))
    parser.add_argument("--seed", type=int, default=2026081502)
    return parser.parse_args()


def registered(host: str) -> str:
    item = PSL_EXTRACTOR(host)
    return ".".join(value for value in (item.domain, item.suffix) if value) or host


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def main() -> None:
    args = parse_args(); output = args.root / "confirmation_urls.csv"; report_path = args.root / "collection_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Confirmation cohort already frozen at {args.root}")
    threshold = load_json(args.threshold_analysis)
    if not threshold["deployment_gate_passed"]:
        raise RuntimeError("Recovery model did not pass the predeclared calibration gate")
    vocabulary = load_json(Path("data/processed/vocabulary.json")); maximum = 128
    forbidden_urls: set[str] = set(); forbidden_hosts: set[str] = set(); forbidden_domains: set[str] = set(); forbidden_hashes: set[bytes] = set()

    def forbid(raw: str, hostname: str = "") -> None:
        try: url, parsed_host, _ = canonicalize_url(raw)
        except ValueError: return
        host = (hostname or parsed_host).lower(); forbidden_urls.add(url); forbidden_hosts.add(host)
        forbidden_domains.add(registered(host)); forbidden_hashes.add(encoded_digest(encode_url(url, vocabulary, maximum)))

    database = sqlite3.connect("data/processed/master.sqlite")
    for url, host in database.execute("SELECT normalized_url, hostname FROM urls"): forbid(url, host)
    database.close()
    identity_paths = []
    for pattern in (
        "data/experiments/hybrid_cnn_coral_v1/processed/*manifest.csv",
        "data/experiments/deployment_recovery_v1/gan_10pct/processed/train_manifest.csv",
        "data/calibration/*/calibration_urls.csv", "data/deployment/2026-08-15-precision/calibration_urls.csv",
        "data/external/*/external_urls.csv", "data/development/*/benign_candidates.csv",
    ):
        identity_paths.extend(Path().glob(pattern))
    for path in sorted(set(identity_paths)):
        for row in read_csv(path): forbid(row.get("raw_url") or row.get("url") or "", row.get("canonical_hostname", ""))

    openphish = args.root / "raw/openphish_feed.txt"; tranco = args.root / "raw/tranco_top1m.csv.zip"
    phishing = []; phish_seen: set[str] = set(); rejects: Counter = Counter()
    candidates = [line.strip() for line in openphish.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    candidates.sort(key=lambda url: stable(args.seed, url))
    for raw in candidates:
        try: url, host, _ = canonicalize_url(raw)
        except ValueError: rejects["invalid_phishing"] += 1; continue
        domain = registered(host); digest = encoded_digest(encode_url(url, vocabulary, maximum))
        if url in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains or digest in forbidden_hashes:
            rejects["prior_phishing_identity"] += 1; continue
        if domain in phish_seen: rejects["duplicate_phishing_domain"] += 1; continue
        phish_seen.add(domain)
        phishing.append({"url": url, "label": 1, "source": "OpenPhish post-threshold feed",
                         "canonical_hostname": host, "registered_domain": domain, "source_id": ""})
    if len(phishing) < 100:
        raise RuntimeError(f"Only {len(phishing)} eligible post-threshold phishing domains")

    phishing_target = min(300, len(phishing)); phishing = phishing[:phishing_target]
    legitimate_target = phishing_target * 99
    blocked_phish_hosts = {row["canonical_hostname"] for row in phishing}
    legitimate = []; legit_seen: set[str] = set()
    with zipfile.ZipFile(tranco) as archive:
        member = archive.namelist()[0]
        rows = csv.reader((line.decode("utf-8", errors="replace") for line in archive.open(member)))
        for rank, domain_value in rows:
            if len(legitimate) >= legitimate_target: break
            raw = f"https://{domain_value.strip().lower()}/"
            try: url, host, _ = canonicalize_url(raw)
            except ValueError: rejects["invalid_legitimate"] += 1; continue
            domain = registered(host); digest = encoded_digest(encode_url(url, vocabulary, maximum))
            if (url in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains or digest in forbidden_hashes
                    or host in blocked_phish_hosts or domain in phish_seen):
                rejects["prior_or_phishing_legitimate_identity"] += 1; continue
            if domain in legit_seen: rejects["duplicate_legitimate_domain"] += 1; continue
            legit_seen.add(domain)
            legitimate.append({"url": url, "label": 0, "source": "Tranco post-threshold root proxy",
                               "canonical_hostname": host, "registered_domain": domain, "source_id": rank})
    if len(legitimate) < legitimate_target:
        raise RuntimeError(f"Only {len(legitimate)} eligible legitimate domains; require {legitimate_target}")
    rows = legitimate + phishing; random.Random(args.seed).shuffle(rows)
    fields = ["url", "label", "source", "canonical_hostname", "registered_domain", "source_id"]
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output)
    checkpoints = [{"seed": seed, "path": str(args.checkpoint_root / f"seed_{seed}" / "checkpoint.pt"),
                    "sha256": sha256_file(args.checkpoint_root / f"seed_{seed}" / "checkpoint.pt")} for seed in (42, 43, 44)]
    report = {
        "project_version": 3, "purpose": "single untouched post-threshold confirmation evaluation",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "selection_locked_before_inference": True,
        "rows": len(rows), "legitimate_rows": len(legitimate), "phishing_rows": len(phishing),
        "phishing_prevalence": len(phishing) / len(rows), "unique_registered_domains": len(rows),
        "dataset": str(output), "dataset_sha256": sha256_file(output),
        "threshold": threshold["selected_threshold"], "threshold_analysis": str(args.threshold_analysis),
        "threshold_analysis_sha256": sha256_file(args.threshold_analysis), "model_id": threshold["model_id"],
        "checkpoints": checkpoints, "raw_source_hashes": {"openphish": sha256_file(openphish), "tranco": sha256_file(tranco)},
        "source_collection_mtime_utc": {name: datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                                         for name, path in (("openphish", openphish), ("tranco", tranco))},
        "identity_exclusions": {"urls": len(forbidden_urls), "hosts": len(forbidden_hosts), "registered_domains": len(forbidden_domains), "encoded_inputs": len(forbidden_hashes)},
        "filter_rejections": dict(rejects),
        "one_shot_contract": "no model, feature, ensemble, threshold, or dataset selection change after inference",
        "limitations": [
            "OpenPhish community-feed membership is the phishing label; URLs were not manually visited.",
            "Tranco membership is a legitimate-domain proxy rather than page-level adjudication.",
            "The Tranco archive hash matches the calibration download, but every selected registered domain is disjoint and the file was downloaded again only after threshold lock.",
            "Root URLs dominate this browser-style cohort, so path-heavy traffic needs separate monitoring.",
        ],
    }
    write_json(report_path, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
