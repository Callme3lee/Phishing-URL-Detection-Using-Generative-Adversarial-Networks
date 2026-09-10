from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .common import load_json, sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


DEFAULT_ROOT = Path("data/deployment/2026-08-15-precision")
WINNER = Path("artifacts/results/all_models_equal_calibration_v1/definitive_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a leak-free deployment calibration cohort.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--winner", type=Path, default=WINNER)
    parser.add_argument("--root-legitimate", type=int, default=20_000)
    parser.add_argument("--path-legitimate", type=int, default=5_000)
    parser.add_argument("--phishing", type=int, default=2_500)
    parser.add_argument("--maximum-paths-per-domain", type=int, default=60)
    parser.add_argument("--phishing-cutoff", default="2026-08-15T00:00:00+00:00")
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def registered(hostname: str) -> str:
    item = PSL_EXTRACTOR(hostname.lower().strip("."))
    return ".".join(part for part in (item.domain, item.suffix) if part) or hostname.lower().strip(".")


def structure(raw_url: str) -> str:
    parsed = urlsplit(raw_url if "://" in raw_url else "http://" + raw_url.lstrip("/"))
    if parsed.query:
        return "query"
    depth = len([part for part in (parsed.path or "").split("/") if part])
    return "root" if depth == 0 else ("shallow" if depth == 1 else "deep")


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def main() -> None:
    args = parse_args()
    output_csv = args.output_root / "calibration_urls.csv"
    report_path = args.output_root / "calibration_report.json"
    freeze_path = args.output_root / "frozen_model_manifest.json"
    if output_csv.exists() or report_path.exists() or freeze_path.exists():
        raise FileExistsError(f"Deployment calibration is already frozen at {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    winner = load_json(args.winner)["winner"]
    if winner["id"] != "experiments__fair__hybrid__gan__v1__gan__10pct__hyb__ms":
        raise RuntimeError("The expected selected model is no longer the recorded winner")
    checkpoints = []
    for item in winner["checkpoints"]:
        path = Path(item["path"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        checkpoints.append({**item, "verified_sha256": actual})

    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = 128
    forbidden_urls: set[str] = set()
    forbidden_hosts: set[str] = set()
    forbidden_domains: set[str] = set()
    forbidden_hashes: set[bytes] = set()

    def forbid(raw: str, hostname: str = "") -> None:
        try:
            normalized, parsed_host, _ = canonicalize_url(raw)
        except ValueError:
            return
        host = (hostname or parsed_host).lower()
        forbidden_urls.add(normalized)
        forbidden_hosts.add(host)
        forbidden_domains.add(registered(host))
        forbidden_hashes.add(encoded_digest(encode_url(normalized, vocabulary, maximum)))

    database = sqlite3.connect("data/processed/master.sqlite")
    for url, host in database.execute("SELECT normalized_url, hostname FROM urls"):
        forbid(url, host)
    database.close()
    identity_files: list[Path] = []
    for pattern in (
        "data/experiments/hybrid_cnn_coral_v1/processed/*manifest.csv",
        "data/calibration/*/calibration_urls.csv",
        "data/external/*/external_urls.csv",
        "data/development/benign_coverage_2026-08-11/benign_candidates.csv",
    ):
        identity_files.extend(Path().glob(pattern))
    for path in sorted(set(identity_files)):
        for row in read_csv(path):
            forbid(row.get("raw_url") or row.get("url") or "", row.get("canonical_hostname", ""))

    phishtank = args.output_root / "raw/phishtank_online_valid.csv.gz"
    tranco = args.output_root / "raw/tranco_top1m.csv.zip"
    cache = Path("data/development/benign_coverage_2026-08-11/cache")
    blocked_phishing_hosts: set[str] = set()
    phishing_candidates: list[dict[str, object]] = []
    phishing_seen_domains: set[str] = set()
    phishing_rejections: Counter = Counter()
    with gzip.open(phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for row in source_rows:
        try:
            normalized, host, _ = canonicalize_url(row.get("url", ""))
        except ValueError:
            phishing_rejections["invalid"] += 1
            continue
        blocked_phishing_hosts.add(host)
        if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
            phishing_rejections["not_verified_online"] += 1
            continue
        if row.get("verification_time", "") >= args.phishing_cutoff:
            phishing_rejections["reserved_after_cutoff"] += 1
            continue
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        domain = registered(host)
        if (normalized in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains
                or digest in forbidden_hashes or domain in phishing_seen_domains):
            phishing_rejections["identity_overlap_or_duplicate_domain"] += 1
            continue
        phishing_seen_domains.add(domain)
        phishing_candidates.append({
            "url": normalized, "label": 1, "source": "PhishTank verified online",
            "structure_category": structure(normalized), "canonical_hostname": host,
            "registered_domain": domain, "source_id": row.get("phish_id", ""),
            "source_timestamp": row.get("verification_time", ""),
        })
    phishing_candidates.sort(key=lambda row: stable(args.seed, str(row["url"])))
    phishing_rows = phishing_candidates[:args.phishing]
    if len(phishing_rows) < args.phishing:
        raise RuntimeError(f"Only {len(phishing_rows)} eligible phishing identities")

    seen_urls: set[str] = set()
    seen_hashes: set[bytes] = set()
    selected_domains: set[str] = set()
    legitimate_rows: list[dict[str, object]] = []
    legitimate_rejections: Counter = Counter()

    def accept_legitimate(raw_url: str, source: str, source_id: str, timestamp: str = "") -> bool:
        try:
            normalized, host, _ = canonicalize_url(raw_url)
        except ValueError:
            legitimate_rejections["invalid"] += 1
            return False
        domain = registered(host)
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if host in blocked_phishing_hosts:
            legitimate_rejections["current_phishtank_host"] += 1
            return False
        if (normalized in forbidden_urls or host in forbidden_hosts or domain in forbidden_domains
                or digest in forbidden_hashes):
            legitimate_rejections["prior_identity"] += 1
            return False
        if normalized in seen_urls or digest in seen_hashes:
            legitimate_rejections["cohort_duplicate"] += 1
            return False
        seen_urls.add(normalized); seen_hashes.add(digest); selected_domains.add(domain)
        legitimate_rows.append({
            "url": normalized, "label": 0, "source": source,
            "structure_category": structure(normalized), "canonical_hostname": host,
            "registered_domain": domain, "source_id": source_id,
            "source_timestamp": timestamp,
        })
        return True

    with zipfile.ZipFile(tranco) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as binary:
            tranco_rows = [(int(rank), domain.decode() if isinstance(domain, bytes) else domain)
                           for rank, domain in csv.reader(
                               (line.decode("utf-8", errors="replace") for line in binary))]
    root_count = 0
    for rank, domain in tranco_rows:
        if root_count >= args.root_legitimate:
            break
        if accept_legitimate(f"https://{domain.strip().lower()}/", "Tranco root proxy", str(rank)):
            root_count += 1

    path_candidates: list[tuple[bytes, str, dict]] = []
    for path in cache.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                raw = record.get("url", "")
                normalized, host, _ = canonicalize_url(raw)
                category = structure(raw)
            except (ValueError, json.JSONDecodeError):
                continue
            if category == "root":
                continue
            domain = registered(host)
            if domain in forbidden_domains or host in forbidden_hosts or host in blocked_phishing_hosts:
                continue
            path_candidates.append((stable(args.seed, normalized), raw, record))
    path_candidates.sort(key=lambda item: item[0])
    per_domain: Counter = Counter()
    category_caps = {"shallow": 500, "deep": 3_500, "query": 1_000}
    category_counts: Counter = Counter(row["structure_category"] for row in legitimate_rows)
    for _, raw, record in path_candidates:
        if sum(category_counts[name] for name in category_caps) >= args.path_legitimate:
            break
        try:
            _, host, _ = canonicalize_url(raw)
        except ValueError:
            continue
        domain = registered(host); category = structure(raw)
        if per_domain[domain] >= args.maximum_paths_per_domain or category_counts[category] >= category_caps[category]:
            continue
        if accept_legitimate(raw, "Common Crawl URL Index", record.get("digest", ""), record.get("timestamp", "")):
            per_domain[domain] += 1; category_counts[category] += 1
    path_count = sum(row["structure_category"] != "root" for row in legitimate_rows)
    if path_count < args.path_legitimate:
        raise RuntimeError(f"Only {path_count} eligible path URLs; require {args.path_legitimate}")

    rows = legitimate_rows + phishing_rows
    random.Random(args.seed).shuffle(rows)
    fields = ["url", "label", "source", "structure_category", "canonical_hostname",
              "registered_domain", "source_id", "source_timestamp"]
    temporary = output_csv.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output_csv)

    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "model_id": winner["id"],
        "model_display": "Weighted HYB-MS +10% MS-HardGAN-C (three-seed mean-score ensemble)",
        "original_threshold": winner["threshold"], "original_threshold_policy": winner["threshold_policy"],
        "checkpoints": checkpoints, "winner_source": str(args.winner),
        "winner_source_sha256": sha256_file(args.winner),
        "deployment_gate": {"maximum_fpr": 0.0025, "minimum_recall": 0.80},
        "selection_rule": "maximize F1 among observed thresholds satisfying FPR <= 0.25% and recall >= 80%; no final-confirmation labels may be used",
    }
    write_json(freeze_path, freeze)
    report = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "threshold calibration and false-positive analysis only",
        "rows": len(rows), "legitimate_rows": len(legitimate_rows), "phishing_rows": len(phishing_rows),
        "phishing_prevalence": len(phishing_rows) / len(rows),
        "structure_counts": dict(Counter(str(row["structure_category"]) for row in rows)),
        "legitimate_structure_counts": dict(Counter(str(row["structure_category"]) for row in legitimate_rows)),
        "unique_urls": len({str(row["url"]) for row in rows}),
        "unique_registered_domains": len({str(row["registered_domain"]) for row in rows}),
        "identity_exclusions": {"urls": len(forbidden_urls), "hosts": len(forbidden_hosts), "registered_domains": len(forbidden_domains), "encoded_inputs": len(forbidden_hashes)},
        "identity_files": [str(path) for path in sorted(set(identity_files))],
        "phishing_cutoff_exclusive": args.phishing_cutoff,
        "reserved_newer_phishtank_rows": phishing_rejections["reserved_after_cutoff"],
        "legitimate_rejections": dict(legitimate_rejections), "phishing_rejections": dict(phishing_rejections),
        "dataset": str(output_csv), "dataset_sha256": sha256_file(output_csv),
        "source_hashes": {"phishtank": sha256_file(phishtank), "tranco": sha256_file(tranco)},
        "label_limitations": [
            "Tranco membership and archived Common Crawl 200/HTML metadata are legitimate proxies, not manual page-level adjudication.",
            "The path sample is capped per registrable domain but has less domain diversity than the root sample.",
            "Calibration prevalence is methodological and is not presented as browser prevalence.",
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
