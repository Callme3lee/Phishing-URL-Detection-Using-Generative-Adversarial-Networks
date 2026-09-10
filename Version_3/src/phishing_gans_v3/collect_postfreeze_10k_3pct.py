from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .collect_final_evaluation_dataset import (
    collect_forbidden,
    output_row,
    read_tranco,
    registered_domain,
    safe_structure,
    stable,
    write_rows,
)
from .common import sha256_file, write_json
from .url_data import canonicalize_url, encode_url, encoded_digest


OUTPUT_ROOT = Path("data/evaluation/postfreeze_10k_3pct_2026-09-01")
DATASET = OUTPUT_ROOT / "evaluation_urls.csv"
REPORT = OUTPUT_ROOT / "collection_report.json"
PHISHTANK = Path("Final Evaluation Dataset/raw/phishtank_online_valid.csv.gz")
TRANCO = Path("Final Evaluation Dataset/raw/tranco_top1m.csv.zip")
VOCABULARY = Path("data/processed/vocabulary.json")
FINAL_DATASET_ROOT = Path("Final Evaluation Dataset")
TOTAL = 10_000
PHISHING = 300
LEGITIMATE = TOTAL - PHISHING
SEED = 20_260_901_03


def add_identity(
    raw_url: str,
    vocabulary: dict[str, int],
    urls: set[str],
    hosts: set[str],
    hashes: set[bytes],
) -> None:
    try:
        normalized, hostname, _ = canonicalize_url(raw_url)
    except ValueError:
        return
    urls.add(normalized)
    hosts.add(hostname.lower().strip("."))
    hashes.add(encoded_digest(encode_url(normalized, vocabulary, 128)))


def main() -> None:
    if DATASET.exists() or REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen cohort under {OUTPUT_ROOT}")
    for path in (PHISHTANK, TRANCO, VOCABULARY):
        if not path.exists():
            raise FileNotFoundError(path)

    vocabulary = json.loads(VOCABULARY.read_text(encoding="utf-8"))
    forbidden_urls, forbidden_hosts, forbidden_hashes, identity_sources = collect_forbidden(
        OUTPUT_ROOT, vocabulary
    )

    # The original final dataset lives outside data/, so add it explicitly.
    for path in sorted(FINAL_DATASET_ROOT.glob("*.csv")):
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                add_identity(
                    row.get("url") or row.get("raw_url") or "",
                    vocabulary,
                    forbidden_urls,
                    forbidden_hosts,
                    forbidden_hashes,
                )
        identity_sources.append({"path": str(path), "sha256": sha256_file(path)})

    counters: Counter[str] = Counter()
    feed_rows: list[dict[str, str]] = []
    current_phishing_hosts: set[str] = set()
    current_phishing_domains: set[str] = set()
    with gzip.open(PHISHTANK, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
                continue
            try:
                _, hostname, _ = canonicalize_url(row.get("url", ""))
            except ValueError:
                counters["phishtank_invalid_source"] += 1
                continue
            current_phishing_hosts.add(hostname)
            current_phishing_domains.add(registered_domain(hostname))
            feed_rows.append(row)

    feed_rows.sort(
        key=lambda row: (
            str(row.get("verification_time", "")),
            stable(SEED, str(row.get("phish_id", ""))),
        ),
        reverse=True,
    )
    phishing_rows: list[dict[str, object]] = []
    selected_phishing_domains: set[str] = set()
    selected_urls: set[str] = set()
    selected_hashes: set[bytes] = set()
    for row in feed_rows:
        raw_url = str(row.get("url", "")).strip()
        structure = safe_structure(raw_url)
        if structure is None:
            counters["phishtank_non_page"] += 1
            continue
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url)
        except ValueError:
            counters["phishtank_invalid_url"] += 1
            continue
        domain = registered_domain(hostname)
        digest = encoded_digest(encode_url(normalized, vocabulary, 128))
        if scheme not in {"http", "https"}:
            counters["phishtank_wrong_scheme"] += 1
            continue
        if domain in selected_phishing_domains:
            counters["phishtank_domain_duplicate"] += 1
            continue
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters["phishtank_prior_project_identity"] += 1
            continue
        if normalized in selected_urls or digest in selected_hashes:
            counters["phishtank_snapshot_duplicate"] += 1
            continue
        phishing_rows.append(
            output_row(
                normalized,
                raw_url,
                1,
                "PhishTank verified-online",
                str(row.get("phish_id", "")),
                str(row.get("verification_time", "")),
                "",
                hostname,
                structure,
            )
        )
        selected_phishing_domains.add(domain)
        selected_urls.add(normalized)
        selected_hashes.add(digest)
        if len(phishing_rows) == PHISHING:
            break
    if len(phishing_rows) != PHISHING:
        raise RuntimeError(f"Only {len(phishing_rows)} eligible phishing rows; require {PHISHING}")

    tranco_rows = read_tranco(TRANCO)
    legitimate_rows: list[dict[str, object]] = []
    selected_legitimate_domains: set[str] = set()
    for rank, source_domain in sorted(tranco_rows, key=lambda item: stable(SEED + 1, item[1])):
        raw_url = f"https://{source_domain}/"
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url)
        except ValueError:
            counters["tranco_invalid_url"] += 1
            continue
        domain = registered_domain(hostname)
        digest = encoded_digest(encode_url(normalized, vocabulary, 128))
        if scheme != "https":
            continue
        if domain in current_phishing_domains or domain in selected_phishing_domains:
            counters["tranco_current_phishing_domain"] += 1
            continue
        if domain in selected_legitimate_domains:
            counters["tranco_registered_domain_duplicate"] += 1
            continue
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters["tranco_prior_project_identity"] += 1
            continue
        if normalized in selected_urls or digest in selected_hashes:
            counters["tranco_snapshot_duplicate"] += 1
            continue
        structure = safe_structure(raw_url)
        if structure is None:
            continue
        legitimate_rows.append(
            output_row(
                normalized,
                raw_url,
                0,
                "Tranco current root-domain proxy",
                str(rank),
                "",
                str(rank),
                hostname,
                structure,
            )
        )
        selected_legitimate_domains.add(domain)
        selected_urls.add(normalized)
        selected_hashes.add(digest)
        if len(legitimate_rows) == LEGITIMATE:
            break
    if len(legitimate_rows) != LEGITIMATE:
        raise RuntimeError(f"Only {len(legitimate_rows)} eligible legitimate rows; require {LEGITIMATE}")

    rows = legitimate_rows + phishing_rows
    random.Random(SEED + 2).shuffle(rows)
    domains = [str(row["registered_domain"]) for row in rows]
    urls = [str(row["url"]) for row in rows]
    encoded = [encoded_digest(encode_url(url, vocabulary, 128)) for url in urls]
    if len(rows) != TOTAL or sum(int(row["label"]) for row in rows) != PHISHING:
        raise RuntimeError("Unexpected class counts")
    if len(domains) != len(set(domains)):
        raise RuntimeError("Registered-domain duplicates remain in the cohort")
    if len(urls) != len(set(urls)) or len(encoded) != len(set(encoded)):
        raise RuntimeError("URL or encoded-identity duplicates remain in the cohort")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    write_rows(DATASET, rows)
    report = {
        "project_version": 3,
        "dataset_name": "Post-freeze 10K evaluation at 3% phishing prevalence",
        "purpose": "evaluation only with unchanged frozen thresholds and checkpoints",
        "selection_locked_before_inference": True,
        "model_inference_performed": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "rows": TOTAL,
        "legitimate_rows": LEGITIMATE,
        "phishing_rows": PHISHING,
        "phishing_prevalence": PHISHING / TOTAL,
        "source_counts": dict(Counter(str(row["source"]) for row in rows)),
        "identity_contract": {
            "prior_project_url_overlap": 0,
            "prior_project_hostname_overlap": 0,
            "prior_project_encoded_input_overlap": 0,
            "registered_domains_unique_within_cohort": True,
            "identity_source_count": len(identity_sources),
            "identity_sources": identity_sources,
        },
        "raw_sources": {
            "phishtank": {
                "reference": "https://www.phishtank.com/developer_info.php",
                "path": str(PHISHTANK),
                "sha256": sha256_file(PHISHTANK),
                "verified_online_feed_rows": len(feed_rows),
            },
            "tranco": {
                "reference": "https://tranco-list.eu/",
                "path": str(TRANCO),
                "sha256": sha256_file(TRANCO),
                "list_rows": len(tranco_rows),
            },
        },
        "filter_counts": dict(counters),
        "limitations": [
            "Tranco membership supplies a legitimate-domain proxy rather than manual page-level labels.",
            "Legitimate rows are root-domain URLs, so difficult deep legitimate paths are underrepresented.",
            "PhishTank verified-online membership supplies phishing labels at source-snapshot time; live URLs were not opened.",
            "The three-percent phishing prevalence is a declared browser-like simulation, not measured population prevalence.",
            "The cached source snapshots were collected on the same date as the original final evaluation; identities are new, but the source date is not later.",
        ],
    }
    write_json(REPORT, report)
    (OUTPUT_ROOT / "EVALUATION_NOT_RUN.txt").write_text(
        "Cohort frozen before model inference. Thresholds and checkpoints must remain unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": str(DATASET),
        "sha256": report["dataset_sha256"],
        "rows": TOTAL,
        "legitimate": LEGITIMATE,
        "phishing": PHISHING,
        "identity_overlap": report["identity_contract"],
        "model_inference_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
