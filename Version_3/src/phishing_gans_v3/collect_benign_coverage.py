from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import json
import random
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .common import load_json, sha256_file, write_json
from .url_data import canonicalize_url, encode_url, encoded_digest, split_for_hostname


STRUCTURES = ("root", "shallow", "deep", "query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a structured benign-coverage development dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--tranco-zip", type=Path, default=Path("data/development/benign_coverage_2026-08-11/raw/tranco_Q2XX4_top1m.csv.zip"))
    parser.add_argument("--phishtank", type=Path, default=Path("data/external/2026-08-11/raw/phishtank_online_valid.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/development/benign_coverage_2026-08-11"))
    parser.add_argument("--common-crawl-index", default="CC-MAIN-2026-30")
    parser.add_argument("--tranco-list-id", default="Q2XX4")
    parser.add_argument("--target-per-category", type=int, default=2500)
    parser.add_argument("--reserved-root-domains", type=int, default=2500)
    parser.add_argument("--max-index-domains", type=int, default=1500)
    parser.add_argument("--max-per-domain-category", type=int, default=5)
    parser.add_argument("--index-limit", type=int, default=250)
    parser.add_argument("--minimum-rank", type=int, default=1001)
    parser.add_argument("--maximum-rank", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--request-delay", type=float, default=0.45)
    return parser.parse_args()


def structure(raw_url: str) -> str:
    parsed = urlsplit(raw_url)
    if parsed.query:
        return "query"
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if not segments:
        return "root"
    if len(segments) == 1:
        return "shallow"
    return "deep"


def load_tranco(path: Path, minimum: int, maximum: int) -> list[tuple[int, str]]:
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as binary:
            rows = csv.reader((line.decode("utf-8", errors="replace") for line in binary))
            return [(int(rank), domain.strip().lower()) for rank, domain in rows if minimum <= int(rank) <= maximum]


def load_prior_hosts() -> set[str]:
    hosts: set[str] = set()
    for root in (Path("data/external"), Path("../Version_2/data/external")):
        for path in root.glob("*/external_urls.csv"):
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    host = row.get("canonical_hostname", "").lower()
                    if host:
                        hosts.add(host)
    return hosts


def load_phishing_blocklist(path: Path) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hosts: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                normalized, hostname, _ = canonicalize_url(row.get("url", ""))
            except ValueError:
                continue
            urls.add(normalized)
            hosts.add(hostname)
    return urls, hosts


def hostname_suffixes(hosts: set[str]) -> set[str]:
    suffixes: set[str] = set()
    for host in hosts:
        labels = host.split(".")
        for index in range(max(0, len(labels) - 4), max(0, len(labels) - 1)):
            suffixes.add(".".join(labels[index:]))
    return suffixes


def cache_name(rank: int, domain: str) -> str:
    digest = hashlib.blake2b(domain.encode("utf-8"), digest_size=6).hexdigest()
    return f"{rank:07d}_{digest}.jsonl"


def query_index(index: str, domain: str, limit: int, cache: Path, delay: float,
                counters: Counter) -> list[dict]:
    if cache.exists():
        counters["common_crawl_cache_hits"] += 1
        return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    endpoint = f"https://index.commoncrawl.org/{index}-index"
    query = urllib.parse.urlencode([
        ("url", f"{domain}/*"), ("output", "json"), ("filter", "status:200"),
        ("filter", "mime:text/html"), ("filter", r"~urlkey:.*\)/.+"),
        ("limit", str(limit)),
    ])
    request = urllib.request.Request(
        f"{endpoint}?{query}", headers={"User-Agent": "phishing-gans-v3-research/1.0"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                text = response.read().decode("utf-8", errors="replace")
            cache.write_text(text, encoding="utf-8")
            counters["common_crawl_requests"] += 1
            time.sleep(delay)
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except urllib.error.HTTPError as error:
            counters[f"common_crawl_http_{error.code}"] += 1
            if error.code == 404:
                cache.write_text("", encoding="utf-8")
                return []
            time.sleep(min(20.0, 2.0 ** attempt))
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
            counters["common_crawl_network_error"] += 1
            time.sleep(min(20.0, 2.0 ** attempt))
    counters["common_crawl_failed_domains"] += 1
    return []


def main() -> None:
    args = parse_args()
    output_csv = args.output_dir / "benign_candidates.csv"
    output_report = args.output_dir / "collection_report.json"
    if output_csv.exists() or output_report.exists():
        raise FileExistsError(f"Benign coverage dataset is already frozen at {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    project = load_json(args.config)
    config = project["data"]
    vocabulary = load_json(Path("data/processed/vocabulary.json"))
    maximum = int(config["maximum_encoded_length"])
    database = sqlite3.connect("data/processed/master.sqlite")
    master_hosts = {row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls")}
    master_phishing_urls = {row[0] for row in database.execute("SELECT normalized_url FROM urls WHERE label=1")}
    master_phishing_hosts = {row[0] for row in database.execute("SELECT DISTINCT hostname FROM urls WHERE label=1")}
    prior_hosts = load_prior_hosts()
    phishtank_urls, phishtank_hosts = load_phishing_blocklist(args.phishtank)
    blocked_urls = master_phishing_urls | phishtank_urls
    blocked_hosts = master_phishing_hosts | phishtank_hosts
    master_domain_suffixes = hostname_suffixes(master_hosts)
    prior_domain_suffixes = hostname_suffixes(prior_hosts)
    blocked_domain_suffixes = hostname_suffixes(blocked_hosts)
    tranco_rows = load_tranco(args.tranco_zip, args.minimum_rank, args.maximum_rank)
    random.Random(args.seed).shuffle(tranco_rows)
    counters: Counter = Counter()
    selected: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hashes: set[bytes] = set()
    root_domains: set[str] = set()
    index_domains: set[str] = set()
    category_counts: Counter = Counter()

    def accept(raw_url: str, category: str, rank: int, domain: str, source: str,
               crawl_timestamp: str, record_digest: str) -> bool:
        counters[f"candidate::{category}"] += 1
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url, int(config["maximum_raw_url_length"]))
        except ValueError as error:
            counters[f"rejected::invalid::{error}"] += 1
            return False
        if scheme not in set(config["allowed_schemes"]):
            counters["rejected::unsupported_scheme"] += 1
            return False
        if hostname not in {domain, f"www.{domain}"}:
            counters["rejected::noncanonical_subdomain"] += 1
            return False
        digest = encoded_digest(encode_url(normalized, vocabulary, maximum))
        if normalized in seen_urls or digest in seen_hashes:
            counters["rejected::dataset_duplicate"] += 1
            return False
        if normalized in blocked_urls or hostname in blocked_hosts:
            counters["rejected::phishing_blocklist"] += 1
            return False
        if hostname in prior_hosts:
            counters["rejected::prior_external_hostname"] += 1
            return False
        if database.execute("SELECT 1 FROM urls WHERE normalized_url=?", (normalized,)).fetchone():
            counters["rejected::master_exact_overlap"] += 1
            return False
        if database.execute("SELECT 1 FROM encoded_rows WHERE encoded_hash=?", (digest,)).fetchone():
            counters["rejected::master_encoded_overlap"] += 1
            return False
        if hostname in master_hosts:
            counters["rejected::master_hostname_overlap"] += 1
            return False
        if category_counts[category] >= args.target_per_category:
            return False
        seen_urls.add(normalized)
        seen_hashes.add(digest)
        category_counts[category] += 1
        selected.append({
            "url": normalized, "label": 0, "label_semantics": "popular-domain legitimate proxy",
            "structure_category": category, "source": source, "tranco_rank": rank,
            "tranco_domain": domain, "tranco_list_id": args.tranco_list_id,
            "common_crawl_index": args.common_crawl_index if source == "Common Crawl URL Index" else "",
            "crawl_timestamp": crawl_timestamp, "common_crawl_digest": record_digest,
            "canonical_hostname": hostname,
            "proposed_split": split_for_hostname(domain, int(project["seed"]), 0.8, 0.1),
            "raw_url": raw_url,
        })
        return True

    # Root-domain coverage is supplied directly by the frozen ranking. These are
    # explicitly proxies, not claims that every possible path under a domain is safe.
    for rank, domain in tranco_rows:
        if len(root_domains) >= args.reserved_root_domains:
            break
        if domain in root_domains or domain in prior_domain_suffixes or domain in master_domain_suffixes or domain in blocked_domain_suffixes:
            continue
        root_domains.add(domain)
        if category_counts["root"] < args.target_per_category:
            accept(f"https://{domain}/", "root", rank, domain, "Tranco root proxy", "", "")

    # Use different pay-level domains for archived non-root pages so the root and
    # path buckets do not become repeated variants from the same small host set.
    queried = 0
    for rank, domain in tranco_rows:
        if all(category_counts[name] >= args.target_per_category for name in ("shallow", "deep", "query")):
            break
        if queried >= args.max_index_domains:
            break
        if domain in root_domains or domain in index_domains or domain in prior_domain_suffixes or domain in blocked_domain_suffixes:
            continue
        if domain in master_domain_suffixes:
            continue
        queried += 1
        index_domains.add(domain)
        records = query_index(
            args.common_crawl_index, domain, args.index_limit,
            cache_dir / cache_name(rank, domain), args.request_delay, counters,
        )
        per_domain = Counter()
        # Later captures first, then one canonical URL/digest is accepted once.
        records.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
        for record in records:
            raw_url = record.get("url", "")
            category = structure(raw_url)
            if category == "root" or category_counts[category] >= args.target_per_category:
                continue
            if per_domain[category] >= args.max_per_domain_category:
                continue
            if accept(raw_url, category, rank, domain, "Common Crawl URL Index",
                      record.get("timestamp", ""), record.get("digest", "")):
                per_domain[category] += 1
        if queried % 10 == 0:
            print(json.dumps({"queried_domains": queried, "counts": dict(category_counts), "requests": counters["common_crawl_requests"]}), flush=True)

    selected.sort(key=lambda row: (STRUCTURES.index(str(row["structure_category"])), int(row["tranco_rank"]), str(row["url"])))
    complete = all(category_counts[name] >= args.target_per_category for name in STRUCTURES)
    written_csv = output_csv if complete else args.output_dir / "partial_benign_candidates.csv"
    written_report = output_report if complete else args.output_dir / "partial_collection_report.json"
    with written_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "project_version": 3, "purpose": "candidate real-data coverage expansion; not external evaluation",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
        "target_per_category": args.target_per_category, "rows": len(selected),
        "reserved_root_domains": args.reserved_root_domains,
        "category_counts": {name: category_counts[name] for name in STRUCTURES},
        "unique_urls": len(seen_urls), "unique_encoded_rows": len(seen_hashes),
        "unique_hostnames": len({str(row["canonical_hostname"]) for row in selected}),
        "queried_common_crawl_domains": queried, "filter_counters": dict(counters),
        "dataset": str(written_csv), "dataset_sha256": sha256_file(written_csv),
        "sources": {
            "tranco": {"list_id": args.tranco_list_id, "reference": f"https://tranco-list.eu/list/{args.tranco_list_id}",
                       "path": str(args.tranco_zip), "sha256": sha256_file(args.tranco_zip),
                       "rank_range": [args.minimum_rank, args.maximum_rank]},
            "common_crawl": {"index": args.common_crawl_index,
                             "reference": f"https://index.commoncrawl.org/{args.common_crawl_index}-index",
                             "status_filter": 200, "mime_filter": "text/html"},
            "phishing_screen": {"phishtank_path": str(args.phishtank),
                                "phishtank_sha256": sha256_file(args.phishtank),
                                "plus_version_3_phishing_training_rows": True},
        },
        "overlap_contract": "candidate URL, 128-token encoding, and hostname excluded if present in Version 3; earlier external hostnames also excluded",
        "label_note": "Rows are popular-domain legitimate proxies. Blocklist absence and archived HTTP 200 HTML status do not prove that every URL is harmless.",
        "limitations": [
            "Tranco membership and Common Crawl HTTP 200 status are proxy labels, not manual adjudication.",
            "Blocklists have incomplete coverage and can miss compromised popular-domain pages.",
            "Only the pay-level domain and www hostname are accepted, reducing but not eliminating shared-hosting risk.",
            "This dataset must be audited and split by hostname before any retraining.",
        ],
    }
    write_json(written_report, report)
    database.close()
    print(json.dumps(report, indent=2))
    if not complete:
        raise RuntimeError(f"Collection did not reach every structural target: {dict(category_counts)}")


if __name__ == "__main__":
    main()
