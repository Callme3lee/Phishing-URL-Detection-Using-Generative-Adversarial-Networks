from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sqlite3
import time
import urllib.parse
import urllib.request
import urllib.error
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


DEFAULT_ROOT = Path("data/external/2026-08-20-long-url-raw")
CRAWL = "CC-MAIN-2026-30"
USER_AGENT = "GraduationResearch/Version3 long-URL evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a new long-legitimate-URL external evaluation set.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--legitimate-target", type=int, default=1400)
    parser.add_argument("--phishing-target", type=int, default=150)
    parser.add_argument("--domain-query-budget", type=int, default=180)
    parser.add_argument("--maximum-per-domain", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--request-delay", type=float, default=0.35)
    parser.add_argument("--allow-prior-phishing-registered-domains", action="store_true",
                        help="Keep URL/host/encoding disjointness but permit a new phishing host under a familiar registered domain.")
    return parser.parse_args()


def registered(host: str) -> str:
    item = PSL_EXTRACTOR(host.lower().strip("."))
    return ".".join(value for value in (item.domain, item.suffix) if value) or host.lower()


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def collect_known_identities(vocabulary: dict[str, int], root: Path):
    urls: set[str] = set(); hosts: set[str] = set(); domains: set[str] = set(); encodings: set[bytes] = set()

    def add(raw: str, supplied_host: str = "") -> None:
        try: normalized, parsed_host, _ = canonicalize_url(raw)
        except ValueError: return
        host = (supplied_host or parsed_host).lower().strip(".")
        urls.add(normalized); hosts.add(host); domains.add(registered(host))
        encodings.add(encoded_digest(encode_url(normalized, vocabulary, 128)))

    database = sqlite3.connect("data/processed/master.sqlite")
    for url, host in database.execute("SELECT normalized_url, hostname FROM urls"):
        add(url, host)
    database.close()
    patterns = (
        "data/experiments/hybrid_cnn_coral_v1/processed/*manifest.csv",
        "data/experiments/recent_update_gan10_v1/*/processed/train_manifest.csv",
        "data/calibration/**/*.csv", "data/external/**/*.csv", "data/deployment/**/*.csv",
        "data/development/**/*.csv", "data/updates/**/*.csv", "data/evaluation/**/*.csv",
    )
    paths: set[Path] = set()
    for pattern in patterns: paths.update(Path().glob(pattern))
    paths = {path for path in paths if root not in path.parents}
    for path in sorted(paths):
        try:
            for row in rows(path): add(row.get("raw_url") or row.get("url") or "", row.get("canonical_hostname", ""))
        except (OSError, csv.Error):
            continue
    return urls, hosts, domains, encodings, sorted(paths)


def read_tranco(path: Path):
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        stream = (line.decode("utf-8", errors="replace") for line in archive.open(member))
        yield from csv.reader(stream)


def fetch_commoncrawl(domain: str, cache: Path, delay: float) -> tuple[list[dict], str]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{domain.replace('.', '_')}.jsonl"
    if path.exists():
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        query = urllib.parse.urlencode({
            "url": f"{domain}/*", "output": "json", "filter": ["status:200", "mime:text/html"],
            "limit": "500",
        }, doseq=True)
        request = urllib.request.Request(
            f"https://index.commoncrawl.org/{CRAWL}-index?{query}", headers={"User-Agent": USER_AGENT},
        )
        error = ""
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    content = response.read().decode("utf-8", errors="replace")
                path.write_text(content, encoding="utf-8"); break
            except urllib.error.HTTPError as exc:
                if exc.code in {400, 404}:
                    return [], f"HTTPError: {exc.code}"
                error = f"HTTPError: {exc.code}"
                time.sleep(1 + attempt)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(1 + attempt)
        else:
            return [], error
        time.sleep(delay)
    output = []
    for line in content.splitlines():
        try: output.append(json.loads(line))
        except json.JSONDecodeError: continue
    return output, ""


def structure(url: str) -> str:
    parsed = urllib.parse.urlsplit("http://" + url.lstrip("/"))
    if parsed.query: return "query"
    segments = [part for part in parsed.path.split("/") if part]
    return "deep_path" if len(segments) >= 3 else "shallow_path"


def main() -> None:
    args = parse_args(); output = args.root / "evaluation_urls.csv"; report_path = args.root / "collection_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Snapshot already frozen at {args.root}")
    raw = args.root / "raw"; cache = raw / "commoncrawl_cache"
    required = {
        "phishtank": raw / "phishtank_online_valid.csv.gz",
        "openphish": raw / "openphish_feed.txt",
        "tranco": raw / "tranco_top1m.csv.zip",
        "commoncrawl_collections": raw / "commoncrawl_collinfo.json",
    }
    for path in required.values():
        if not path.exists(): raise FileNotFoundError(path)
    vocabulary = json.loads(Path("data/processed/vocabulary.json").read_text())
    forbidden_urls, forbidden_hosts, forbidden_domains, forbidden_encodings, identity_paths = collect_known_identities(vocabulary, args.root)
    counters: Counter = Counter(); domain_ranks = []
    for rank, value in read_tranco(required["tranco"]):
        domain = value.strip().lower()
        if not domain: continue
        domain_ranks.append((int(rank), domain))
        if len(domain_ranks) >= max(5000, args.domain_query_budget * 20): break
    domain_ranks.sort(key=lambda item: stable(args.seed, item[1]))
    selected_domains = domain_ranks[:args.domain_query_budget]
    legitimate_candidates = []; seen_legit_urls: set[str] = set(); per_domain: defaultdict[str, int] = defaultdict(int)
    request_errors = {}
    for query_index, (rank, domain) in enumerate(selected_domains, 1):
        records, error = fetch_commoncrawl(domain, cache, args.request_delay)
        if error: request_errors[domain] = error; counters["commoncrawl_request_error"] += 1; continue
        shuffled = sorted(records, key=lambda row: stable(args.seed, str(row.get("url", ""))))
        for record in shuffled:
            raw_url = str(record.get("url", "")).strip()
            try: url, host, scheme = canonicalize_url(raw_url)
            except ValueError: counters["invalid_legitimate"] += 1; continue
            domain_value = registered(host); digest = encoded_digest(encode_url(url, vocabulary, 128))
            if scheme not in {"http", "https"} or domain_value != registered(domain):
                counters["wrong_domain_or_scheme"] += 1; continue
            if len(url) < 80:
                counters["legitimate_not_long"] += 1; continue
            if structure(url) not in {"query", "deep_path"}:
                counters["legitimate_not_complex"] += 1; continue
            if url in forbidden_urls or digest in forbidden_encodings or url in seen_legit_urls:
                counters["prior_or_duplicate_legitimate_identity"] += 1; continue
            if per_domain[domain_value] >= args.maximum_per_domain:
                counters["per_domain_cap"] += 1; continue
            seen_legit_urls.add(url); per_domain[domain_value] += 1
            legitimate_candidates.append({
                "url": url, "label": 0, "source": f"Tranco-domain {CRAWL} 200/HTML long URL proxy",
                "canonical_hostname": host, "registered_domain": domain_value, "source_id": str(record.get("digest", "")),
                "source_timestamp": str(record.get("timestamp", "")), "tranco_rank": rank,
                "url_length": len(url), "structure": structure(url),
                "prior_hostname_seen": int(host in forbidden_hosts),
                "prior_registered_domain_seen": int(domain_value in forbidden_domains),
            })
        print(json.dumps({"stage": "commoncrawl", "domain": query_index, "budget": len(selected_domains),
                          "accepted_candidates": len(legitimate_candidates)}), flush=True)
        if len(legitimate_candidates) >= args.legitimate_target: break
    legitimate_candidates.sort(key=lambda row: stable(args.seed, row["url"]))
    legitimate = legitimate_candidates[:args.legitimate_target]
    if len(legitimate) < args.legitimate_target:
        raise RuntimeError(f"Only {len(legitimate)} long legitimate candidates; require {args.legitimate_target}")

    phishing_candidates = []
    with gzip.open(required["phishtank"], "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verified") != "yes" or row.get("online") != "yes": continue
            phishing_candidates.append((row.get("url", ""), "PhishTank verified-online", row.get("phish_id", ""), row.get("verification_time", "")))
    for index, value in enumerate(required["openphish"].read_text(encoding="utf-8", errors="replace").splitlines()):
        if value.strip(): phishing_candidates.append((value.strip(), "OpenPhish community feed", str(index), "2026-08-20 collection"))
    phishing_candidates.sort(key=lambda row: stable(args.seed, row[0]))
    phishing = []; seen_phishing_domains = set(); legitimate_domains = {row["registered_domain"] for row in legitimate}
    for raw_url, source, source_id, timestamp in phishing_candidates:
        try: url, host, scheme = canonicalize_url(raw_url)
        except ValueError: counters["invalid_phishing"] += 1; continue
        domain_value = registered(host); digest = encoded_digest(encode_url(url, vocabulary, 128))
        if scheme not in {"http", "https"}: counters["unsupported_phishing_scheme"] += 1; continue
        prior_domain_blocked = domain_value in forbidden_domains and not args.allow_prior_phishing_registered_domains
        if (url in forbidden_urls or host in forbidden_hosts or prior_domain_blocked
                or digest in forbidden_encodings or domain_value in seen_phishing_domains or domain_value in legitimate_domains):
            counters["prior_or_duplicate_phishing_identity"] += 1; continue
        seen_phishing_domains.add(domain_value)
        phishing.append({
            "url": url, "label": 1, "source": source, "canonical_hostname": host,
            "registered_domain": domain_value, "source_id": source_id, "source_timestamp": timestamp,
            "tranco_rank": "", "url_length": len(url), "structure": structure(url),
            "prior_hostname_seen": 0, "prior_registered_domain_seen": 0,
        })
        if len(phishing) >= args.phishing_target: break
    if len(phishing) < args.phishing_target:
        raise RuntimeError(f"Only {len(phishing)} phishing candidates; require {args.phishing_target}")
    final_rows = legitimate + phishing; random.Random(args.seed).shuffle(final_rows)
    fields = ["url", "label", "source", "canonical_hostname", "registered_domain", "source_id",
              "source_timestamp", "tranco_rank", "url_length", "structure", "prior_hostname_seen",
              "prior_registered_domain_seen"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(final_rows)
    lengths = [int(row["url_length"]) for row in legitimate]
    report = {
        "project_version": 3, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "new raw-source stress test emphasizing long legitimate paths and queries",
        "selection_locked_before_model_inference": True, "seed": args.seed, "commoncrawl_collection": CRAWL,
        "rows": len(final_rows), "legitimate_rows": len(legitimate), "phishing_rows": len(phishing),
        "phishing_prevalence": len(phishing) / len(final_rows), "dataset": str(output), "dataset_sha256": sha256_file(output),
        "long_legitimate_definition": "scheme-free canonical URL length >=80 and either query or at least three path segments",
        "legitimate_length": {"minimum": min(lengths), "mean": sum(lengths) / len(lengths), "maximum": max(lengths)},
        "legitimate_structure": dict(Counter(row["structure"] for row in legitimate)),
        "legitimate_registered_domains": len({row["registered_domain"] for row in legitimate}),
        "legitimate_prior_hostname_overlap_rows": sum(int(row["prior_hostname_seen"]) for row in legitimate),
        "legitimate_prior_registered_domain_overlap_rows": sum(int(row["prior_registered_domain_seen"]) for row in legitimate),
        "phishing_registered_domains": len({row["registered_domain"] for row in phishing}),
        "phishing_prior_registered_domain_overlap_rows": sum(
            int(row["registered_domain"] in forbidden_domains) for row in phishing
        ),
        "phishing_identity_policy": (
            "zero exact URL, hostname, and encoded-input overlap; familiar registered domains allowed"
            if args.allow_prior_phishing_registered_domains else
            "zero exact URL, hostname, registered-domain, and encoded-input overlap"
        ),
        "identity_exclusions": {"urls": len(forbidden_urls), "hosts": len(forbidden_hosts),
                                "registered_domains": len(forbidden_domains), "encoded_inputs": len(forbidden_encodings),
                                "source_files": len(identity_paths)},
        "filter_counts": dict(counters), "commoncrawl_request_errors": request_errors,
        "raw_sources": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in required.items()},
        "limitations": [
            "PhishTank/OpenPhish feed membership supplies phishing labels; no phishing page was opened.",
            "Tranco membership plus Common Crawl 200/HTML metadata is a legitimate proxy, not manual page-level adjudication.",
            "The stress set deliberately over-represents long legitimate paths/queries and is not a population prevalence estimate.",
            "Legitimate URLs are exact-URL and encoded-input disjoint; familiar host/domain overlap is retained and reported because real users visit new pages on known sites.",
            "Common Crawl captures predate the 2026-08-20 collection date; the URL identities are new to this project, not necessarily newly created.",
        ],
    }
    write_json(report_path, report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
