from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file, write_json
from .url_data import PSL_EXTRACTOR, canonicalize_url, encode_url, encoded_digest


DEFAULT_ROOT = Path("Browser Policy Evaluation Dataset 10K")
DEFAULT_CRAWL = "CC-MAIN-2026-34"
USER_AGENT = "AliAlganbour-GraduationResearch/Version3 final-evaluation-collection"
LOGIN_PATTERN = re.compile(
    r"(?:^|[/._?&=+-])(login|log-in|signin|sign-in|signon|sign-on|auth|authentication|sso|account|session)(?:$|[/._?&=+-])",
    re.IGNORECASE,
)
EXCLUDED_FILENAMES = {
    "robots.txt", "sitemap.xml", "sitemap_index.xml", "favicon.ico",
    "ads.txt", "security.txt", "humans.txt",
}
EXCLUDED_SUFFIXES = {
    ".css", ".js", ".json", ".xml", ".txt", ".ico", ".png", ".jpg",
    ".jpeg", ".gif", ".svg", ".webp", ".pdf", ".zip", ".gz", ".mp4",
    ".mp3", ".woff", ".woff2", ".ttf", ".map",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and freeze the final URL evaluation dataset without running inference."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--crawl", default=DEFAULT_CRAWL)
    parser.add_argument("--root-legitimate", type=int, default=8_800)
    parser.add_argument("--complex-legitimate", type=int, default=700)
    parser.add_argument("--minimum-login-urls", type=int, default=50)
    parser.add_argument("--phishing", type=int, default=500)
    parser.add_argument("--development-rows", type=int, default=4_000)
    parser.add_argument("--domain-query-budget", type=int, default=1_000)
    parser.add_argument("--maximum-complex-per-domain", type=int, default=5)
    parser.add_argument("--commoncrawl-limit", type=int, default=500)
    parser.add_argument("--request-delay", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20_260_901)
    return parser.parse_args()


def registered_domain(hostname: str) -> str:
    extracted = PSL_EXTRACTOR(hostname.lower().strip("."))
    return ".".join(value for value in (extracted.domain, extracted.suffix) if value) or hostname.lower()


def stable(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=16).digest()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def safe_structure(raw_url: str) -> dict[str, object] | None:
    try:
        parsed = urllib.parse.urlsplit(raw_url if "://" in raw_url else "http://" + raw_url.lstrip("/"))
    except ValueError:
        return None
    path = parsed.path or "/"
    filename = path.rstrip("/").rsplit("/", 1)[-1].lower()
    lower_path = path.lower()
    if filename in EXCLUDED_FILENAMES or lower_path.endswith("/robots.txt"):
        return None
    if any(lower_path.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return None
    if "logout" in lower_path or "signout" in lower_path or "sign-out" in lower_path:
        return None
    depth = len([value for value in path.split("/") if value])
    parameters = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    searchable = f"{path}?{parsed.query}".lower()
    match = LOGIN_PATTERN.search(searchable)
    return {
        "path_depth": depth,
        "query_parameter_count": len(parameters),
        "is_login_url": int(match is not None),
        "login_keyword": "" if match is None else match.group(1).lower(),
    }


def collect_forbidden(root: Path, vocabulary: dict[str, int]) -> tuple[set[str], set[str], set[bytes], list[dict[str, str]]]:
    forbidden_urls: set[str] = set()
    forbidden_hosts: set[str] = set()
    forbidden_hashes: set[bytes] = set()
    sources: list[dict[str, str]] = []

    def add(raw: str, supplied_host: str = "") -> None:
        try:
            normalized, parsed_host, _ = canonicalize_url(raw)
        except ValueError:
            return
        host = (supplied_host or parsed_host).lower().strip(".")
        forbidden_urls.add(normalized)
        forbidden_hosts.add(host)
        forbidden_hashes.add(encoded_digest(encode_url(normalized, vocabulary, 128)))

    database_path = Path("data/processed/master.sqlite")
    database = sqlite3.connect(database_path)
    for normalized, hostname in database.execute("SELECT normalized_url, hostname FROM urls"):
        add(normalized, hostname)
    database.close()
    sources.append({"path": str(database_path), "sha256": sha256_file(database_path)})

    patterns = (
        "data/processed/*manifest.csv",
        "data/experiments/*/processed/*manifest.csv",
        "data/experiments/*/*/processed/*manifest.csv",
        "data/calibration/**/*.csv",
        "data/development/**/*.csv",
        "data/deployment/**/*.csv",
        "data/evaluation/**/*.csv",
        "data/external/**/*.csv",
        "data/updates/**/*.csv",
    )
    identity_paths: set[Path] = set()
    for pattern in patterns:
        identity_paths.update(Path().glob(pattern))
    identity_paths = {
        path for path in identity_paths
        if path.is_file() and root.resolve() not in path.resolve().parents
    }
    for path in sorted(identity_paths):
        try:
            for row in csv_rows(path):
                add(row.get("raw_url") or row.get("url") or "", row.get("canonical_hostname", ""))
        except (OSError, csv.Error):
            continue
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    return forbidden_urls, forbidden_hosts, forbidden_hashes, sources


def read_tranco(path: Path) -> list[tuple[int, str]]:
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        stream = (line.decode("utf-8", errors="replace") for line in archive.open(member))
        return [(int(rank), domain.strip().lower()) for rank, domain in csv.reader(stream) if domain.strip()]


def fetch_commoncrawl(
    crawl: str, domain: str, cache: Path, limit: int, delay: float
) -> tuple[list[dict], str]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{domain.replace('.', '_')}.jsonl"
    if path.exists():
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        query = urllib.parse.urlencode(
            {
                "url": f"{domain}/*",
                "output": "json",
                "filter": ["status:200", "mime:text/html"],
                "limit": str(limit),
            },
            doseq=True,
        )
        request = urllib.request.Request(
            f"https://index.commoncrawl.org/{crawl}-index?{query}",
            headers={"User-Agent": USER_AGENT},
        )
        error = ""
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    content = response.read().decode("utf-8", errors="replace")
                path.write_text(content, encoding="utf-8")
                break
            except urllib.error.HTTPError as exc:
                if exc.code in {400, 404}:
                    return [], f"HTTPError:{exc.code}"
                error = f"HTTPError:{exc.code}"
            except Exception as exc:  # network failures are recorded in the collection report
                error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
        else:
            return [], error
        time.sleep(delay)
    records = []
    for line in content.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, ""


def output_row(
    normalized: str,
    raw_url: str,
    label: int,
    source: str,
    source_id: str,
    source_timestamp: str,
    source_rank: str,
    hostname: str,
    structure: dict[str, object],
) -> dict[str, object]:
    return {
        "url": normalized,
        "raw_url": raw_url,
        "label": label,
        "source": source,
        "source_id": source_id,
        "source_timestamp": source_timestamp,
        "source_rank": source_rank,
        "canonical_hostname": hostname,
        "registered_domain": registered_domain(hostname),
        "url_length": len(normalized),
        "path_depth": structure["path_depth"],
        "query_parameter_count": structure["query_parameter_count"],
        "is_login_url": structure["is_login_url"],
        "login_keyword": structure["login_keyword"],
        "prior_hostname_seen": 0,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.minimum_login_urls > args.complex_legitimate:
        raise ValueError("minimum login URLs cannot exceed complex legitimate target")
    total = args.root_legitimate + args.complex_legitimate + args.phishing
    if total != 10_000 or args.phishing != 500:
        raise ValueError("The registered final design is exactly 10,000 rows with 500 phishing rows")
    if args.development_rows <= 0 or args.development_rows >= total:
        raise ValueError("Invalid development split size")

    root = args.root
    raw = root / "raw"
    cache = raw / "commoncrawl_cache"
    outputs = {
        "all": root / "all_urls.csv",
        "development": root / "policy_development.csv",
        "final": root / "frozen_final_test.csv",
        "report": root / "collection_report.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"A frozen output already exists under {root}")
    required = {
        "phishtank": raw / "phishtank_online_valid.csv.gz",
        "tranco": raw / "tranco_top1m.csv.zip",
        "commoncrawl_collections": raw / "commoncrawl_collinfo.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    vocabulary = json.loads(Path("data/processed/vocabulary.json").read_text(encoding="utf-8"))
    forbidden_urls, forbidden_hosts, forbidden_hashes, identity_sources = collect_forbidden(root, vocabulary)
    counters: Counter[str] = Counter()

    phishtank_rows = []
    current_phishing_hosts: set[str] = set()
    current_phishing_domains: set[str] = set()
    with gzip.open(required["phishtank"], "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verified", "").lower() != "yes" or row.get("online", "").lower() != "yes":
                continue
            try:
                _, host, _ = canonicalize_url(row.get("url", ""))
            except ValueError:
                counters["invalid_phishtank_source"] += 1
                continue
            current_phishing_hosts.add(host)
            current_phishing_domains.add(registered_domain(host))
            phishtank_rows.append(row)

    tranco_rows = read_tranco(required["tranco"])
    candidate_domains = []
    seen_candidate_domains = set()
    tranco_rank_by_registered: dict[str, tuple[int, str]] = {}
    for rank, domain in tranco_rows[:250_000]:
        try:
            _, host, _ = canonicalize_url(f"https://{domain}/")
        except ValueError:
            continue
        registered = registered_domain(host)
        tranco_rank_by_registered.setdefault(registered, (rank, host))
        if registered in seen_candidate_domains:
            continue
        if host in current_phishing_hosts or registered in current_phishing_domains:
            continue
        seen_candidate_domains.add(registered)
        candidate_domains.append((rank, host, registered))
    candidate_domains.sort(key=lambda row: stable(args.seed, row[2]))
    candidate_domains = candidate_domains[: args.domain_query_budget]

    login_candidates: list[dict[str, object]] = []
    other_complex_candidates: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_hashes: set[bytes] = set()
    request_errors: dict[str, str] = {}
    per_domain: defaultdict[str, int] = defaultdict(int)

    def process_commoncrawl_records(
        records: list[dict], rank: int, registered: str, crawl_label: str
    ) -> None:
        # Prioritize login-like paths so the per-domain cap does not fill with
        # ordinary pages before the targeted stress examples are considered.
        ordered = sorted(records, key=lambda row: (
            0 if LOGIN_PATTERN.search(str(row.get("url", ""))) else 1,
            stable(args.seed, str(row.get("url", ""))),
        ))
        for record in ordered:
            raw_url = str(record.get("url", "")).strip()
            structure = safe_structure(raw_url)
            if structure is None:
                counters["commoncrawl_non_page"] += 1
                continue
            try:
                normalized, hostname, scheme = canonicalize_url(raw_url)
            except ValueError:
                counters["commoncrawl_invalid_url"] += 1
                continue
            domain_value = registered_domain(hostname)
            digest = encoded_digest(encode_url(normalized, vocabulary, 128))
            if scheme not in {"http", "https"} or domain_value != registered:
                counters["commoncrawl_wrong_domain_or_scheme"] += 1
                continue
            if hostname in current_phishing_hosts or domain_value in current_phishing_domains:
                counters["commoncrawl_current_phishing_identity"] += 1
                continue
            if normalized in forbidden_urls or digest in forbidden_hashes:
                counters["commoncrawl_prior_project_identity"] += 1
                continue
            if normalized in seen_urls or digest in seen_hashes:
                counters["commoncrawl_duplicate"] += 1
                continue
            is_login = bool(structure["is_login_url"])
            is_complex = len(normalized) >= 80 or int(structure["path_depth"]) >= 3 or int(structure["query_parameter_count"]) >= 2
            if not is_login and not is_complex:
                counters["commoncrawl_not_login_or_complex"] += 1
                continue
            if per_domain[domain_value] >= args.maximum_complex_per_domain:
                counters["commoncrawl_domain_cap"] += 1
                continue
            row = output_row(
                normalized, raw_url, 0, f"Tranco-domain {crawl_label} HTML capture",
                str(record.get("digest", "")), str(record.get("timestamp", "")),
                str(rank), hostname, structure,
            )
            row["prior_hostname_seen"] = int(hostname in forbidden_hosts)
            if hostname in forbidden_hosts:
                counters["commoncrawl_prior_hostname_seen"] += 1
            seen_urls.add(normalized)
            seen_hashes.add(digest)
            per_domain[domain_value] += 1
            (login_candidates if is_login else other_complex_candidates).append(row)

    # Process every successful response already cached from the latest crawl
    # before making more network requests. This keeps retries bounded and makes
    # interrupted collection runs truly resumable even if domain ordering changes.
    for path in sorted(cache.glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        grouped: defaultdict[str, list[dict]] = defaultdict(list)
        for record in records:
            try:
                _, hostname, _ = canonicalize_url(str(record.get("url", "")))
            except ValueError:
                continue
            grouped[registered_domain(hostname)].append(record)
        for registered, group in grouped.items():
            tranco_item = tranco_rank_by_registered.get(registered)
            if tranco_item is None or registered in current_phishing_domains:
                continue
            process_commoncrawl_records(group, tranco_item[0], registered, args.crawl)
    print(json.dumps({
        "stage": "commoncrawl_latest_cache",
        "login_candidates": len(login_candidates),
        "other_complex_candidates": len(other_complex_candidates),
    }), flush=True)

    # Reuse only previously unselected identities from the immediately preceding
    # crawl cache. Rows selected in the earlier experiment are rejected by the
    # forbidden URL/encoding index above.
    supplement_cache = Path(
        "data/external/2026-08-20-long-url-raw/raw/commoncrawl_cache"
    )
    for path in sorted(supplement_cache.glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        grouped: defaultdict[str, list[dict]] = defaultdict(list)
        for record in records:
            try:
                _, hostname, _ = canonicalize_url(str(record.get("url", "")))
            except ValueError:
                continue
            grouped[registered_domain(hostname)].append(record)
        for registered, group in grouped.items():
            tranco_item = tranco_rank_by_registered.get(registered)
            if tranco_item is None or registered in current_phishing_domains:
                continue
            process_commoncrawl_records(group, tranco_item[0], registered, "CC-MAIN-2026-30 cached supplement")
    print(json.dumps({
        "stage": "commoncrawl_supplement",
        "login_candidates": len(login_candidates),
        "other_complex_candidates": len(other_complex_candidates),
    }), flush=True)

    for index, (rank, domain, registered) in enumerate(candidate_domains, start=1):
        if (
            len(login_candidates) >= round(args.minimum_login_urls * 1.05)
            and len(other_complex_candidates) >= round(
                (args.complex_legitimate - args.minimum_login_urls) * 1.05
            )
        ):
            break
        records, error = fetch_commoncrawl(
            args.crawl, domain, cache, args.commoncrawl_limit, args.request_delay
        )
        if error:
            request_errors[domain] = error
            counters["commoncrawl_request_error"] += 1
            continue
        process_commoncrawl_records(records, rank, registered, args.crawl)
        if index % 50 == 0:
            print(json.dumps({
                "stage": "commoncrawl", "queried_domains": index,
                "login_candidates": len(login_candidates),
                "other_complex_candidates": len(other_complex_candidates),
            }), flush=True)

    login_candidates.sort(key=lambda row: stable(args.seed, str(row["url"])))
    other_complex_candidates.sort(key=lambda row: stable(args.seed, str(row["url"])))
    selected_login = login_candidates[: args.minimum_login_urls]
    selected_other = other_complex_candidates[: args.complex_legitimate - args.minimum_login_urls]
    if len(selected_login) < args.minimum_login_urls:
        raise RuntimeError(
            f"Only {len(selected_login)} eligible login URLs; require {args.minimum_login_urls}"
        )
    if len(selected_other) < args.complex_legitimate - args.minimum_login_urls:
        raise RuntimeError(
            f"Only {len(selected_other)} eligible other complex URLs; "
            f"require {args.complex_legitimate - args.minimum_login_urls}"
        )
    complex_legitimate = selected_login + selected_other
    selected_legitimate_domains = {str(row["registered_domain"]) for row in complex_legitimate}

    root_legitimate = []
    root_seen_domains = set(selected_legitimate_domains)
    for rank, domain in sorted(tranco_rows, key=lambda row: stable(args.seed + 1, row[1])):
        raw_url = f"https://{domain}/"
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url)
        except ValueError:
            counters["tranco_invalid_url"] += 1
            continue
        domain_value = registered_domain(hostname)
        digest = encoded_digest(encode_url(normalized, vocabulary, 128))
        if domain_value in root_seen_domains or hostname in current_phishing_hosts or domain_value in current_phishing_domains:
            counters["tranco_duplicate_or_phishing_domain"] += 1
            continue
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters["tranco_prior_project_identity"] += 1
            continue
        structure = safe_structure(raw_url)
        if structure is None:
            continue
        root_seen_domains.add(domain_value)
        root_legitimate.append(output_row(
            normalized, raw_url, 0, "Tranco current root-domain proxy", str(rank), "",
            str(rank), hostname, structure,
        ))
        if len(root_legitimate) >= args.root_legitimate:
            break
    if len(root_legitimate) < args.root_legitimate:
        raise RuntimeError(
            f"Only {len(root_legitimate)} eligible Tranco roots; require {args.root_legitimate}"
        )

    legitimate = complex_legitimate + root_legitimate
    legitimate_domains = {str(row["registered_domain"]) for row in legitimate}
    phishing_candidates = []
    phishtank_rows.sort(
        key=lambda row: (
            str(row.get("verification_time", "")),
            stable(args.seed, str(row.get("phish_id", ""))),
        ),
        reverse=True,
    )
    phishing_seen_domains = set()
    for row in phishtank_rows:
        raw_url = str(row.get("url", ""))
        structure = safe_structure(raw_url)
        if structure is None:
            counters["phishtank_non_page"] += 1
            continue
        try:
            normalized, hostname, scheme = canonicalize_url(raw_url)
        except ValueError:
            counters["phishtank_invalid_url"] += 1
            continue
        domain_value = registered_domain(hostname)
        digest = encoded_digest(encode_url(normalized, vocabulary, 128))
        if scheme not in {"http", "https"}:
            counters["phishtank_wrong_scheme"] += 1
            continue
        if domain_value in legitimate_domains or domain_value in phishing_seen_domains:
            counters["phishtank_domain_overlap"] += 1
            continue
        if normalized in forbidden_urls or hostname in forbidden_hosts or digest in forbidden_hashes:
            counters["phishtank_prior_project_identity"] += 1
            continue
        phishing_seen_domains.add(domain_value)
        phishing_candidates.append(output_row(
            normalized, raw_url, 1, "PhishTank verified-online",
            str(row.get("phish_id", "")), str(row.get("verification_time", "")), "",
            hostname, structure,
        ))
        if len(phishing_candidates) >= args.phishing:
            break
    if len(phishing_candidates) < args.phishing:
        raise RuntimeError(
            f"Only {len(phishing_candidates)} eligible PhishTank rows; require {args.phishing}"
        )

    # Domain-disjoint 40/60 split. Complex legitimate domains are assigned as groups;
    # single-domain Tranco roots fill the exact legitimate target.
    development_legitimate_target = int(args.development_rows * 0.95)
    development_phishing_target = args.development_rows - development_legitimate_target
    grouped_complex: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in complex_legitimate:
        grouped_complex[str(row["registered_domain"])].append(row)
    development_legitimate = []
    final_legitimate = []
    development_domains = set()
    for domain in sorted(grouped_complex, key=lambda value: stable(args.seed + 2, value)):
        group = grouped_complex[domain]
        if len(development_legitimate) + len(group) <= min(
            development_legitimate_target, round(args.complex_legitimate * 0.40)
        ):
            development_legitimate.extend(group)
            development_domains.add(domain)
        else:
            final_legitimate.extend(group)
    roots_ordered = sorted(root_legitimate, key=lambda row: stable(args.seed + 3, str(row["url"])))
    needed = development_legitimate_target - len(development_legitimate)
    development_legitimate.extend(roots_ordered[:needed])
    development_domains.update(str(row["registered_domain"]) for row in roots_ordered[:needed])
    final_legitimate.extend(roots_ordered[needed:])
    if len(development_legitimate) != development_legitimate_target:
        raise RuntimeError("Could not construct the exact development legitimate split")

    phishing_ordered = sorted(phishing_candidates, key=lambda row: stable(args.seed + 4, str(row["url"])))
    development_phishing = phishing_ordered[:development_phishing_target]
    final_phishing = phishing_ordered[development_phishing_target:]
    development = development_legitimate + development_phishing
    final = final_legitimate + final_phishing
    random.Random(args.seed + 5).shuffle(development)
    random.Random(args.seed + 6).shuffle(final)
    all_rows = development + final
    for row in development:
        row["split"] = "policy_development"
    for row in final:
        row["split"] = "frozen_final_test"
    random.Random(args.seed + 7).shuffle(all_rows)

    if len(development) != args.development_rows or len(final) != total - args.development_rows:
        raise RuntimeError("Unexpected split row counts")
    if Counter(int(row["label"]) for row in all_rows) != Counter({0: 9_500, 1: 500}):
        raise RuntimeError("Unexpected class counts")
    development_domain_set = {str(row["registered_domain"]) for row in development}
    final_domain_set = {str(row["registered_domain"]) for row in final}
    if development_domain_set & final_domain_set:
        raise RuntimeError("Registered-domain overlap between development and final splits")
    normalized_urls = [str(row["url"]) for row in all_rows]
    if len(normalized_urls) != len(set(normalized_urls)):
        raise RuntimeError("Duplicate canonical URLs in final dataset")
    final_encoded_hashes = [
        encoded_digest(encode_url(url, vocabulary, 128)) for url in normalized_urls
    ]
    if len(final_encoded_hashes) != len(set(final_encoded_hashes)):
        raise RuntimeError("Duplicate 128-token encoded identities in final dataset")

    root.mkdir(parents=True, exist_ok=True)
    write_rows(outputs["all"], all_rows)
    write_rows(outputs["development"], development)
    write_rows(outputs["final"], final)
    source_counts = Counter(str(row["source"]) for row in all_rows)
    login_rows = [row for row in all_rows if int(row["label"]) == 0 and int(row["is_login_url"]) == 1]
    login_keywords = Counter(str(row["login_keyword"]) for row in login_rows)
    report = {
        "project_version": 3,
        "dataset_name": "Browser Policy Evaluation Dataset 10K",
        "purpose": "policy development and one untouched final browser-policy evaluation",
        "collection_only": True,
        "model_inference_performed": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "commoncrawl_collection": args.crawl,
        "rows": len(all_rows),
        "legitimate_rows": 9_500,
        "phishing_rows": 500,
        "phishing_prevalence": 0.05,
        "source_counts": dict(source_counts),
        "login_legitimate_rows": len(login_rows),
        "login_keyword_counts": dict(login_keywords),
        "splits": {
            "policy_development": {
                "rows": len(development),
                "legitimate": sum(int(row["label"]) == 0 for row in development),
                "phishing": sum(int(row["label"]) == 1 for row in development),
                "sha256": sha256_file(outputs["development"]),
            },
            "frozen_final_test": {
                "rows": len(final),
                "legitimate": sum(int(row["label"]) == 0 for row in final),
                "phishing": sum(int(row["label"]) == 1 for row in final),
                "sha256": sha256_file(outputs["final"]),
            },
        },
        "all_urls_sha256": sha256_file(outputs["all"]),
        "identity_contract": {
            "prior_project_url_overlap": 0,
            "prior_project_encoded_input_overlap": 0,
            "prior_project_hostname_overlap_rows": sum(
                int(row["prior_hostname_seen"]) for row in all_rows
            ),
            "hostname_policy": "root and phishing rows are hostname-disjoint; Common Crawl complex pages may use familiar hosts but must have unseen URL and encoded identities",
            "development_final_registered_domain_overlap": 0,
            "identity_source_count": len(identity_sources),
            "identity_sources": identity_sources,
        },
        "raw_sources": {
            "phishtank": {
                "reference": "https://www.phishtank.com/developer_info.php",
                "path": str(required["phishtank"]),
                "sha256": sha256_file(required["phishtank"]),
                "verified_online_rows": len(phishtank_rows),
            },
            "tranco": {
                "reference": "https://tranco-list.eu/",
                "path": str(required["tranco"]),
                "sha256": sha256_file(required["tranco"]),
                "rows": len(tranco_rows),
            },
            "commoncrawl": {
                "reference": "https://index.commoncrawl.org/",
                "collections_path": str(required["commoncrawl_collections"]),
                "collections_sha256": sha256_file(required["commoncrawl_collections"]),
                "cache_path": str(cache),
                "queried_domain_budget": args.domain_query_budget,
                "supplemental_collection": "CC-MAIN-2026-30",
                "supplemental_cache_path": str(supplement_cache),
            },
        },
        "filter_counts": dict(counters),
        "commoncrawl_request_errors": request_errors,
        "limitations": [
            "Tranco and Common Crawl membership supplies legitimate proxies rather than manual page-level labels.",
            "PhishTank verified-online feed membership supplies phishing labels; live URLs were not opened.",
            "Login-page legitimacy is inferred from a successful HTML capture beneath an eligible Tranco domain.",
            "The latest Common Crawl collection is primary; previously unselected identities from the immediately preceding cached collection supplement rate-limited queries.",
            "The declared five-percent prevalence simulates deployment and is not measured user prevalence.",
            "The policy-development split may be used for browser policy choices; the frozen final split must not be used for tuning.",
        ],
    }
    write_json(outputs["report"], report)
    (root / "EVALUATION_NOT_RUN.txt").write_text(
        "Collection completed and dataset frozen. No model inference or evaluation has been run.\n"
        "Do not evaluate frozen_final_test.csv until the user explicitly approves the final run.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset": str(root),
        "rows": len(all_rows),
        "login_legitimate_rows": len(login_rows),
        "splits": report["splits"],
        "model_inference_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
