#!/usr/bin/env python3
"""Normalize and merge the newly supplied phishing/non-phishing URL CSVs."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "Version_3/src"))

from phishing_gans_v3.url_data import PSL_EXTRACTOR, canonicalize_url  # noqa: E402


INPUTS = (
    (REPOSITORY / "phishing_test/phishing.csv", 1, "new_phishing_csv"),
    (REPOSITORY / "phishing_test/not-phishing.csv", 0, "new_not_phishing_csv"),
)
TRAINING_MANIFEST = REPOSITORY / "Version_3/data/experiments/recent_update_gan10_v1/gan_10pct/processed/train_manifest.csv"
OUTPUT = REPOSITORY / "phishing_test/normalized_combined_test_v2_raw_preserved"
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_identity(model_url: str) -> str:
    """Keep root slash and no-slash forms distinct for a fair robustness test."""
    return model_url


def category(full_url: str, model_url: str) -> str:
    parsed = urlsplit(full_url)
    hostname = (parsed.hostname or "").lower()
    lower = model_url.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    try:
        ipaddress.ip_address(hostname)
        return "ip_host"
    except ValueError:
        pass
    if any(token in lower for token in ("login", "signin", "sign-in", "logon", "account", "verify", "password")):
        return "login_or_account"
    if len(model_url) >= 128 or len(query) >= 80:
        return "long_url_or_query"
    if "%" in model_url:
        return "encoded_characters"
    if parsed.port is not None:
        return "explicit_port"
    if path in {"", "/"} and not query and not parsed.fragment:
        return "root_domain"
    if query:
        return "query_parameters"
    extracted = PSL_EXTRACTOR(hostname)
    if extracted.subdomain:
        return "subdomain_path"
    return "path"


def read_training_identities() -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    domains: set[str] = set()
    with TRAINING_MANIFEST.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            model_url = row["url"].strip()
            identities.add(root_identity(model_url))
            domains.add(row["canonical_hostname"].strip().lower())
    return identities, domains


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Combined test set already exists: {OUTPUT}")
    csv.field_size_limit(sys.maxsize)
    OUTPUT.mkdir(parents=True)
    training_identities, training_domains = read_training_identities()
    parsed_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    input_counts: Counter[str] = Counter()

    for path, label, source in INPUTS:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for source_index, row in enumerate(csv.DictReader(handle), start=1):
                input_counts[source] += 1
                raw = str(row.get("url") or "").strip()
                try:
                    model_url, hostname, scheme = canonicalize_url(raw)
                    if scheme not in {"http", "https"}:
                        raise ValueError("unsupported_scheme")
                    normalized = f"{scheme}://{model_url}"
                    identity = root_identity(model_url)
                    raw_parsed = urlsplit(raw)
                    is_root = raw_parsed.path in {"", "/"} and not raw_parsed.query and not raw_parsed.fragment
                    parsed_rows.append({
                        "url": raw,
                        "normalized_url": normalized,
                        "model_url": model_url,
                        "label": label,
                        "label_name": "phishing" if label else "legitimate",
                        "domain": hostname,
                        "category": category(raw, model_url),
                        "source": source,
                        "source_row_id": row.get("_id") or str(source_index),
                        "root_representation": (
                            "slash" if raw_parsed.path == "/" else "empty"
                        ) if is_root else "not_root",
                        "identity": identity,
                        "training_identity_overlap": identity in training_identities,
                        "training_domain_overlap": hostname in training_domains,
                    })
                except (ValueError, UnicodeError) as error:
                    exclusions.append({"raw_url": raw, "label": label, "source": source,
                                       "reason": f"invalid_url:{error}"})

    labels_by_identity: dict[str, set[int]] = defaultdict(set)
    for row in parsed_rows:
        labels_by_identity[str(row["identity"])].add(int(row["label"]))
    conflicts = {identity for identity, labels in labels_by_identity.items() if len(labels) > 1}

    retained: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    reason_counts: Counter[str] = Counter()
    for row in parsed_rows:
        identity = str(row["identity"]); label = int(row["label"])
        if identity in conflicts:
            reason = "conflicting_labels_after_normalization"
        elif bool(row["training_identity_overlap"]):
            reason = "normalized_identity_seen_in_extension_training"
        elif (identity, label) in seen:
            reason = "duplicate_after_normalization"
        else:
            seen.add((identity, label)); retained.append(row); continue
        reason_counts[reason] += 1
        exclusions.append({"raw_url": row["url"], "label": label, "source": row["source"], "reason": reason})

    random.Random(SEED).shuffle(retained)
    output_fields = ["url", "normalized_url", "model_url", "label", "label_name", "domain", "category", "source",
                     "source_row_id", "root_representation", "training_domain_overlap"]
    with (OUTPUT / "combined_test_urls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in retained:
            writer.writerow({key: row[key] for key in output_fields})
    with (OUTPUT / "excluded_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["raw_url", "label", "source", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(exclusions)

    label_counts = Counter(int(row["label"]) for row in retained)
    category_counts = Counter(str(row["category"]) for row in retained)
    source_counts = Counter(str(row["source"]) for row in retained)
    root_counts = Counter(str(row["root_representation"]) for row in retained)
    manifest = {
        "name": "normalized_combined_test_v2_raw_preserved",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fresh external comparison of the frozen Version 3 ensemble and Final root-paired candidate",
        "inputs": {str(path.relative_to(REPOSITORY)): {"label": label, "rows": input_counts[source], "sha256": sha256(path)}
                   for path, label, source in INPUTS},
        "excluded_old_file": "phishing_test/phish.txt was not merged because it predates the two newly added CSVs",
        "normalization": {
            "pipeline": "Version 3 canonicalize_url",
            "raw_url": "url is preserved exactly except surrounding whitespace is trimmed",
            "scheme": "preserved in url and normalized_url; removed only in model_url",
            "hostname": "lowercase IDNA",
            "paths_queries_fragments": "preserved",
            "root_empty_and_slash": "kept as distinct identities; no slash is added or removed",
            "training_overlap_policy": "exclude normalized identity overlap; domain overlap is retained and reported",
        },
        "rows": len(retained),
        "labels": {"legitimate": label_counts[0], "phishing": label_counts[1]},
        "phishing_prevalence": label_counts[1] / max(1, len(retained)),
        "sources": dict(sorted(source_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "root_representations": dict(sorted(root_counts.items())),
        "training_domain_overlap_rows": int(sum(bool(row["training_domain_overlap"]) for row in retained)),
        "training_domain_overlap_rate": sum(bool(row["training_domain_overlap"]) for row in retained) / max(1, len(retained)),
        "excluded_rows": len(exclusions),
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "conflicting_normalized_identities": len(conflicts),
        "shuffle_seed": SEED,
        "files": {
            "combined_test_urls.csv": sha256(OUTPUT / "combined_test_urls.csv"),
            "excluded_rows.csv": sha256(OUTPUT / "excluded_rows.csv"),
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
