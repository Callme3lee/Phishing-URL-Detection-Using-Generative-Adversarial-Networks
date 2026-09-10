from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .common import load_json
from .prepare_later_collected import normalized_hostname
from .train_gan import decode


HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)
REPEATED_CHARACTER = re.compile(r"(.)\1{5,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly filter an offline generated URL corpus.")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument(
        "--forbidden-csv",
        type=Path,
        action="append",
        default=None,
        help="Optional CSV containing URL or hostname columns whose hostnames must never be generated.",
    )
    parser.add_argument(
        "--forbidden-host-cache",
        type=Path,
        default=None,
        help="Optional .npy cache for compact forbidden-host hashes.",
    )
    return parser.parse_args()


def rejection_reason(url: str) -> str | None:
    if not 8 <= len(url) <= 128:
        return "length"
    if not url.isascii() or not url.isprintable() or any(char.isspace() for char in url):
        return "non_ascii_or_whitespace"
    if REPEATED_CHARACTER.search(url.lower()):
        return "repeated_character"
    try:
        parsed = urlsplit(url if "://" in url else f"http://{url}")
    except ValueError:
        return "parse_error"
    hostname = (parsed.hostname or "").strip(".").lower()
    if not hostname or len(hostname) > 253 or "." not in hostname:
        return "hostname"
    labels = hostname.split(".")
    if any(not HOST_LABEL.fullmatch(label) for label in labels):
        return "hostname_label"
    tld = labels[-1]
    if not 2 <= len(tld) <= 24 or not tld.isalpha():
        return "tld"
    return None


def training_matches(samples: np.ndarray, data_dir: Path) -> set[bytes]:
    generated = {row.tobytes() for row in samples}
    matches: set[bytes] = set()
    for split in ("train", "val", "test"):
        token_path = data_dir / f"{split}_tokens.npy"
        label_path = data_dir / f"{split}_labels.npy"
        if not token_path.exists() or not label_path.exists():
            continue
        tokens = np.load(token_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        for index in np.flatnonzero(labels == 1):
            value = tokens[index].tobytes()
            if value in generated:
                matches.add(value)
    return matches


def hostname_key(host: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(host.encode("ascii", errors="ignore"), digest_size=8).digest(),
        "big",
    )


def load_forbidden_hosts(paths: list[Path], cache: Path | None = None) -> set[int]:
    if cache is not None and cache.exists():
        return {int(value) for value in np.load(cache)}
    hosts: set[int] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                host = normalized_hostname(row.get("hostname") or row.get("url") or "")
                if host:
                    hosts.add(hostname_key(host))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.fromiter(hosts, dtype=np.uint64, count=len(hosts)))
    return hosts


def main() -> None:
    args = parse_args()
    tokens = np.load(args.tokens)
    metadata = load_json(args.data_dir / "metadata.json")
    inverse = {
        value: key for key, value in metadata["vocab"].items() if len(key) == 1
    }
    urls = decode(tokens, inverse)
    memorized = training_matches(tokens, args.data_dir)
    forbidden_hosts = load_forbidden_hosts(
        args.forbidden_csv or [], args.forbidden_host_cache
    )
    reasons = Counter()
    accepted_tokens = []
    accepted_urls = []
    seen: set[bytes] = set()
    for row, url in zip(tokens, urls, strict=True):
        key = row.tobytes()
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        if key in memorized:
            reasons["training_memorization"] += 1
            continue
        reason = rejection_reason(url)
        if reason is not None:
            reasons[reason] += 1
            continue
        generated_host = normalized_hostname(url)
        if generated_host and hostname_key(generated_host) in forbidden_hosts:
            reasons["forbidden_hostname"] += 1
            continue
        if len(accepted_urls) >= args.limit:
            reasons["over_limit"] += 1
            continue
        accepted_tokens.append(row)
        accepted_urls.append(url)

    accepted = (
        np.stack(accepted_tokens).astype(np.uint8)
        if accepted_tokens
        else np.empty((0, tokens.shape[1]), dtype=np.uint8)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "filtered_tokens.npy", accepted)
    (args.output_dir / "filtered_urls.txt").write_text(
        "\n".join(accepted_urls), encoding="utf-8"
    )
    report = {
        "source": str(args.tokens),
        "raw_count": len(tokens),
        "accepted_count": len(accepted_urls),
        "acceptance_rate": len(accepted_urls) / max(len(tokens), 1),
        "memorization_scope": "all available train/val/test malicious rows",
        "forbidden_hostname_count": len(forbidden_hosts),
        "forbidden_host_cache": str(args.forbidden_host_cache) if args.forbidden_host_cache else None,
        "rejections": dict(reasons),
    }
    (args.output_dir / "filter_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("first accepted samples:")
    for url in accepted_urls[:10]:
        print(f"  {url}")


if __name__ == "__main__":
    main()
