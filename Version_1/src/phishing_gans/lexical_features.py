from __future__ import annotations

import ipaddress
import math
from collections import Counter
from urllib.parse import urlsplit

import numpy as np


SUSPICIOUS_WORDS = (
    "account",
    "auth",
    "bank",
    "confirm",
    "credential",
    "login",
    "password",
    "pay",
    "secure",
    "signin",
    "update",
    "verify",
    "wallet",
)

FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "letter_count",
    "digit_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "question_count",
    "equals_count",
    "ampersand_count",
    "at_count",
    "percent_count",
    "special_count",
    "letter_ratio",
    "digit_ratio",
    "special_ratio",
    "subdomain_count",
    "tld_length",
    "is_ip_hostname",
    "uses_https",
    "character_entropy",
    "suspicious_word_count",
    "longest_digit_run",
    "double_slash_count",
)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _longest_digit_run(text: str) -> int:
    best = current = 0
    for character in text:
        if character.isdigit():
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def extract_lexical_features(url: str) -> np.ndarray:
    lowered = url.lower()
    try:
        parsed = urlsplit(url if "://" in url else f"http://{url}")
        hostname = (parsed.hostname or "").lower()
        path, query, fragment = parsed.path, parsed.query, parsed.fragment
    except ValueError:
        # Retain labeled malformed inputs using raw-string features. An empty
        # hostname prevents an invalid bracketed IP from aborting extraction.
        hostname = ""
        path, query, fragment = url, "", ""
    letters = sum(character.isalpha() for character in url)
    digits = sum(character.isdigit() for character in url)
    specials = len(url) - letters - digits
    length = max(len(url), 1)
    labels = [part for part in hostname.split(".") if part]
    try:
        is_ip = float(bool(hostname) and ipaddress.ip_address(hostname) is not None)
    except ValueError:
        is_ip = 0.0
    values = (
        len(url),
        len(hostname),
        len(path),
        len(query),
        len(fragment),
        letters,
        digits,
        url.count("."),
        url.count("-"),
        url.count("_"),
        url.count("/"),
        url.count("?"),
        url.count("="),
        url.count("&"),
        url.count("@"),
        url.count("%"),
        specials,
        letters / length,
        digits / length,
        specials / length,
        max(len(labels) - 2, 0),
        len(labels[-1]) if labels else 0,
        is_ip,
        float(lowered.startswith("https://")),
        _entropy(url),
        sum(lowered.count(word) for word in SUSPICIOUS_WORDS),
        _longest_digit_run(url),
        url.count("//"),
    )
    return np.asarray(values, dtype=np.float32)
