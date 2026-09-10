from __future__ import annotations

import hashlib
import html
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)


def normalize_url(raw: str, maximum_length: int = 4096) -> tuple[str, str]:
    value = html.unescape(str(raw)).strip()
    if not value:
        raise ValueError("empty_url")
    if len(value) > maximum_length:
        raise ValueError("raw_url_too_long")
    if any(ord(character) < 32 for character in value):
        raise ValueError("control_character")

    parse_value = value if "://" in value else "http://" + value.lstrip("/")
    try:
        parsed = urlsplit(parse_value)
        hostname = (parsed.hostname or "").strip(".").lower()
    except ValueError as error:
        raise ValueError("url_parse_error") from error
    if not hostname or len(hostname) > 253:
        raise ValueError("invalid_hostname")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise ValueError("invalid_idna_hostname") from error

    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("invalid_port") from error
    host_display = hostname
    if ":" in hostname and not hostname.startswith("["):
        host_display = f"[{hostname}]"
    netloc = host_display + (f":{port}" if port is not None else "")
    scheme = parsed.scheme.lower()
    normalized_with_scheme = urlunsplit(
        (scheme, netloc, parsed.path or "", parsed.query or "", parsed.fragment or "")
    )
    normalized = normalized_with_scheme if "://" in value else normalized_with_scheme[len("http://"):]
    return normalized, hostname


def split_for_hostname(hostname: str, seed: int, train: float, validation: float) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{hostname}".encode("utf-8"), digest_size=8
    ).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def encode_url(url: str, vocabulary: dict[str, int], maximum_length: int) -> np.ndarray:
    output = np.zeros(maximum_length, dtype=np.uint16)
    usable = maximum_length - 1
    characters = list(url[:usable])
    unknown = vocabulary[UNK]
    for index, character in enumerate(characters):
        output[index] = vocabulary.get(character, unknown)
    output[len(characters)] = vocabulary[EOS]
    return output


def encoded_digest(tokens: np.ndarray) -> bytes:
    return hashlib.blake2b(tokens.tobytes(), digest_size=16).digest()


def lexical_features(url: str) -> np.ndarray:
    parse_value = url if "://" in url else "http://" + url.lstrip("/")
    parsed = urlsplit(parse_value)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    digits = sum(character.isdigit() for character in url)
    letters = sum(character.isalpha() for character in url)
    special = len(url) - digits - letters
    counts = np.fromiter((url.count(c) for c in ".-_/?:=@%&"), dtype=np.float32)
    frequencies = np.fromiter(
        (url.count(character) for character in set(url)), dtype=np.float64
    )
    probabilities = frequencies / max(1, frequencies.sum())
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0
    suspicious = ("login", "verify", "account", "secure", "update", "bank", "signin", "password", "wallet")
    try:
        ip_host = float(bool(host) and isinstance(ipaddress.ip_address(host), (ipaddress.IPv4Address, ipaddress.IPv6Address)))
    except ValueError:
        ip_host = 0.0
    return np.concatenate([
        np.asarray([
            len(url), len(host), len(path), len(query),
            digits, digits / max(1, len(url)), letters, special,
            host.count("."), max(0, host.count(".") - 1),
            len(re.findall(r"[=&]", query)), float(parsed.scheme.lower() == "https"),
            ip_host, entropy, sum(url.lower().count(token) for token in suspicious),
            url.count("//"), url.count(".."), len(set(url)),
        ], dtype=np.float32),
        counts,
    ])


LEXICAL_FEATURE_NAMES = (
    "url_length", "hostname_length", "path_length", "query_length",
    "digit_count", "digit_ratio", "letter_count", "special_count",
    "dot_count_host", "estimated_subdomains", "query_delimiters", "https",
    "ip_hostname", "character_entropy", "suspicious_token_count",
    "double_slash_count", "double_dot_count", "unique_character_count",
    "dot_count", "hyphen_count", "underscore_count", "slash_count",
    "question_count", "colon_count", "equals_count", "at_count",
    "percent_count", "ampersand_count",
)
