from __future__ import annotations

import hashlib
import html
import ipaddress
import math
import os
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import tldextract


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)


def canonicalize_url(raw: str, maximum_length: int = 4096) -> tuple[str, str, str]:
    """Return a scheme-free model URL, canonical hostname, and original scheme.

    Removing the scheme is intentional: in URLs_New.csv it nearly determines the
    label and would let classifiers learn collection provenance instead of phishing.
    User information is also removed because it can contain credentials or duplicate
    the same destination under many superficial representations.
    """
    value = html.unescape(str(raw)).strip()
    if not value:
        raise ValueError("empty_url")
    if len(value) > maximum_length:
        raise ValueError("raw_url_too_long")
    if any(ord(character) < 32 for character in value):
        raise ValueError("control_character")

    # Redirect targets can contain ``://`` inside a query even when the model URL
    # itself is scheme-free. Only a scheme at the beginning belongs to the outer URL.
    has_outer_scheme = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value))
    parse_value = value if has_outer_scheme else "http://" + value.lstrip("/")
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

    host_display = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    netloc = host_display + (f":{port}" if port is not None else "")
    with_placeholder_scheme = urlunsplit(
        ("http", netloc, parsed.path or "", parsed.query or "", parsed.fragment or "")
    )
    scheme_free = with_placeholder_scheme[len("http://"):]
    return scheme_free, hostname, parsed.scheme.lower()


def split_for_hostname(hostname: str, seed: int, train: float, validation: float) -> str:
    digest = hashlib.blake2b(f"{seed}:{hostname}".encode("utf-8"), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def encode_url(url: str, vocabulary: dict[str, int], maximum_length: int) -> np.ndarray:
    output = np.zeros(maximum_length, dtype=np.uint16)
    characters = list(url[: maximum_length - 1])
    unknown = vocabulary[UNK]
    for index, character in enumerate(characters):
        output[index] = vocabulary.get(character, unknown)
    output[len(characters)] = vocabulary[EOS]
    return output


def encoded_digest(tokens: np.ndarray) -> bytes:
    return hashlib.blake2b(tokens.tobytes(), digest_size=16).digest()


def lexical_features(url: str) -> np.ndarray:
    """Scheme-independent lexical features shared by RF and SVM."""
    parsed = urlsplit("http://" + url.lstrip("/"))
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    digits = sum(character.isdigit() for character in url)
    letters = sum(character.isalpha() for character in url)
    special = len(url) - digits - letters
    counts = np.fromiter((url.count(c) for c in ".-_/?:=@%&"), dtype=np.float32)
    frequencies = np.fromiter((url.count(c) for c in set(url)), dtype=np.float64)
    probabilities = frequencies / max(1, frequencies.sum())
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0
    suspicious = ("login", "verify", "account", "secure", "update", "bank", "signin", "password", "wallet")
    try:
        ip_host = float(bool(host) and isinstance(ipaddress.ip_address(host), (ipaddress.IPv4Address, ipaddress.IPv6Address)))
    except ValueError:
        ip_host = 0.0
    return np.concatenate((np.asarray([
        len(url), len(host), len(path), len(query), digits, digits / max(1, len(url)),
        letters, special, host.count("."), max(0, host.count(".") - 1),
        len(re.findall(r"[=&]", query)), ip_host, entropy,
        sum(url.lower().count(token) for token in suspicious), url.count("//"),
        url.count(".."), len(set(url)),
    ], dtype=np.float32), counts))


LEXICAL_FEATURE_NAMES = (
    "url_length", "hostname_length", "path_length", "query_length", "digit_count",
    "digit_ratio", "letter_count", "special_count", "dot_count_host",
    "estimated_subdomains", "query_delimiters", "ip_hostname", "character_entropy",
    "suspicious_token_count", "double_slash_count", "double_dot_count",
    "unique_character_count", "dot_count", "hyphen_count", "underscore_count",
    "slash_count", "question_count", "colon_count", "equals_count", "at_count",
    "percent_count", "ampersand_count",
)


PSL_EXTRACTOR = tldextract.TLDExtract(
    cache_dir=os.environ.get("PHISHING_GANS_TLDEXTRACT_CACHE", "/tmp/phishing_gans_v3_tldextract"),
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    frequencies = np.fromiter((value.count(character) for character in set(value)), dtype=np.float64)
    probabilities = frequencies / frequencies.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _ratio_count(value: str, predicate) -> float:
    return sum(bool(predicate(character)) for character in value) / max(1, len(value))


def _maximum_run_ratio(value: str) -> float:
    if not value:
        return 0.0
    maximum = current = 1
    for previous, character in zip(value, value[1:]):
        current = current + 1 if character == previous else 1
        maximum = max(maximum, current)
    return maximum / len(value)


def psl_aware_lexical_features(url: str) -> np.ndarray:
    """Return normalized, component-aware lexical features without TLD identity.

    The bundled tldextract suffix snapshot separates the registrable domain from
    public/private suffixes. Exact hostname, suffix, and token identity are never
    emitted, reducing direct source memorization.
    """
    parsed = urlsplit("http://" + url.lstrip("/"))
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    extracted = PSL_EXTRACTOR(hostname)
    registered = ".".join(value for value in (extracted.domain, extracted.suffix) if value)
    subdomain = extracted.subdomain or ""
    total_length = len(url)
    suspicious = ("login", "verify", "account", "secure", "update", "bank", "signin", "password", "wallet")
    try:
        ip_hostname = float(bool(hostname) and isinstance(
            ipaddress.ip_address(hostname), (ipaddress.IPv4Address, ipaddress.IPv6Address)
        ))
    except ValueError:
        ip_hostname = 0.0
    special = lambda character: not character.isalnum()
    values = (
        math.log1p(total_length), math.log1p(len(hostname)), math.log1p(len(registered)),
        math.log1p(len(extracted.domain or "")), math.log1p(len(extracted.suffix or "")),
        math.log1p(len(subdomain)), math.log1p(len(path)), math.log1p(len(query)),
        len(hostname) / max(1, total_length), len(path) / max(1, total_length),
        len(query) / max(1, total_length),
        _ratio_count(url, str.isdigit), _ratio_count(url, special),
        _ratio_count(hostname, str.isdigit), hostname.count("-") / max(1, len(hostname)),
        hostname.count(".") / max(1, len(hostname)),
        _ratio_count(path, str.isdigit), _ratio_count(path, special),
        _ratio_count(query, str.isdigit), _ratio_count(query, special),
        len([value for value in subdomain.split(".") if value]),
        len([value for value in path.split("/") if value]),
        len([value for value in re.split(r"[&;]", query) if value]),
        _entropy(hostname), _entropy(path), _entropy(query), ip_hostname,
        float("xn--" in hostname), float(bool(extracted.suffix)),
        url.count("%") / max(1, total_length),
        sum(url.lower().count(token) for token in suspicious) / max(1, total_length),
        _maximum_run_ratio(url), url.count("=") / max(1, total_length),
        url.count("&") / max(1, total_length), float(parsed.port is not None),
    )
    return np.asarray(values, dtype=np.float32)


PSL_AWARE_LEXICAL_FEATURE_NAMES = (
    "log_url_length", "log_hostname_length", "log_registrable_domain_length",
    "log_domain_label_length", "log_suffix_length", "log_subdomain_length",
    "log_path_length", "log_query_length", "hostname_length_ratio", "path_length_ratio",
    "query_length_ratio", "url_digit_ratio", "url_special_ratio", "hostname_digit_ratio",
    "hostname_hyphen_ratio", "hostname_dot_ratio", "path_digit_ratio", "path_special_ratio",
    "query_digit_ratio", "query_special_ratio", "subdomain_label_count", "path_depth",
    "query_parameter_count", "hostname_entropy", "path_entropy", "query_entropy",
    "ip_hostname", "punycode_hostname", "valid_public_suffix", "percent_encoding_ratio",
    "suspicious_token_ratio", "maximum_character_run_ratio", "equals_ratio", "ampersand_ratio",
    "explicit_port",
)
