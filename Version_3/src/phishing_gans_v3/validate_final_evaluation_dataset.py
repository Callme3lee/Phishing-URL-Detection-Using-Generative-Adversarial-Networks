from __future__ import annotations

import csv
import gzip
import json
import zipfile
from collections import Counter
from pathlib import Path

from .common import sha256_file
from .url_data import encode_url, encoded_digest


ROOT = Path("Final Evaluation Dataset")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((ROOT / "collection_report.json").read_text(encoding="utf-8"))
    all_rows = rows(ROOT / "all_urls.csv")
    development = rows(ROOT / "policy_development.csv")
    final = rows(ROOT / "frozen_final_test.csv")
    vocabulary = json.loads(Path("data/processed/vocabulary.json").read_text(encoding="utf-8"))

    assert report["collection_only"] is True
    assert report["model_inference_performed"] is False
    assert len(all_rows) == 10_000
    assert len(development) == 4_000
    assert len(final) == 6_000
    assert Counter(int(row["label"]) for row in all_rows) == Counter({0: 9_500, 1: 500})
    assert Counter(int(row["label"]) for row in development) == Counter({0: 3_800, 1: 200})
    assert Counter(int(row["label"]) for row in final) == Counter({0: 5_700, 1: 300})

    all_urls = [row["url"] for row in all_rows]
    development_urls = {row["url"] for row in development}
    final_urls = {row["url"] for row in final}
    assert len(all_urls) == len(set(all_urls))
    assert not development_urls & final_urls
    assert development_urls | final_urls == set(all_urls)
    digests = [encoded_digest(encode_url(url, vocabulary, 128)) for url in all_urls]
    assert len(digests) == len(set(digests))

    development_domains = {row["registered_domain"] for row in development}
    final_domains = {row["registered_domain"] for row in final}
    assert not development_domains & final_domains

    login_rows = [
        row for row in all_rows
        if int(row["label"]) == 0 and int(row["is_login_url"]) == 1
    ]
    phishing_login_rows = [
        row for row in all_rows
        if int(row["label"]) == 1 and int(row["is_login_url"]) == 1
    ]
    assert len(login_rows) == 50
    assert all(int(row["label"]) == 0 for row in login_rows)
    assert all("HTML capture" in row["source"] for row in login_rows)
    forbidden_names = ("robots.txt", "sitemap.xml", "sitemap_index.xml", "favicon.ico")
    assert not any(any(name in row["url"].lower() for name in forbidden_names) for row in all_rows)

    source_counts = Counter(row["source"] for row in all_rows)
    assert source_counts["Tranco current root-domain proxy"] == 8_800
    assert source_counts["PhishTank verified-online"] == 500
    assert sum(count for source, count in source_counts.items() if "HTML capture" in source) == 700

    assert sha256_file(ROOT / "all_urls.csv") == report["all_urls_sha256"]
    assert sha256_file(ROOT / "policy_development.csv") == report["splits"]["policy_development"]["sha256"]
    assert sha256_file(ROOT / "frozen_final_test.csv") == report["splits"]["frozen_final_test"]["sha256"]

    with gzip.open(ROOT / "raw/phishtank_online_valid.csv.gz", "rb") as handle:
        while handle.read(1024 * 1024):
            pass
    with zipfile.ZipFile(ROOT / "raw/tranco_top1m.csv.zip") as archive:
        assert archive.testzip() is None
    json.loads((ROOT / "raw/commoncrawl_collinfo.json").read_text(encoding="utf-8"))
    not_run_marker = ROOT / "EVALUATION_NOT_RUN.txt"
    completed_marker = ROOT / "EVALUATION_COMPLETED.txt"
    assert not (not_run_marker.exists() and completed_marker.exists())
    assert not_run_marker.exists() or completed_marker.exists()
    evaluation_status = "not_run"
    if completed_marker.exists():
        result_path = Path("artifacts/results/final_browser_evaluation_2026-09-01_v1/results.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["evaluation_status"] == "COMPLETED ONCE; NO POST-TEST TUNING PERMITTED"
        assert result["dataset"]["sha256"] == report["splits"]["frozen_final_test"]["sha256"]
        assert result["dataset"]["rows"] == len(final)
        evaluation_status = "completed_once"

    print(json.dumps({
        "validation_passed": True,
        "collection_model_inference_performed": False,
        "subsequent_evaluation_status": evaluation_status,
        "rows": len(all_rows),
        "class_counts": dict(Counter(row["label"] for row in all_rows)),
        "source_counts": dict(source_counts),
        "login_urls": len(login_rows),
        "phishing_login_themed_urls": len(phishing_login_rows),
        "unique_urls": len(set(all_urls)),
        "unique_encoded_inputs": len(set(digests)),
        "development_final_domain_overlap": 0,
        "all_urls_sha256": report["all_urls_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
