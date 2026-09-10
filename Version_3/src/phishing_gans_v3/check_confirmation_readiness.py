from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

from .common import load_json, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refuse confirmation until source feeds genuinely advance.")
    parser.add_argument("--calibration-report", type=Path,
                        default=Path("data/calibration/2026-08-11-post-hybrid/collection_report.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/calibration/2026-08-11-post-hybrid/raw"))
    parser.add_argument("--minimum-new-phishtank", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/results/hybrid_cnn_coral_v1/confirmation_readiness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_json(args.calibration_report)
    frozen_at = datetime.fromisoformat(report["frozen_at_utc"])
    openphish = args.raw_dir / "openphish_feed.txt"
    tranco = args.raw_dir / "tranco_top1m.csv.zip"
    phishtank = args.raw_dir / "phishtank_online_valid.csv.gz"
    with gzip.open(phishtank, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    later_rows = []
    for row in rows:
        try:
            verified_at = datetime.fromisoformat(row.get("verification_time", ""))
        except ValueError:
            continue
        if verified_at > frozen_at and row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes":
            later_rows.append(row)
    checks = {
        "openphish_hash_changed": sha256_file(openphish) != report["sources"]["openphish_audit"]["sha256"],
        "tranco_hash_changed": sha256_file(tranco) != report["sources"]["legitimate"]["sha256"],
        "enough_phishtank_rows_verified_after_calibration": len(later_rows) >= args.minimum_new_phishtank,
    }
    payload = {
        "ready": all(checks.values()), "checks": checks,
        "calibration_frozen_at_utc": report["frozen_at_utc"],
        "phishtank_rows_verified_after_calibration": len(later_rows),
        "minimum_required": args.minimum_new_phishtank,
        "instruction": "Download fresh official OpenPhish, Tranco, and PhishTank files into a new raw directory, then rerun this check. Do not evaluate confirmation labels unless ready is true.",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
