from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .prepare_later_collected import sha256
from .prepare_post_training_snapshot import load_used_hosts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen hostname exclusions for prospective evaluation.")
    parser.add_argument("--used-csv", type=Path, action="append", required=True)
    parser.add_argument("--used-url-list", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hosts, counts = load_used_hosts(args.used_csv, args.used_url_list or [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = np.fromiter(hosts, dtype=np.uint64, count=len(hosts))
    values.sort()
    np.save(args.output, values)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "immutable prior/training/evaluation/generated hostname exclusions for prospective testing",
        "unique_hostname_hashes": len(values),
        "hash": "BLAKE2b-64 of canonical exact hostname",
        "counts": dict(counts),
        "used_csv": [{"path": str(path), "sha256": sha256(path)} for path in args.used_csv],
        "used_url_list": [
            {"path": str(path), "sha256": sha256(path)} for path in (args.used_url_list or [])
        ],
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
