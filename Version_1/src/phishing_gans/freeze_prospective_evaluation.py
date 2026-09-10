from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import encode_url, load_json
from .prepare_later_collected import normalized_hostname, sha256
from .prepare_post_training_snapshot import host_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a target-sized prospective URL evaluation exactly once.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--tranco-zip", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, default=Path("artifacts/data_2m"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evaluation: {args.output_dir}")

    with args.candidates.open("r", encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    if len(candidates) < args.target:
        raise RuntimeError(
            f"Prospective ledger has {len(candidates)} phishing hostnames; target is {args.target}. "
            "Keep collecting without evaluation."
        )
    phishing = candidates[: args.target]
    forbidden = {int(value) for value in np.load(args.exclusions)}
    forbidden.update(host_key(row["hostname"]) for row in phishing)
    benign = []
    with zipfile.ZipFile(args.tranco_zip) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as binary_handle:
            lines = (line.decode("utf-8", errors="replace") for line in binary_handle)
            for raw in csv.reader(lines):
                if len(raw) < 2:
                    continue
                host = normalized_hostname(raw[1])
                if not host:
                    continue
                key = host_key(host)
                if key in forbidden:
                    continue
                forbidden.add(key)
                benign.append(
                    {
                        "url": f"https://{host}/",
                        "hostname": host,
                        "label": 0,
                        "source": "tranco_unused_control",
                        "observed_at": "",
                        "source_id": "",
                    }
                )
                if len(benign) == args.target:
                    break
    if len(benign) != args.target:
        raise RuntimeError(f"Found {len(benign)} benign controls; needed {args.target}.")

    rows = []
    for benign_row, phishing_row in zip(benign, phishing):
        malicious = {
            "url": phishing_row["url"],
            "hostname": phishing_row["hostname"],
            "label": 1,
            "source": phishing_row["source"],
            "observed_at": phishing_row["observed_at"],
            "source_id": phishing_row["source_id"],
        }
        rows.extend((benign_row, malicious))

    encoding = load_json(args.training_data_dir / "metadata.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokens = np.lib.format.open_memmap(
        args.output_dir / "test_tokens.npy", mode="w+", dtype=np.uint8,
        shape=(len(rows), encoding["max_length"]),
    )
    labels = np.lib.format.open_memmap(
        args.output_dir / "test_labels.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    with (args.output_dir / "frozen_urls.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["url", "hostname", "label", "source", "observed_at", "source_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            tokens[index] = encode_url(row["url"], encoding["vocab"], encoding["max_length"])
            labels[index] = row["label"]
            writer.writerow(row)
    tokens.flush()
    labels.flush()
    report = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "first target-sized candidate prefix; evaluation-only; thresholds and checkpoints must remain frozen",
        "target_per_class": args.target,
        "total_examples": 2 * args.target,
        "candidate_ledger": str(args.candidates),
        "candidate_ledger_sha256": sha256(args.candidates),
        "exclusions_sha256": sha256(args.exclusions),
        "tranco_sha256": sha256(args.tranco_zip),
        "frozen_urls_sha256": sha256(args.output_dir / "frozen_urls.csv"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
