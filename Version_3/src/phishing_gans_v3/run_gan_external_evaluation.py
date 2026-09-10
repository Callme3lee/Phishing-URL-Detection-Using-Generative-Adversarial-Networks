from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SNAPSHOTS = {
    "topdomains_phishtank": Path("data/external/2026-08-11"),
    "tranco_openphish": Path("data/external/2026-08-11-openphish-tranco"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every completed GAN condition on both frozen external sets.")
    parser.add_argument("--model", choices=("seqgan", "relgan"), required=True)
    parser.add_argument("--base", choices=("urls_new", "coverage_plus"), required=True)
    parser.add_argument("--percentages", default="10,20,30,40")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments/gan_augmentation_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/results/gan_augmentation_v1"))
    args = parser.parse_args()
    for percentage in (int(value) for value in args.percentages.split(",") if value):
        checkpoints = args.artifact_root / args.model / args.base / f"pct_{percentage:02d}"
        for snapshot_name, snapshot in SNAPSHOTS.items():
            output = args.output_root / args.model / args.base / f"pct_{percentage:02d}" / snapshot_name
            if (output / "summary.json").exists():
                print(f"skip completed: {output}", flush=True); continue
            subprocess.run([
                sys.executable, "-m", "phishing_gans_v3.evaluate_external",
                "--dataset", str(snapshot / "external_urls.csv"),
                "--collection-report", str(snapshot / "collection_report.json"),
                "--checkpoint-dir", str(checkpoints), "--output-dir", str(output),
            ], check=True)


if __name__ == "__main__":
    main()
