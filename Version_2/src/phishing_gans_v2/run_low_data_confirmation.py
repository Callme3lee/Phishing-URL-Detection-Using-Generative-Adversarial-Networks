from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm promoted low-data conditions on paired seeds.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("data/low_data/real_10_percent"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/low_data/real_10_percent"))
    parser.add_argument("--seeds", default="43,44")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    indices = args.experiment_dir / "train_indices.npy"
    promoted = (
        ("real_only", "cnn", None),
        ("real_only", "lstm", None),
        ("seqgan_02_percent", "cnn", args.experiment_dir / "seqgan_generated_02_percent.json"),
        ("seqgan_10_percent", "lstm", args.experiment_dir / "seqgan_generated_10_percent.json"),
    )
    tasks = [(seed, *condition) for seed in seeds for condition in promoted]
    state = {"protocol": "paired seeds for promoted model-specific conditions", "requested": len(tasks), "completed": [], "skipped": [], "failed": []}
    for position, (seed, condition, model, manifest) in enumerate(tasks, 1):
        output = args.output_root / condition; target = output / model / f"seed_{seed}" / "metrics.json"
        label = f"{condition}|{model}|seed_{seed}"
        if target.exists(): state["skipped"].append(label); continue
        command = [sys.executable, "-m", "phishing_gans_v2.train_baselines", "--models", model, "--seeds", str(seed), "--output-dir", str(output)]
        if manifest is None: command.extend(("--real-indices", str(indices)))
        else: command.extend(("--augmentation-manifest", str(manifest)))
        print(json.dumps({"run": f"{position}/{len(tasks)}", "status": "start", "label": label}), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode or not target.exists():
            state["failed"].append(label); write_json(args.output_root / "confirmation_status.json", state); raise RuntimeError(label)
        state["completed"].append(label); write_json(args.output_root / "confirmation_status.json", state)
    write_json(args.output_root / "confirmation_status.json", state); print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
