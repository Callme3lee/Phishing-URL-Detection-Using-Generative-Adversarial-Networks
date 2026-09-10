from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable seed-42 low-data CNN/LSTM screen.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("data/low_data/real_10_percent"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/low_data/real_10_percent"))
    parser.add_argument("--models", default="cnn,lstm"); parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); models = [value.strip() for value in args.models.split(",") if value.strip()]
    indices = args.experiment_dir / "train_indices.npy"
    tasks = [("real_only", None, model) for model in models]
    for dose in (1, 2, 5, 10, 15):
        manifest = args.experiment_dir / f"seqgan_generated_{dose:02d}_percent.json"
        tasks.extend((f"seqgan_{dose:02d}_percent", manifest, model) for model in models)
    state = {"requested": len(tasks), "completed": [], "skipped": [], "failed": []}
    for index, (condition, manifest, model) in enumerate(tasks, 1):
        output = args.output_root / condition; target = output / model / f"seed_{args.seed}" / "metrics.json"
        label = f"{condition}|{model}|seed_{args.seed}"
        if target.exists(): state["skipped"].append(label); continue
        command = [sys.executable, "-m", "phishing_gans_v2.train_baselines", "--models", model, "--seeds", str(args.seed), "--output-dir", str(output)]
        if manifest is None: command.extend(("--real-indices", str(indices)))
        else: command.extend(("--augmentation-manifest", str(manifest)))
        print(json.dumps({"run": f"{index}/{len(tasks)}", "status": "start", "label": label}), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode or not target.exists(): state["failed"].append(label); write_json(args.output_root / "status.json", state); raise RuntimeError(label)
        state["completed"].append(label); write_json(args.output_root / "status.json", state)
    write_json(args.output_root / "status.json", state); print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
