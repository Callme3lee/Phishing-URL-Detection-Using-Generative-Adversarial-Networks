from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import load_json, write_json


MODELS = ("cnn", "lstm", "random_forest", "linear_svm")
GENERATORS = ("gumbel_lsgan", "seqgan", "relgan")
DOSES = (10, 20, 30, 40)


def comma_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the Version 2 experiment matrix one run at a time.")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--generators", default=",".join(GENERATORS))
    parser.add_argument("--doses", default="10,20,30,40")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-augmentation", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--status", type=Path, default=Path("artifacts/results/experiment_matrix/status.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def result_path(model: str, seed: int, generator: str | None = None, dose: int | None = None) -> Path:
    if generator is None:
        return Path("artifacts/baselines") / model / f"seed_{seed}" / "metrics.json"
    return Path("artifacts/augmentation") / generator / f"generated_{dose:02d}_percent" / model / f"seed_{seed}" / "metrics.json"


def command(model: str, seed: int, config: Path, generator: str | None = None, dose: int | None = None) -> list[str]:
    output = [sys.executable, "-m", "phishing_gans_v2.train_baselines", "--models", model, "--seeds", str(seed), "--config", str(config)]
    if generator is not None:
        output.extend(("--augmentation-manifest", f"data/augmented/{generator}/generated_{dose:02d}_percent.json"))
    return output


def main() -> None:
    args = parse_args()
    models = comma_list(args.models); seeds = comma_list(args.seeds, int)
    generators = comma_list(args.generators); doses = comma_list(args.doses, int)
    if set(models) - set(MODELS): raise ValueError("Unknown classifier in --models")
    if set(generators) - set(GENERATORS): raise ValueError("Unknown generator in --generators")
    if set(doses) - set(DOSES): raise ValueError("Dose must be 10, 20, 30, or 40")
    if not args.include_baselines and not args.include_augmentation:
        raise ValueError("Select --include-baselines and/or --include-augmentation")
    project = load_json(args.config)
    if set(seeds) - set(int(value) for value in project["seeds"]):
        raise ValueError("Seed is outside the predeclared project seeds")
    tasks = []
    if args.include_baselines:
        tasks.extend((model, seed, None, None) for seed in seeds for model in models)
    if args.include_augmentation:
        tasks.extend((model, seed, generator, dose) for generator in generators for dose in doses for seed in seeds for model in models)
    state = {"project_version": 2, "master_sha256": project["data"]["master_sha256"], "requested": len(tasks), "completed": [], "skipped_existing": [], "failed": [], "pending": [], "updated_at_unix": time.time()}
    for index, (model, seed, generator, dose) in enumerate(tasks, start=1):
        target = result_path(model, seed, generator, dose)
        label = f"{generator or 'real_only'}|{dose or 0}|{model}|seed_{seed}"
        if target.exists():
            state["skipped_existing"].append(label); print(json.dumps({"matrix": f"{index}/{len(tasks)}", "status": "skip", "run": label}), flush=True); continue
        run_command = command(model, seed, args.config, generator, dose)
        if args.dry_run:
            state["pending"].append(label); print(json.dumps({"matrix": f"{index}/{len(tasks)}", "status": "dry_run", "run": label, "command": run_command}), flush=True); continue
        print(json.dumps({"matrix": f"{index}/{len(tasks)}", "status": "start", "run": label}), flush=True)
        completed = subprocess.run(run_command, check=False)
        if completed.returncode != 0 or not target.exists():
            state["failed"].append({"run": label, "returncode": completed.returncode}); state["updated_at_unix"] = time.time(); write_json(args.status, state)
            raise RuntimeError(f"Experiment failed: {label}")
        state["completed"].append(label); state["updated_at_unix"] = time.time(); write_json(args.status, state)
    state["updated_at_unix"] = time.time(); write_json(args.status, state); print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
