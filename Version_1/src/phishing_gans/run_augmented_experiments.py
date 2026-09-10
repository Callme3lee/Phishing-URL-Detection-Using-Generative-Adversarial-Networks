from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = ("cnn", "lstm", "rf", "svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all controlled augmented-data classifier experiments."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/augmented_experiments"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional separate output root, useful for independent random seeds.",
    )
    parser.add_argument("--model", action="append", choices=MODELS, default=None)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def command_for(
    model: str, condition: str, root: Path, output_dir: Path, args: argparse.Namespace
) -> list[str]:
    if model in {"cnn", "lstm"}:
        return [
            sys.executable,
            "-m",
            "phishing_gans.train_classifier",
            "--model",
            model,
            "--data-dir",
            str(root / "data" / condition),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--quiet",
        ]
    return [
        sys.executable,
        "-m",
        "phishing_gans.train_traditional",
        "--model",
        model,
        "--feature-dir",
        str(root / "features" / condition),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    models = args.model or list(MODELS)
    results_root = args.results_root or (args.experiment_root / "results")
    available = sorted(path.name for path in (args.experiment_root / "data").iterdir())
    conditions = args.condition or available
    unknown = sorted(set(conditions) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    total = len(models) * len(conditions)
    position = 0
    for model in models:
        for condition in conditions:
            position += 1
            output_dir = results_root / condition / model
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"[{position}/{total}] SKIP {model} {condition}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = command_for(model, condition, args.experiment_root, output_dir, args)
            print(f"[{position}/{total}] START {model} {condition}", flush=True)
            completed = subprocess.run(
                command,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (output_dir / "training.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            if completed.returncode != 0:
                print(completed.stdout, flush=True)
                raise subprocess.CalledProcessError(completed.returncode, command)
            print(f"[{position}/{total}] DONE  {model} {condition}", flush=True)


if __name__ == "__main__":
    main()
