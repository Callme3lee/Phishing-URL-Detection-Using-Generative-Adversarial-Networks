from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CONDITIONS = ("lsgan_quality_10", "lsgan_quality_20", "lsgan_quality_30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune augmented CNN checkpoints on the shared real-only corpus."
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/robust_augmentation")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_root(root: Path, seed: int) -> Path:
    return root / ("results" if seed == 42 else f"results_seed{seed}")


def main() -> None:
    args = parse_args()
    conditions = args.condition or list(CONDITIONS)
    environment = dict(os.environ)
    source_dir = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_dir, environment.get("PYTHONPATH", "")) if value
    )
    total = len(args.seeds) * len(conditions)
    position = 0
    for seed in args.seeds:
        for condition in conditions:
            position += 1
            initial = source_root(args.experiment_root, seed) / condition / "cnn" / "best.pt"
            output = (
                args.experiment_root
                / "finetuned"
                / f"seed{seed}"
                / f"{condition}_then_real"
                / "cnn"
            )
            if (output / "metrics.json").exists() and not args.force:
                print(f"[{position}/{total}] SKIP seed={seed} {condition}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "phishing_gans.train_classifier",
                "--model",
                "cnn",
                "--data-dir",
                str(args.experiment_root / "data" / "real_only"),
                "--output-dir",
                str(output),
                "--initial-checkpoint",
                str(initial),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(seed),
                "--quiet",
            ]
            print(f"[{position}/{total}] START seed={seed} {condition}", flush=True)
            completed = subprocess.run(
                command,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (output / "training.log").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                print(completed.stdout, flush=True)
                raise subprocess.CalledProcessError(completed.returncode, command)
            print(f"[{position}/{total}] DONE  seed={seed} {condition}", flush=True)


if __name__ == "__main__":
    main()
