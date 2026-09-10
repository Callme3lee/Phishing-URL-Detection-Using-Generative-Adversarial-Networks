from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_json
from .train_neural import train_one
from .train_traditional import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen GAN augmentation classifier conditions reproducibly.")
    parser.add_argument("--model", choices=("seqgan", "relgan"), required=True)
    parser.add_argument("--base", choices=("urls_new", "coverage_plus"), required=True)
    parser.add_argument("--percentages", default="10,20,30,40")
    parser.add_argument("--families", choices=("neural", "traditional", "all"), default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments/gan_augmentation_v1"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments/gan_augmentation_v1"))
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    percentages = [int(value) for value in args.percentages.split(",") if value]
    for percentage in percentages:
        condition = f"{args.model}_{args.base}_{percentage}pct"
        root = args.data_root / args.model / args.base / f"pct_{percentage:02d}"
        output = args.artifact_root / args.model / args.base / f"pct_{percentage:02d}"
        if args.families in {"neural", "all"}:
            for seed in project["seeds"]:
                for name in ("cnn", "lstm", "rnn"):
                    result = train_one(name, int(seed), project, root / "processed", output,
                                       args.num_workers, False, condition)
                    print(json.dumps({"condition": condition, "model": name, "seed": seed,
                                      "f1": result["f1"], "roc_auc": result["roc_auc"]}), flush=True)
        if args.families in {"traditional", "all"}:
            for seed in project["seeds"]:
                for name in ("random_forest", "linear_svm"):
                    result = train(name, int(seed), project, root / "processed", root / "features",
                                   output, condition)
                    print(json.dumps({"condition": condition, "model": name, "seed": seed,
                                      "f1": result["f1"], "roc_auc": result["roc_auc"]}), flush=True)


if __name__ == "__main__":
    main()
