from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_json, write_json
from .train_neural import train_one
from .train_traditional import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen MS-HardGAN-C classifier conditions.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/experiments/ms_hardgan_augmentation_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/ms_hardgan_augmentation_v2"))
    parser.add_argument("--percentages", default="10,20,30,40")
    parser.add_argument("--families", choices=("neural", "traditional", "all"), default="all")
    parser.add_argument("--models", default="cnn,lstm,rnn,random_forest,linear_svm")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def record_training_composition(result: dict, run_dir: Path, experiment: dict) -> dict:
    """Keep real and synthetic row counts unambiguous in every saved run."""
    result["real_training_rows"] = int(experiment["base_real_rows"])
    result["synthetic_training_rows"] = int(experiment["generated_phishing_rows"])
    result["training_rows"] = int(experiment["final_training_rows"])
    result["generated_fraction"] = float(experiment["generated_fraction"])
    write_json(run_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args(); project = load_json(args.config)
    percentages = [int(value) for value in args.percentages.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    selected = {value.strip() for value in args.models.split(",") if value.strip()}
    for percentage in percentages:
        condition = f"ms_hardgan_v2_{percentage}pct"
        root = args.data_root / f"pct_{percentage:02d}"
        output = args.output_root / f"pct_{percentage:02d}"
        experiment = load_json(root / "experiment_report.json")
        if args.families in {"neural", "all"}:
            for seed in seeds:
                for name in ("cnn", "lstm", "rnn"):
                    if name not in selected: continue
                    result = train_one(name, seed, project, root / "processed", output,
                                       args.num_workers, False, condition)
                    result = record_training_composition(result, output / name / f"seed_{seed}", experiment)
                    print(json.dumps({"condition": condition, "model": name, "seed": seed,
                                      "f1": result["f1"], "roc_auc": result["roc_auc"]}), flush=True)
        if args.families in {"traditional", "all"}:
            for seed in seeds:
                for name in ("random_forest", "linear_svm"):
                    if name not in selected: continue
                    result = train(name, seed, project, root / "processed", root / "features", output, condition)
                    result = record_training_composition(result, output / name / f"seed_{seed}", experiment)
                    print(json.dumps({"condition": condition, "model": name, "seed": seed,
                                      "f1": result["f1"], "roc_auc": result["roc_auc"]}), flush=True)


if __name__ == "__main__":
    main()
