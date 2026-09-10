from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v2_mpl")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_json, write_json


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate")
DISPLAY = {"cnn":"Character CNN","lstm":"Character LSTM","random_forest":"Random Forest","linear_svm":"Linear SVM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/baselines"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); grouped = defaultdict(list)
    for path in sorted(args.input_dir.glob("*/seed_*/metrics.json")):
        result = load_json(path); grouped[result["model"]].append(result)
    if not grouped:
        raise RuntimeError("No completed baseline metrics found")
    summary = {"project_version":2,"conditions":{}}
    rows = []
    for model, runs in grouped.items():
        item = {"runs":len(runs),"seeds":[run["seed"] for run in runs]}
        for metric in METRICS:
            values=np.asarray([run[metric] for run in runs],dtype=float)
            item[metric]={"mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else 0.0,"values":values.tolist()}
        summary["conditions"][model]=item
        rows.append({"model":model,**{metric:item[metric]["mean"] for metric in METRICS}})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/"summary.json",summary)
    with (args.output_dir/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["model",*METRICS]);writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,5,figsize=(16,4.6))
    order=[name for name in DISPLAY if name in grouped]
    for axis,metric in zip(axes,("accuracy","precision","recall","f1","roc_auc")):
        means=[100*summary["conditions"][name][metric]["mean"] for name in order]
        errors=[100*summary["conditions"][name][metric]["std"] for name in order]
        axis.bar(range(len(order)),means,yerr=errors,capsize=3,color=["#2b6cb0","#319795","#dd6b20","#805ad5"][:len(order)])
        axis.set_xticks(range(len(order)),[DISPLAY[name].replace("Character ","") for name in order],rotation=25,ha="right")
        axis.set_title(metric.replace("_"," ").upper());axis.set_ylim(0,100);axis.grid(axis="y",alpha=.25)
    axes[0].set_ylabel("Score (%)");fig.suptitle("Version 2 real-only baselines — one frozen dataset")
    fig.tight_layout();fig.savefig(args.output_dir/"comparison.png",dpi=200,bbox_inches="tight");fig.savefig(args.output_dir/"comparison.pdf",bbox_inches="tight");plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
