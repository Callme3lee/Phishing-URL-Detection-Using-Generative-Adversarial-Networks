from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishing_gans_v3_mpl")
import joblib
import numpy as np
import torch

from .common import load_json, require_cuda, sha256_file, write_json
from .evaluate_hybrid_external import create_model as create_hybrid
from .metrics import classification_metrics, select_threshold
from .models import create_classifier
from .url_data import encode_url, lexical_features, psl_aware_lexical_features


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "false_positive_rate", "false_negative_rate")
SEEDS = (42, 43, 44)
HYBRID_NAMES = {"hybrid_character_cnn", "hybrid_cnn_coral", "cnn_psl", "hyb_ms"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate every frozen classifier with one equal calibration protocol.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/all_models_equal_calibration_v1"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--maximum-fpr", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["url"] for row in rows], np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)


def frozen_datasets():
    specs = {
        "calibration": (Path("data/calibration/2026-08-11-post-hybrid/calibration_urls.csv"),
                        Path("data/calibration/2026-08-11-post-hybrid/collection_report.json")),
        "external_10k": (Path("data/external/2026-08-12-large-frozen/external_urls.csv"),
                         Path("data/external/2026-08-12-large-frozen/collection_report.json")),
        "external_30k": (Path("data/external/2026-08-13-imbalanced-30k/external_urls.csv"),
                         Path("data/external/2026-08-13-imbalanced-30k/collection_report.json")),
    }
    result = {}
    for name, (path, report_path) in specs.items():
        report = load_json(report_path)
        if sha256_file(path) != report["dataset_sha256"]:
            raise RuntimeError(f"Frozen hash mismatch for {name}")
        urls, labels = read_csv(path)
        if len(urls) != int(report["rows"]):
            raise RuntimeError(f"Frozen row mismatch for {name}")
        result[name] = {"path": path, "report": report, "urls": urls, "labels": labels}
    internal_path = Path("data/processed/test_manifest.csv")
    urls, labels = read_csv(internal_path)
    result["internal_test"] = {"path": internal_path, "report": None, "urls": urls, "labels": labels}
    return result


def config_roots() -> list[dict]:
    files = sorted(Path("artifacts").glob("**/checkpoint.pt")) + sorted(Path("artifacts").glob("**/model.joblib"))
    grouped = {}
    for path in files:
        match = re.search(r"/seed_(42|43|44)/(checkpoint\.pt|model\.joblib)$", "/" + path.as_posix())
        if not match:
            continue
        seed = int(match.group(1)); root = path.parent.parent
        grouped.setdefault(root, {})[seed] = path
    records = []
    for root, paths in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        if set(paths) != set(SEEDS):
            continue
        relative = root.relative_to("artifacts").as_posix()
        slug = re.sub(r"[^a-zA-Z0-9]+", "__", relative).strip("_").lower()
        records.append({"root": root, "relative": relative, "slug": slug, "paths": paths,
                        "format": "neural" if paths[42].suffix == ".pt" else "traditional"})
    return records


def stage_and_display(relative: str, saved: dict) -> tuple[str, str, str]:
    model = saved.get("model")
    if not isinstance(model, str):
        model = "random_forest" if "random_forest" in relative else "linear_svm"
    display_model = {"cnn":"CNN", "lstm":"LSTM", "rnn":"RNN", "random_forest":"RF",
                     "linear_svm":"SVM", "cnn_psl":"CNN+PSL", "hyb_ms":"HYB-MS",
                     "hybrid_cnn_coral":"HYB-MS", "hybrid_character_cnn":"HYB"}.get(model, model)
    if relative.startswith("baselines/"): stage, condition = "ORIG baseline", "Real"
    elif "legitimate_coverage_v1" in relative: stage, condition = "Coverage", "Real + outside legit"
    elif "urls_new_structure_balanced_v1" in relative: stage, condition = "Structure-balanced", "Real"
    elif "multisource_baselines_v1" in relative: stage, condition = "Ordinary MS", "Real"
    elif "gan_augmentation_v1" in relative:
        stage = "SeqGAN v1"
        base = "Coverage" if "coverage_plus" in relative else "ORIG"
        pct = re.search(r"pct_(\d+)", relative).group(1)
        condition = f"{base} +{pct}%"
    elif "ms_hardgan_augmentation_v2" in relative:
        stage = "MS-HardGAN-C baselines"; condition = "+" + re.search(r"pct_(\d+)", relative).group(1) + "%"
    elif "fair_hybrid_gan_v1" in relative:
        stage = "Feature-equal / weighted HYB"
        match = re.search(r"/(real_only|gan_(\d+)pct)/", "/" + relative + "/")
        condition = "Real" if match and match.group(1) == "real_only" else f"+{match.group(2)}%"
    elif "hybrid_original_v1" in relative: stage, condition = "HYB-ORIG ablation", "Real"
    elif "hybrid_cnn_coral_v1" in relative: stage, condition = "Weighted HYB-MS", "Real"
    elif "hybrid_ms_natural" in relative:
        stage = "Natural HYB-MS"
        match = re.search(r"gan_(\d+)pct", relative); condition = f"+{match.group(1)}%" if match else "Real"
    else: stage, condition = "Other", saved.get("condition", relative)
    return stage, display_model, condition


def vocabulary_key(vocabulary: dict) -> str:
    return hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def neural_score(model, tokens, lexical, batch_size, device):
    model.eval(); output=[]
    for start in range(0, len(tokens), batch_size):
        x = torch.from_numpy(tokens[start:start+batch_size].astype(np.int64, copy=False)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if lexical is None:
                logits = model(x)
            else:
                z = torch.from_numpy(lexical[start:start+batch_size]).to(device)
                logits = model(x, z)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def split_scores(scores, lengths, names):
    result={}; offset=0
    for name,length in zip(names,lengths):
        result[name]=scores[offset:offset+length]; offset += length
    return result


def fair_psl_features(relative: str, raw_psl: np.ndarray) -> np.ndarray:
    condition_match = re.search(r"fair_hybrid_gan_v1/(real_only|gan_\d+pct)", relative)
    condition = condition_match.group(1) if condition_match else "real_only"
    scaler = np.load(Path("data/experiments/fair_hybrid_gan_v1") / condition / "processed/psl_lexical_scaler.npz")
    scale = np.asarray(scaler["scale"], dtype=np.float32); scale = np.where(scale == 0, 1.0, scale)
    return ((raw_psl - np.asarray(scaler["center"], dtype=np.float32)) / scale).astype(np.float32)


def main():
    args=parse_args(); args.output.mkdir(parents=True, exist_ok=True); per_dir=args.output/"per_condition"; per_dir.mkdir(exist_ok=True)
    datasets=frozen_datasets(); names=("calibration","internal_test","external_10k","external_30k")
    lengths=[len(datasets[name]["urls"]) for name in names]
    all_urls=sum((datasets[name]["urls"] for name in names), [])
    all_labels={name:datasets[name]["labels"] for name in names}
    print(json.dumps({"stage":"feature_preparation","rows":len(all_urls)}),flush=True)
    raw_lexical=np.stack([lexical_features(url) for url in all_urls]).astype(np.float32)
    raw_psl=np.stack([psl_aware_lexical_features(url) for url in all_urls]).astype(np.float32)
    device=require_cuda(); token_cache={}; records=[]; configs=config_roots()
    print(json.dumps({"stage":"inventory","configurations":len(configs),"checkpoints":3*len(configs)}),flush=True)
    for index,config in enumerate(configs,1):
        result_path=per_dir/f"{config['slug']}.json"
        if result_path.exists() and not args.overwrite:
            records.append(load_json(result_path)); print(json.dumps({"stage":"skip","index":index,"slug":config["slug"]}),flush=True); continue
        seed_outputs=[]; checkpoint_records=[]; first_saved=None
        for seed in SEEDS:
            path=config["paths"][seed]
            if config["format"] == "neural":
                saved=torch.load(path,map_location=device,weights_only=False); first_saved=first_saved or saved
                model_name=saved["model"]; vocab=saved["vocabulary"]; key=vocabulary_key(vocab)
                if key not in token_cache:
                    token_cache[key]=np.stack([encode_url(url,vocab,128) for url in all_urls])
                tokens=token_cache[key]
                if model_name in HYBRID_NAMES:
                    model=create_hybrid(saved).to(device)
                    scale=np.asarray(saved["lexical_scaler_scale"],dtype=np.float32); scale=np.where(scale==0,1.0,scale)
                    lexical=((raw_psl-np.asarray(saved["lexical_scaler_center"],dtype=np.float32))/scale).astype(np.float32)
                else:
                    model=create_classifier(model_name,len(vocab),saved["model_config"]).to(device)
                model.load_state_dict(saved["state_dict"])
                if model_name in HYBRID_NAMES:
                    scores=neural_score(model,tokens,lexical,args.batch_size,device)
                else:
                    scores=neural_score(model,tokens,None,args.batch_size,device)
                del model; gc.collect(); torch.cuda.empty_cache()
            else:
                saved=joblib.load(path); first_saved=first_saved or saved; model=saved["model"]
                features=fair_psl_features(config["relative"],raw_psl) if "fair_hybrid_gan_v1" in config["relative"] else raw_lexical
                scores=(model.predict_proba(features)[:,1] if "random_forest" in config["relative"] else model.decision_function(features)).astype(np.float32)
            seed_outputs.append(scores); checkpoint_records.append({"seed":seed,"path":str(path),"sha256":sha256_file(path)})
        ensemble=np.mean(np.stack(seed_outputs),axis=0); split=split_scores(ensemble,lengths,names)
        selected=select_threshold(all_labels["calibration"],split["calibration"],maximum_fpr=args.maximum_fpr)
        threshold=float(selected["threshold"]); evaluations={name:classification_metrics(all_labels[name],split[name],threshold) for name in names}
        stage,model_display,condition=stage_and_display(config["relative"],first_saved)
        record={"id":config["slug"],"root":str(config["root"]),"stage":stage,"model":model_display,"condition":condition,
                "format":config["format"],"seeds":list(SEEDS),"checkpoints":checkpoint_records,"threshold":threshold,
                "threshold_policy":f"three-seed mean score; maximum calibration F1 subject to calibration FPR <= {args.maximum_fpr:.2%}",
                "threshold_selection":selected,"evaluations":evaluations}
        write_json(result_path,record); records.append(record)
        print(json.dumps({"stage":"completed","index":index,"total":len(configs),"id":record["id"],"model":model_display,
                          "condition":condition,"external_30k_f1":evaluations["external_30k"]["f1"],
                          "external_30k_fpr":evaluations["external_30k"]["false_positive_rate"]}),flush=True)
    payload={"project_version":3,"protocol":"equal calibration across every valid frozen three-seed classifier configuration",
             "maximum_calibration_fpr":args.maximum_fpr,"configuration_count":len(records),"checkpoint_count":3*len(records),
             "datasets":{name:{"path":str(datasets[name]["path"]),"rows":len(datasets[name]["urls"]),
                               "sha256":sha256_file(datasets[name]["path"])} for name in names},"conditions":records}
    write_json(args.output/"all_results.json",payload)
    with (args.output/"all_results.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["id","stage","model","condition","threshold"]+[f"{ds}_{metric}" for ds in names for metric in METRICS]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for record in records:
            writer.writerow({"id":record["id"],"stage":record["stage"],"model":record["model"],"condition":record["condition"],
                             "threshold":record["threshold"],**{f"{ds}_{metric}":record["evaluations"][ds][metric] for ds in names for metric in METRICS}})
    print(json.dumps({"stage":"done","configurations":len(records),"output":str(args.output)},indent=2),flush=True)


if __name__ == "__main__":
    main()
