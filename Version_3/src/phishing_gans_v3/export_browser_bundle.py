from __future__ import annotations

import argparse
import csv
import hashlib
import html.entities
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .common import load_json, sha256_file, write_json
from .evaluate_hybrid_external import create_model
from .tiered_url_inference import assign_tier
from .url_data import (
    PSL_AWARE_LEXICAL_FEATURE_NAMES,
    PSL_EXTRACTOR,
    canonicalize_url,
    encode_url,
    psl_aware_lexical_features,
)


ROOT = Path("browser_extension")
MODEL_DIR = ROOT / "models"
POLICY = Path("artifacts/deployment_candidates/conservative_hyb_gan10_v1/policy.json")
DEFAULT_FIXTURES = (
    "https://example.com/",
    "http://www.example.co.uk/login?continue=%2Faccount",
    "https://accounts.example.com:8443/verify/session?id=12345&next=wallet",
    "http://192.0.2.10/update/password",
    "https://xn--bcher-kva.example/path?q=abc",
    "https://deep.subdomain.example.org/a/b/c/d?x=1;y=2",
    "https://example.com/%70%61%79%6d%65%6e%74?account=secure&verify=true",
    "https://foo.github.io/project/login",
    "https://localhost:3000/test",
    "example.invalidsuffix/query?a=1&b=2",
    "https://user:password@example.net/private",
    "https://example.com/path?redirect=https%3A%2F%2Ftarget.example%2F",
)


class ProbabilityWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, lexical: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(tokens, lexical))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the frozen HYB-MS+10% ensemble for browser inference.")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def export_one(saved: dict, output: Path, opset: int) -> tuple[np.ndarray, dict]:
    model = create_model(saved).cpu()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    wrapped = ProbabilityWrapper(model).eval()
    tokens = torch.zeros((2, 128), dtype=torch.int64)
    lexical = torch.zeros((2, 35), dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (tokens, lexical),
        output,
        input_names=("tokens", "lexical"),
        output_names=("probability",),
        dynamic_axes={"tokens": {0: "batch"}, "lexical": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    return np.asarray(saved["lexical_scaler_center"], dtype=np.float32), {
        "onnx_sha256": sha256_file(output),
        "onnx_bytes": output.stat().st_size,
        "opset": opset,
    }


def main() -> None:
    args = parse_args()
    global ROOT, MODEL_DIR
    ROOT = args.output
    MODEL_DIR = ROOT / "models"
    manifest_path = MODEL_DIR / "bundle_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Browser bundle already exists: {manifest_path}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_json(args.policy)
    reference_vocabulary = None
    reference_center = None
    reference_scale = None
    reference_features = None
    model_records = []
    torch_models: list[nn.Module] = []

    for item in policy["checkpoints"]:
        checkpoint = Path(item["path"])
        if sha256_file(checkpoint) != item["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        vocabulary = saved["vocabulary"]
        center = np.asarray(saved["lexical_scaler_center"], dtype=np.float32)
        scale = np.where(np.asarray(saved["lexical_scaler_scale"], dtype=np.float32) == 0, 1.0,
                         np.asarray(saved["lexical_scaler_scale"], dtype=np.float32))
        features = list(saved["lexical_feature_names"])
        if features != list(PSL_AWARE_LEXICAL_FEATURE_NAMES):
            raise RuntimeError("Checkpoint lexical feature order differs from the deployed extractor")
        if reference_vocabulary is None:
            reference_vocabulary, reference_center, reference_scale, reference_features = vocabulary, center, scale, features
        elif vocabulary != reference_vocabulary or not np.array_equal(center, reference_center) or not np.array_equal(scale, reference_scale):
            raise RuntimeError("Ensemble checkpoints do not share one preprocessing contract")
        model = create_model(saved).cpu(); model.load_state_dict(saved["state_dict"]); model.eval()
        torch_models.append(model)
        onnx_path = MODEL_DIR / f"hyb_gan10_seed{item['seed']}.onnx"
        _, exported = export_one(saved, onnx_path, args.opset)
        model_records.append({
            "seed": int(item["seed"]), "filename": onnx_path.name,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": item["sha256"], **exported,
        })

    assert reference_vocabulary is not None and reference_center is not None and reference_scale is not None
    stable_json(MODEL_DIR / "vocabulary.json", reference_vocabulary)
    stable_json(MODEL_DIR / "scaler.json", {
        "feature_names": reference_features,
        "center": reference_center.tolist(), "scale": reference_scale.tolist(),
        "transform": "(raw_feature - center) / scale",
    })
    browser_policy = {
        "policy_version": policy["policy_version"], "model": policy["model"],
        "status": policy["status"], "maximum_characters": 128,
        "ensemble_rule": policy["ensemble_rule"], "primary_binary_rule": policy["primary_binary_rule"],
        "tiers": policy["tiers"], "guardrails": policy["guardrails"],
        "source_policy_sha256": sha256_file(args.policy),
    }
    stable_json(MODEL_DIR / "policy.json", browser_policy)
    rules = sorted(PSL_EXTRACTOR.tlds)
    stable_json(MODEL_DIR / "psl_rules.json", {
        "rules": rules,
        "include_private_domains": True,
        "source": "tldextract bundled Public Suffix List snapshot used by Version 3",
    })
    stable_json(MODEL_DIR / "html_entities.json", html.entities.html5)

    normalized = [canonicalize_url(raw)[0] for raw in DEFAULT_FIXTURES]
    raw_features = np.stack([psl_aware_lexical_features(url) for url in normalized]).astype(np.float32)
    scaled = ((raw_features - reference_center) / reference_scale).astype(np.float32)
    tokens = np.stack([encode_url(url, reference_vocabulary, 128) for url in normalized]).astype(np.int64)
    torch_scores = []
    ort_scores = []
    for model, record in zip(torch_models, model_records):
        with torch.inference_mode():
            score = torch.sigmoid(model(torch.from_numpy(tokens), torch.from_numpy(scaled))).numpy().astype(np.float32)
        torch_scores.append(score)
        session = ort.InferenceSession(str(MODEL_DIR / record["filename"]), providers=["CPUExecutionProvider"])
        ort_score = session.run(["probability"], {"tokens": tokens, "lexical": scaled})[0].astype(np.float32)
        ort_scores.append(ort_score)
        record["fixture_max_abs_torch_onnx_difference"] = float(np.max(np.abs(score - ort_score)))
    maximum_difference = float(np.max(np.abs(np.stack(torch_scores) - np.stack(ort_scores))))
    if maximum_difference > 1e-5:
        raise RuntimeError(f"ONNX parity exceeded 1e-5: {maximum_difference}")
    ensemble = np.mean(np.stack(ort_scores), axis=0).astype(np.float32)
    fixtures = []
    for index, raw in enumerate(DEFAULT_FIXTURES):
        canonical, hostname, scheme = canonicalize_url(raw)
        fixtures.append({
            "raw_url": raw, "canonical_url": canonical, "hostname": hostname, "scheme": scheme,
            "tokens": tokens[index].tolist(), "raw_lexical": raw_features[index].tolist(),
            "scaled_lexical": scaled[index].tolist(),
            "seed_scores": [float(scores[index]) for scores in ort_scores],
            "ensemble_score": float(ensemble[index]), "tier": assign_tier(float(ensemble[index]), policy["tiers"]),
        })
    stable_json(ROOT / "tests" / "parity_fixtures.json", {
        "score_tolerance": 1e-5, "feature_tolerance": 1e-6, "fixtures": fixtures,
    })
    manifest = {
        "bundle_version": "hyb_gan10_browser_v1", "project_version": 3,
        "candidate": policy["model"], "models": model_records,
        "preprocessing": {
            "vocabulary_sha256": sha256_file(MODEL_DIR / "vocabulary.json"),
            "scaler_sha256": sha256_file(MODEL_DIR / "scaler.json"),
            "psl_rules_sha256": sha256_file(MODEL_DIR / "psl_rules.json"),
            "html_entities_sha256": sha256_file(MODEL_DIR / "html_entities.json"),
            "policy_sha256": sha256_file(MODEL_DIR / "policy.json"),
        },
        "parity": {"fixtures": len(fixtures), "maximum_absolute_torch_onnx_difference": maximum_difference},
        "limitations": [
            "This is a research browser prototype and not a production-ready security product.",
            "The primary model uses URL text only; DNS is intentionally not included in this bundle.",
            "Browser preprocessing must pass the committed parity fixtures before evaluation.",
        ],
    }
    stable_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
