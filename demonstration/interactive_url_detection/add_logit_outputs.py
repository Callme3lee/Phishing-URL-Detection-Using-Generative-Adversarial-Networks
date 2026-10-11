#!/usr/bin/env python3
"""Expose existing pre-sigmoid logits in the explainer's ONNX copies.

This does not retrain or alter model computation. It adds an Identity output
from the existing pre-sigmoid tensor, records the frozen classifier bias, and
updates the local demonstration manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MANIFEST = MODELS / "bundle_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability(path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    tokens = np.zeros((2, 128), dtype=np.int64)
    lexical = np.zeros((2, 35), dtype=np.float32)
    return session.run(["probability"], {"tokens": tokens, "lexical": lexical})[0]


def expose(path: Path) -> tuple[float, float]:
    before = probability(path)
    model = onnx.load(path)
    outputs = {item.name for item in model.graph.output}
    squeeze = next(node for node in model.graph.node if node.op_type == "Squeeze" and node.name.endswith("/model/Squeeze"))
    if "logit" not in outputs:
        model.graph.node.append(helper.make_node("Identity", [squeeze.output[0]], ["logit"], name="/explanation/LogitOutput"))
        model.graph.output.append(helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch"]))
    bias_tensor = next(item for item in model.graph.initializer if item.name == "model.classifier.bias")
    bias = float(numpy_helper.to_array(bias_tensor).reshape(-1)[0])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    after = probability(path)
    return bias, float(np.max(np.abs(before - after)))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    biases: dict[str, float] = {}
    maximum_probability_difference = 0.0
    for record in manifest["models"]:
        path = MODELS / record["filename"]
        bias, difference = expose(path)
        maximum_probability_difference = max(maximum_probability_difference, difference)
        biases[str(record["seed"])] = bias
        record["onnx_sha256"] = sha256(path)
        record["onnx_bytes"] = path.stat().st_size
        record["outputs"] = ["probability", "logit"]
    manifest["explanation_outputs"] = {
        "logit": "Existing pre-sigmoid classifier output exposed for demonstration",
        "probability_max_abs_difference_after_graph_edit": maximum_probability_difference,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (MODELS / "classifier_biases.json").write_text(
        json.dumps({"bias_by_seed": biases}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bias_by_seed": biases, "probability_max_abs_difference": maximum_probability_difference}, indent=2))


if __name__ == "__main__":
    main()
