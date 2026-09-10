const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ort = require("onnxruntime-web");
const { createPreprocessor } = require("../preprocessing.js");

const root = path.resolve(__dirname, "..");
const read = (relative) => JSON.parse(fs.readFileSync(path.join(root, relative), "utf8"));
const preprocessor = createPreprocessor({
  vocabulary: read("models/vocabulary.json"), scaler: read("models/scaler.json"),
  pslRules: read("models/psl_rules.json"), htmlEntities: read("models/html_entities.json"),
});
const bundle = read("models/bundle_manifest.json");
const fixtures = read("tests/parity_fixtures.json");

async function main() {
  ort.env.wasm.numThreads = 1;
  if (process.env.ORT_WASM_PATH) ort.env.wasm.wasmPaths = process.env.ORT_WASM_PATH;
  const initializationStarted = performance.now();
  const sessions = await Promise.all(bundle.models.map((model) => ort.InferenceSession.create(
    path.join(root, "models", model.filename), { executionProviders: ["wasm"] },
  )));
  const initializationMilliseconds = performance.now() - initializationStarted;
  let maximum = 0;
  const latencies = [];
  for (const fixture of fixtures.fixtures) {
    const started = performance.now();
    const input = preprocessor.prepare(fixture.raw_url);
    assert.ok(Array.isArray(input.signals));
    const tokens = new ort.Tensor("int64", input.tokens, [1, 128]);
    const lexical = new ort.Tensor("float32", input.lexical, [1, 35]);
    const scores = [];
    for (const session of sessions) scores.push(Number((await session.run({ tokens, lexical })).probability.data[0]));
    latencies.push(performance.now() - started);
    scores.forEach((score, index) => { maximum = Math.max(maximum, Math.abs(score - fixture.seed_scores[index])); });
  }
  assert.ok(maximum <= fixtures.score_tolerance, `ONNX Runtime Web score difference ${maximum}`);
  const sorted = [...latencies].sort((a, b) => a - b);
  const percentile = (fraction) => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
  const report = {
    measured_at_utc: new Date().toISOString(), runtime: "onnxruntime-web/WASM", model_count: sessions.length,
    fixture_count: latencies.length, initialization_milliseconds: initializationMilliseconds,
    sequential_url_latency_milliseconds: {
      mean: latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
      p50: percentile(0.5), p95: percentile(0.95), maximum: sorted.at(-1),
    },
    maximum_seed_score_difference: maximum,
    note: "Node.js host benchmark of the same WebAssembly runtime; interactive Chrome timing remains device-dependent.",
  };
  fs.writeFileSync(path.join(root, "tests", "wasm_benchmark.json"), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`PASS: ONNX Runtime Web ensemble parity maximum difference ${maximum}.`);
  console.log(`WASM: init ${initializationMilliseconds.toFixed(1)} ms; per URL mean ${report.sequential_url_latency_milliseconds.mean.toFixed(1)} ms, p95 ${report.sequential_url_latency_milliseconds.p95.toFixed(1)} ms.`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
