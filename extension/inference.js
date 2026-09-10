(function (root) {
  "use strict";

  class PhishingInferenceEngine {
    constructor(runtimeBase) {
      this.runtimeBase = runtimeBase;
      this.readyPromise = null;
      this.sessions = [];
      this.preprocessor = null;
      this.policy = null;
    }

    async loadJson(relative) {
      const response = await fetch(chrome.runtime.getURL(relative));
      if (!response.ok) throw new Error(`Unable to load ${relative}: HTTP ${response.status}`);
      return response.json();
    }

    async initialize() {
      if (this.readyPromise) return this.readyPromise;
      this.readyPromise = (async () => {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
        const [manifest, vocabulary, scaler, pslRules, htmlEntities, policy] = await Promise.all([
          this.loadJson("models/bundle_manifest.json"), this.loadJson("models/vocabulary.json"),
          this.loadJson("models/scaler.json"), this.loadJson("models/psl_rules.json"),
          this.loadJson("models/html_entities.json"), this.loadJson("models/policy.json"),
        ]);
        this.policy = policy;
        this.preprocessor = PhishingPreprocessing.createPreprocessor({ vocabulary, scaler, pslRules, htmlEntities });
        this.sessions = await Promise.all(manifest.models.map((model) => ort.InferenceSession.create(
          chrome.runtime.getURL(`models/${model.filename}`),
          { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
        )));
        return this;
      })();
      return this.readyPromise;
    }

    tier(score) {
      for (const tier of this.policy.tiers) {
        const lower = tier.lower_inclusive; const upper = tier.upper_exclusive;
        if ((lower === null || score >= lower) && (upper === null || score < upper)) return tier;
      }
      throw new Error(`No tier contains score ${score}`);
    }

    async scoreSingle(rawUrl) {
      await this.initialize();
      const started = performance.now();
      const input = this.preprocessor.prepare(rawUrl);
      const tokens = new ort.Tensor("int64", input.tokens, [1, 128]);
      const lexical = new ort.Tensor("float32", input.lexical, [1, 35]);
      const seedScores = [];
      for (const session of this.sessions) {
        const output = await session.run({ tokens, lexical });
        seedScores.push(Number(output.probability.data[0]));
      }
      const score = seedScores.reduce((sum, value) => sum + value, 0) / seedScores.length;
      const tier = this.tier(score);
      return {
        rawUrl, canonicalUrl: input.canonicalUrl, hostname: input.hostname, score, seedScores,
        tier: tier.name, action: tier.action,
        primaryPrediction: score >= this.policy.primary_binary_rule.threshold ? "phishing" : "legitimate",
        inferenceMilliseconds: performance.now() - started,
        modelStatus: this.policy.status,
        explanation: {
          method: "hybrid_input_diagnostics_v1",
          signals: input.signals,
          note: "Signals are unusual lexical inputs relative to the frozen training distribution; they do not prove causation or phishing.",
        },
      };
    }

    async scoreUrl(rawUrl, options = {}) {
      const start = performance.now();
      let full;
      try { full = await this.scoreSingle(rawUrl); }
      catch (error) {
        // Long login URLs can exceed the frozen model's 4096-character limit.
        // Retain an explicit unavailable score; do not invent a model prediction.
        if (options.allowUnavailable !== true) throw error;
        full = { rawUrl, canonicalUrl: rawUrl.replace(/^https:\/\//, ""), hostname: PhishingBrowserPolicy.hostname(rawUrl),
          score: null, seedScores: [], tier: "unknown", primaryPrediction: null,
          modelError: String(error?.message ?? error), modelStatus: "unavailable" };
      }
      let normalized = { applied: false };
      try {
        const view = PhishingBrowserPolicy.queryView(rawUrl);
        normalized = { applied: view.applied, parameterCount: view.parameterCount,
          originalQueryLength: view.originalQueryLength, removedValueCharacters: view.removedValueCharacters };
        if (view.applied) {
          const result = await this.scoreSingle(view.url);
          normalized.score = result.score;
          normalized.tier = result.tier;
        }
      } catch (error) { normalized.error = String(error?.message ?? error); }
      const signals = [...(full.explanation?.signals ?? [])];
      if (Number.isFinite(full.score) && Number.isFinite(normalized.score)
          && full.score - normalized.score >= 0.1) {
        signals.unshift({ code: "query_increased_score", strength: 12,
          values: { difference: (full.score - normalized.score).toFixed(2) } });
      }
      const finiteSeeds = (full.seedScores ?? []).filter(Number.isFinite);
      const ensembleSpread = finiteSeeds.length ? Math.max(...finiteSeeds) - Math.min(...finiteSeeds) : null;
      return { ...full, explanation: { ...(full.explanation ?? {}), signals: signals.slice(0, 4), ensembleSpread },
        queryView: normalized, inferenceMilliseconds: performance.now() - start,
        modelViewCount: (Number.isFinite(full.score) ? 1 : 0) + (Number.isFinite(normalized.score) ? 1 : 0) };
    }
  }

  root.PhishingInferenceEngine = PhishingInferenceEngine;
})(globalThis);
