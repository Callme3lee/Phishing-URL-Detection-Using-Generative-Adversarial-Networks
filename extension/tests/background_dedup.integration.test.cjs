const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");
const Pilot = require("../pilot_store.js");

const ROOT = path.join(__dirname, "..");

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function storageArea(backing) {
  return {
    async get(keys) {
      if (typeof keys === "string") return { [keys]: clone(backing[keys]) };
      if (Array.isArray(keys)) return Object.fromEntries(keys.map((key) => [key, clone(backing[key])]));
      if (keys && typeof keys === "object") {
        return Object.fromEntries(Object.entries(keys).map(([key, fallback]) => [
          key, backing[key] === undefined ? fallback : clone(backing[key]),
        ]));
      }
      return clone(backing);
    },
    async set(values) { Object.assign(backing, clone(values)); },
    async remove(keys) {
      for (const key of Array.isArray(keys) ? keys : [keys]) delete backing[key];
    },
  };
}

function canonical(rawUrl) {
  const parsed = new URL(rawUrl);
  return `${parsed.hostname}${parsed.port ? `:${parsed.port}` : ""}${parsed.pathname}${parsed.search}${parsed.hash}`;
}

function makeWorker(localBacking, sessionBacking) {
  const listeners = { before: [], history: [], fragment: [], removed: [], message: [], installed: [] };
  let inferenceCalls = 0;
  const chrome = {
    runtime: {
      getManifest: () => ({ version: "0.5.1" }),
      getURL: (relative) => `chrome-extension://test/${relative}`,
      getContexts: async () => [{ contextType: "OFFSCREEN_DOCUMENT" }],
      sendMessage: async (message) => {
        if (message?.target !== "offscreen") throw new Error("Unexpected internal message");
        inferenceCalls += 1;
        return {
          ok: true,
          result: {
            rawUrl: message.url,
            canonicalUrl: canonical(message.url),
            hostname: new URL(message.url).hostname,
            score: 0.1,
            seedScores: [0.1, 0.1, 0.1],
            tier: "allow",
            action: "allow",
            primaryPrediction: "legitimate",
            inferenceMilliseconds: 5,
            modelStatus: "test",
          },
        };
      },
      onInstalled: { addListener: (listener) => listeners.installed.push(listener) },
      onMessage: { addListener: (listener) => listeners.message.push(listener) },
      openOptionsPage: async () => {},
    },
    offscreen: { createDocument: async () => {} },
    storage: {
      local: storageArea(localBacking),
      session: storageArea(sessionBacking),
    },
    action: {
      setBadgeText: async () => {},
      setBadgeBackgroundColor: async () => {},
    },
    tabs: {
      update: async () => {},
      sendMessage: async () => {},
      create: async () => {},
      onRemoved: { addListener: (listener) => listeners.removed.push(listener) },
    },
    webNavigation: {
      onBeforeNavigate: { addListener: (listener) => listeners.before.push(listener) },
      onHistoryStateUpdated: { addListener: (listener) => listeners.history.push(listener) },
      onReferenceFragmentUpdated: { addListener: (listener) => listeners.fragment.push(listener) },
    },
  };

  const contextObject = {
    chrome,
    console,
    crypto: webcrypto,
    TextEncoder,
    URL,
    URLSearchParams,
    setTimeout,
    clearTimeout,
  };
  const context = vm.createContext(contextObject);
  contextObject.globalThis = context;
  contextObject.importScripts = (...scripts) => {
    for (const script of scripts) {
      const source = fs.readFileSync(path.join(ROOT, script), "utf8");
      vm.runInContext(source, context, { filename: script });
    }
  };
  vm.runInContext(fs.readFileSync(path.join(ROOT, "background.js"), "utf8"), context, {
    filename: "background.js",
  });

  async function waitFor(predicate, timeout = 1000) {
    const started = Date.now();
    while (!predicate()) {
      if (Date.now() - started > timeout) throw new Error("Timed out waiting for background mutation");
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
  }

  async function trigger(type, details, expectedUnique) {
    for (const listener of listeners[type]) listener(details);
    await waitFor(() => localBacking.pilotStudy?.counters?.uniqueAutomaticUrls === expectedUnique);
    await new Promise((resolve) => setTimeout(resolve, 15));
  }

  async function sendExternal(message, sender) {
    const listener = listeners.message[0];
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Background message timed out")), 1000);
      const sendResponse = (response) => { clearTimeout(timeout); resolve(response); };
      const keepAlive = listener(message, sender, sendResponse);
      if (keepAlive !== true) {
        clearTimeout(timeout);
        resolve(undefined);
      }
    });
  }

  return { listeners, trigger, sendExternal, inferenceCalls: () => inferenceCalls };
}

(async () => {
  const localBacking = {
    pilotParticipation: true,
    pilotStudy: Pilot.createStudy("integration-user", "2026-08-28T00:00:00.000Z", "0.5.1"),
  };
  const sessionBacking = {};
  let worker = makeWorker(localBacking, sessionBacking);

  await worker.trigger("before", {
    tabId: 7, frameId: 0, url: "https://example.com/path?x=1#first",
  }, 1);
  assert.equal(localBacking.pilotStudy.counters.automaticInferenceExecutions, 1);
  assert.equal(localBacking.pilotStudy.counters.duplicateAssessmentsSuppressed, 0);

  await worker.trigger("fragment", {
    tabId: 7, frameId: 0, url: "https://example.com/path?x=1#second",
  }, 1);
  assert.equal(localBacking.pilotStudy.counters.duplicateAssessmentsSuppressed, 1);
  assert.equal(localBacking.pilotStudy.duplicateEventCounts.fragment, 1);
  assert.equal(worker.inferenceCalls(), 2, "changed fragment may be rescored but must not become a new URL");

  await worker.trigger("history", {
    tabId: 7, frameId: 0, url: "https://example.com/path?x=2",
  }, 2);
  assert.equal(localBacking.pilotStudy.decisionCounts.allow, 2);

  const executionsBeforeRestart = localBacking.pilotStudy.counters.automaticInferenceExecutions;
  worker = makeWorker(localBacking, sessionBacking);
  const advisory = await worker.sendExternal(
    { type: "GET_PAGE_ADVISORY", url: "https://example.com/path?x=2" },
    { tab: { id: 7 }, frameId: 0, url: "https://example.com/path?x=2" },
  );
  assert.equal(advisory.ok, true);
  assert.equal(worker.inferenceCalls(), 0, "session cache must survive a service-worker restart");
  assert.equal(localBacking.pilotStudy.counters.automaticInferenceExecutions, executionsBeforeRestart);
  assert.equal(localBacking.pilotStudy.counters.uniqueAutomaticUrls, 2);

  await worker.trigger("before", {
    tabId: 8, frameId: 0, url: "https://example.com/path?x=2",
  }, 2);
  assert.equal(localBacking.pilotStudy.counters.duplicateAssessmentsSuppressed, 2);
  assert.equal(localBacking.pilotStudy.duplicateEventCounts.navigation, 1);
  assert.equal(localBacking.pilotStudy.counters.uniqueAutomaticUrls, 2, "same canonical URL in a new tab stays deduplicated");

  assert.ok(Array.isArray(localBacking.pilotDedupFingerprintsV2));
  assert.equal(localBacking.pilotDedupFingerprintsV2.length, 2);
  assert.equal("pilotDedupFingerprintsV2" in localBacking.pilotStudy, false, "fingerprints are outside exported study data");
  assert.equal("pilotDedupSaltV2" in localBacking.pilotStudy, false, "salt is outside exported study data");

  console.log("PASS: repeated URLs, fragments, new tabs, advisory recovery and worker restarts are deduplicated dynamically.");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
