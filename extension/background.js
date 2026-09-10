importScripts("preprocessing.js", "browser_policy.js", "pilot_store.js");

// Page content scripts communicate through the restricted message interface;
// they do not need direct access to study records or trusted-host settings.
chrome.storage.local.setAccessLevel?.({ accessLevel: "TRUSTED_CONTEXTS" })
  ?.catch((error) => console.error("Unable to restrict local storage access", error));

const activeScans = new Map();
const pageResults = new Map();
const navigations = new Map();
const presentations = new Map();
let settingsQueue = Promise.resolve();
let measurementGeneration = 0;
let creatingOffscreen = null;
let pilotMutationQueue = Promise.resolve();
let sitePolicyPromise = null;

const pilotVersion = chrome.runtime.getManifest().version;
const DEDUP_LEDGER_KEY = "pilotIdentitiesV6";
const DEDUP_SALT_KEY = "pilotDedupSaltV6";
const MAX_DEDUP_FINGERPRINTS = 25000;

async function sha256Hex(text) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...digest].map(value => value.toString(16).padStart(2, "0")).join("");
}

async function loadSitePolicy() {
  const [policyResponse, rankingResponse, pslResponse] = await Promise.all([
    fetch(chrome.runtime.getURL("policy/site_policy.json")),
    fetch(chrome.runtime.getURL("policy/tranco_XN6XN_top250.csv")),
    fetch(chrome.runtime.getURL("models/psl_rules.json")),
  ]);
  if (!policyResponse.ok || !rankingResponse.ok || !pslResponse.ok) throw new Error("Site-policy files could not be loaded.");
  const policy = await policyResponse.json();
  const rankingText = await rankingResponse.text();
  const psl = await pslResponse.json();
  if (policy.schema_version !== 1 || policy.policy_version !== PhishingBrowserPolicy.POLICY_VERSION) {
    throw new Error("Unsupported site-policy version.");
  }
  if (await sha256Hex(rankingText) !== policy.popularity_snapshot.shortlist_sha256) {
    throw new Error("The frozen popularity shortlist failed its integrity check.");
  }
  const rows = rankingText.trim().split(/\r?\n/).map(line => line.split(","));
  if (rows.length !== policy.popularity_snapshot.shortlist_size) throw new Error("Popularity shortlist size mismatch.");
  const ranks = {};
  rows.forEach(([rawRank, domain], index) => {
    const rank = Number(rawRank);
    if (rank !== index + 1 || !/^[a-z0-9.-]+$/.test(domain)) throw new Error("Invalid popularity shortlist row.");
    ranks[domain] = rank;
  });
  const ids = new Set(); const hosts = new Set();
  for (const rule of policy.reviewed_rules) {
    if (!rule.id || ids.has(rule.id) || !Array.isArray(rule.hosts) || !rule.hosts.length) throw new Error("Invalid or duplicate site-policy rule.");
    ids.add(rule.id);
    if (ranks[rule.registrable_domain] !== rule.tranco_rank) throw new Error(`Rank mismatch for ${rule.id}.`);
    new RegExp(rule.allowed_path_pattern);
    if (rule.excluded_path_pattern) new RegExp(rule.excluded_path_pattern, "i");
    for (const host of rule.hosts) {
      if (hosts.has(host) || (host !== rule.registrable_domain && !host.endsWith(`.${rule.registrable_domain}`))) {
        throw new Error(`Invalid or duplicate reviewed hostname: ${host}.`);
      }
      hosts.add(host);
    }
  }
  new RegExp(policy.global_excluded_path_pattern, "i");
  const extract = PhishingPreprocessing.createPslExtractor(psl.rules);
  const registrableDomainForHost = host => {
    const value = extract(host);
    return [value.domain, value.suffix].filter(Boolean).join(".");
  };
  return { ...policy, popular_domain_ranks: ranks, registrableDomainForHost };
}

function sitePolicy() {
  if (!sitePolicyPromise) sitePolicyPromise = loadSitePolicy().catch(error => {
    console.error("Site policy unavailable; using normal model decisions", error);
    return null;
  });
  return sitePolicyPromise;
}

function localParticipantId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `pilot-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function localReportId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `report-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function localDedupSalt() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function fingerprint(identity, salt) {
  const bytes = new TextEncoder().encode(`${salt}:${identity}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function clearDedupState() {
  await chrome.storage.local.remove([DEDUP_LEDGER_KEY, DEDUP_SALT_KEY, "pilotIdentitiesV5", "pilotDedupSaltV5", "pilotIdentitiesV4", "pilotDedupSaltV4", "pilotIdentitiesV3", "pilotDedupSaltV3", "pilotDedupFingerprintsV2", "pilotDedupSaltV2"]);
}

function queuePilot(task) {
  pilotMutationQueue = pilotMutationQueue.catch(() => {}).then(task);
  return pilotMutationQueue;
}

function queueSettings(task) {
  settingsQueue = settingsQueue.catch(() => {}).then(task);
  return settingsQueue;
}

function validUrl(value) {
  try { return ["http:", "https:"].includes(new URL(value).protocol); } catch (_) { return false; }
}

function isTrustedUI(sender) {
  if (sender.id !== chrome.runtime.id) return false;
  const base = chrome.runtime.getURL("");
  const page = String(sender.url ?? "").split(/[?#]/)[0];
  return ["popup/popup.html", "pilot/pilot.html", "warning/warning.html", "demo/demo.html"]
    .some((relative) => page === base + relative);
}

async function migrateIfNeeded(study) {
  const migration = PhishingPilotStore.migrateStudy(study, new Date().toISOString(), pilotVersion);
  if (migration.migrated) await clearDedupState();
  return migration;
}

async function setPilotParticipationNow(enabled) {
  const stored = await chrome.storage.local.get(["pilotParticipation", "pilotStudy"]);
  let study = stored.pilotStudy;
  if (study) study = (await migrateIfNeeded(study)).study;
  if (enabled && !study) {
    study = PhishingPilotStore.createStudy(localParticipantId(), new Date().toISOString(), pilotVersion);
    await clearDedupState();
  }
  await chrome.storage.local.set({ pilotParticipation: Boolean(enabled), ...(study ? { pilotStudy: study } : {}) });
  return { enabled: Boolean(enabled), study, summary: PhishingPilotStore.summary(study) };
}

async function pilotStatusNow() {
  const stored = await chrome.storage.local.get(["pilotParticipation", "pilotStudy"]);
  let study = stored.pilotStudy ?? null;
  if (study) {
    const migration = await migrateIfNeeded(study);
    study = migration.study;
    if (migration.migrated) await chrome.storage.local.set({ pilotStudy: study });
  }
  return {
    enabled: stored.pilotParticipation === true,
    hasStudy: Boolean(study),
    study,
    summary: PhishingPilotStore.summary(study),
  };
}

function pilotStatus() { return queuePilot(pilotStatusNow); }
function setPilotParticipation(enabled) {
  measurementGeneration += 1;
  return queuePilot(() => setPilotParticipationNow(enabled));
}

function mutatePilot(mutator, generation = measurementGeneration) {
  return queuePilot(async () => {
    if (generation !== measurementGeneration) return null;
    const stored = await chrome.storage.local.get(["pilotParticipation", "pilotStudy"]);
    if (stored.pilotParticipation !== true) return null;
    let study = stored.pilotStudy
      ?? PhishingPilotStore.createStudy(localParticipantId(), new Date().toISOString(), pilotVersion);
    study = (await migrateIfNeeded(study)).study;
    const value = await mutator(study);
    study.lastUpdatedAt = new Date().toISOString();
    await chrome.storage.local.set({ pilotStudy: study });
    return { study, value, summary: PhishingPilotStore.summary(study) };
  });
}

function recordPilotInference(result, source, generation = measurementGeneration) {
  return mutatePilot((study) => PhishingPilotStore.recordInference(
    study, result, source, new Date().toISOString(),
  ), generation);
}

function recordPilotAutomaticAssessment(result, eventSource, generation, current) {
  return queuePilot(async () => {
    if (generation !== measurementGeneration || !current()) return null;
    const stored = await chrome.storage.local.get([
      "pilotParticipation", "pilotStudy", DEDUP_LEDGER_KEY, DEDUP_SALT_KEY,
    ]);
    if (stored.pilotParticipation !== true) return null;

    let study = stored.pilotStudy
      ?? PhishingPilotStore.createStudy(localParticipantId(), new Date().toISOString(), pilotVersion);
    const migration = PhishingPilotStore.migrateStudy(study, new Date().toISOString(), pilotVersion);
    study = migration.study;
    let ledger = migration.migrated ? {} : (stored[DEDUP_LEDGER_KEY] ?? {});
    let salt = migration.migrated || typeof stored[DEDUP_SALT_KEY] !== "string"
      ? localDedupSalt() : stored[DEDUP_SALT_KEY];

    const identity = PhishingPilotStore.canonicalAssessmentIdentity(result);
    const digest = await fingerprint(identity, salt);
    if (!current() || generation !== measurementGeneration) return null;
    let entry = ledger[digest];
    if (!entry && Object.keys(ledger).length >= MAX_DEDUP_FINGERPRINTS) {
      study.counters.untrackedAssessments += 1;
      await chrome.storage.local.set({ pilotStudy: study });
      return null;
    }
    if (!entry) entry = ledger[digest] = { id: localReportId(), automatic: false };
    const isUnique = !entry.automatic;
    result.assessmentId = entry.id;
    if (isUnique) {
      entry.automatic = true;
      // A manual report may predate the first automatic assessment.
      for (const report of study.feedbackReports) {
        if (report.identityId === entry.id) report.assessmentId = entry.id;
      }
    }
    PhishingPilotStore.recordAutomaticAssessment(
      study, result, isUnique, eventSource, new Date().toISOString(),
    );
    await chrome.storage.local.set({
      pilotStudy: study,
      [DEDUP_LEDGER_KEY]: ledger,
      [DEDUP_SALT_KEY]: salt,
    });
    return { study, isUnique, summary: PhishingPilotStore.summary(study) };
  });
}

async function recordFeedback(message, sender) {
  const generation = measurementGeneration;
  const result = await scoreUrl(message.url);
  await recordPilotInference(result, "manual", generation);
  const [hosts, mode, catalog] = await Promise.all([trustedHosts(), operationMode(), sitePolicy()]);
  const decision = PhishingBrowserPolicy.decide(result, hosts, { mode, sitePolicy: catalog });
  return mutatePilot(async (study) => {
    const stored = await chrome.storage.local.get([DEDUP_LEDGER_KEY, DEDUP_SALT_KEY]);
    const ledger = stored[DEDUP_LEDGER_KEY] ?? {};
    const salt = stored[DEDUP_SALT_KEY] ?? localDedupSalt();
    const digest = await fingerprint(PhishingPilotStore.canonicalAssessmentIdentity(result), salt);
    if (!ledger[digest] && Object.keys(ledger).length >= MAX_DEDUP_FINGERPRINTS) {
      throw new Error("Local URL capacity reached. Export your results and start a new evaluation.");
    }
    const entry = ledger[digest] ?? { id: localReportId(), automatic: false };
    ledger[digest] = entry;
    const report = PhishingPilotStore.addFeedback(study, {
      url: message.url, claim: message.claim, score: result.score,
      modelTier: result.tier, effectiveTier: decision.effectiveTier, mode,
      decisionReason: decision.reason ?? "model_policy", queryView: result.queryView,
      explanationSignals: result.explanation?.signals?.map((signal) => signal.code) ?? [],
      identityId: entry.id, assessmentId: entry.automatic ? entry.id : null,
      source: message.source === "allowed_website" ? "allowed_website"
        : sender.url.includes("/warning/") ? "warning_page" : "popup",
    }, new Date().toISOString(), localReportId());
    await chrome.storage.local.set({ [DEDUP_LEDGER_KEY]: ledger, [DEDUP_SALT_KEY]: salt });
    return report;
  }, generation);
}

function recordPilotError() {
  return mutatePilot((study) => PhishingPilotStore.recordError(study, new Date().toISOString()));
}

function recordPilotAction(action) {
  return mutatePilot((study) => PhishingPilotStore.recordAction(study, action, new Date().toISOString()));
}

async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL("offscreen/offscreen.html");
  if (chrome.runtime.getContexts) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"], documentUrls: [offscreenUrl],
    });
    if (contexts.length > 0) return;
  }
  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen.createDocument({
      url: "offscreen/offscreen.html",
      reasons: ["WORKERS"],
      justification: "Run the frozen URL classifier locally without loading WebAssembly in the extension service worker.",
    }).finally(() => { creatingOffscreen = null; });
  }
  await creatingOffscreen;
}

async function scoreUrl(url) {
  await ensureOffscreenDocument();
  const catalog = await sitePolicy();
  const allowUnavailable = Boolean(PhishingBrowserPolicy.servicePage(url, catalog));
  let timer;
  const response = await Promise.race([
    chrome.runtime.sendMessage({ target: "offscreen", type: "RUN_INFERENCE", url, allowUnavailable }),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Local scan timed out. Navigation remains available.")), 30000); }),
  ]).finally(() => clearTimeout(timer));
  if (!response?.ok) throw new Error(response?.error ?? "The local inference document did not respond.");
  return response.result;
}

async function sessionGet(key) {
  const value = await chrome.storage.session.get(key);
  return value[key];
}

async function sessionSet(value) { await chrome.storage.session.set(value); }

async function localGet(key) {
  const value = await chrome.storage.local.get(key);
  return value[key];
}

function pageResultKey(tabId) { return `pageResultV056:${tabId}`; }

async function getPageResult(tabId) {
  const inMemory = pageResults.get(tabId);
  if (inMemory) return inMemory;
  const stored = await sessionGet(pageResultKey(tabId));
  if (stored) pageResults.set(tabId, stored);
  return stored ?? null;
}

async function savePageResult(tabId, url, result) {
  const record = { url, result, timestamp: Date.now() };
  pageResults.set(tabId, record);
  await sessionSet({
    [pageResultKey(tabId)]: record,
    lastResult: { ...result, tabId, timestamp: record.timestamp },
  });
  return record;
}

async function trustedHosts() { return (await localGet("trustedHosts")) ?? []; }

async function operationMode() {
  return (await localGet("operationMode")) === "passive" ? "passive" : "normal";
}

async function setOperationMode(mode) {
  const normalized = mode === "passive" ? "passive" : "normal";
  await chrome.storage.local.set({ operationMode: normalized });
  return normalized;
}

async function setHostTrust(rawUrl, trusted) {
  const host = PhishingBrowserPolicy.hostname(rawUrl);
  if (!host) throw new Error("This URL does not contain a valid hostname.");
  const hosts = new Set(await trustedHosts());
  if (trusted) hosts.add(host); else hosts.delete(host);
  const ordered = [...hosts].sort();
  await chrome.storage.local.set({ trustedHosts: ordered });
  return { host, trusted, trustedHosts: ordered };
}

async function consumeAllowOnce(url, tabId) {
  return queueSettings(async () => {
  const allowed = (await sessionGet("allowOnce")) ?? [];
  const index = allowed.findIndex((item) => item.url === url && item.tabId === tabId);
  if (index < 0) return false;
  allowed.splice(index, 1);
  await sessionSet({ allowOnce: allowed });
  return true;
  });
}

async function addAllowOnce(url, tabId) {
  return queueSettings(async () => {
  const allowed = (await sessionGet("allowOnce")) ?? [];
  if (!allowed.some((item) => item.url === url && item.tabId === tabId)) allowed.push({url, tabId});
  await sessionSet({ allowOnce: allowed.slice(-25) });
  });
}

async function updateBadge(tabId, result) {
  const display = {
    trusted: ["✓", "#15803d"],
    ip_local: ["IP", "#2563eb"], ip_review: ["IP", "#d97706"],
    allow: ["", "#6b7280"], secondary_review: ["?", "#2563eb"],
    caution: ["!", "#d97706"], warning: ["!!", "#dc2626"], block: ["×", "#7f1d1d"],
  }[result.decision?.effectiveTier ?? result.tier] ?? ["", "#6b7280"];
  await chrome.action.setBadgeText({ tabId, text: display[0] });
  if (display[0]) await chrome.action.setBadgeBackgroundColor({ tabId, color: display[1] });
}

async function inspectNavigation(details, eventSource = "navigation") {
  if (details.frameId !== 0) return null;
  const key = `${details.tabId}:${details.url}`;
  if (activeScans.has(key) && navigations.get(details.tabId)?.url === details.url) return activeScans.get(key);
  const token = { url: details.url };
  navigations.set(details.tabId, token);
  void chrome.tabs.sendMessage(details.tabId, { type: "PHISHGUARD_NAVIGATION" }).catch(() => {});
  if (!validUrl(details.url)) return null;
  const current = () => navigations.get(details.tabId) === token;
  const generation = measurementGeneration;
  const task = (async () => {
    try {
      const cached = await getPageResult(details.tabId);
      let result;
      if (cached?.url === details.url && cached.result) {
        result = { ...cached.result };
      } else {
        result = await scoreUrl(details.url);
        await recordPilotInference(result, "automatic", generation).catch((error) => {
          console.error("Local activity recording failure", error);
        });
      }
      if (!current()) return null;
      const prior = presentations.get(details.tabId) ?? Promise.resolve();
      const present = prior.catch(() => {}).then(async () => {
        if (!current()) return null;
        const tab = await chrome.tabs.get(details.tabId);
        if (!current() || (tab.pendingUrl || tab.url) !== details.url) return null;
        // Read settings after inference, so a trust/mode change affects this result.
        await settingsQueue.catch(() => {});
        const [hosts, mode, continuedOnce, catalog] = await Promise.all([
          trustedHosts(), operationMode(), consumeAllowOnce(details.url, details.tabId), sitePolicy(),
        ]);
        if (!current()) return null;
        result.decision = PhishingBrowserPolicy.decide(result, hosts, { mode, continuedOnce, sitePolicy: catalog });
        await recordPilotAutomaticAssessment(result, eventSource, generation, current);
        if (!current()) return null;
        await savePageResult(details.tabId, details.url, result);
        if (!current()) return null;
        await updateBadge(details.tabId, result);
        if (!current()) return null;
        if (result.decision.interrupt) {
          const latest = await chrome.tabs.get(details.tabId);
          if (!current() || (latest.pendingUrl || latest.url) !== details.url) return null;
          const fragment = new URLSearchParams({
            url: details.url, score: String(result.score), tier: result.tier,
            signals: JSON.stringify(result.explanation?.signals ?? []),
          });
          await chrome.tabs.update(details.tabId, { url: `${chrome.runtime.getURL("warning/warning.html")}#${fragment}` });
          await mutatePilot((study) => {
            study.counters.navigationInterrupts += 1;
            study.interventionRecords.push({ assessmentId: result.assessmentId ?? null,
              mode, tier: result.decision.effectiveTier, at: new Date().toISOString() });
            study.interventionRecords = study.interventionRecords.slice(-250);
          }, generation);
        } else {
          await chrome.tabs.sendMessage(details.tabId, {
            type: "PHISHGUARD_PAGE_RESULT", url: details.url, result,
          }).catch(() => {});
        }
        return result;
      });
      presentations.set(details.tabId, present);
      try { return await present; }
      finally { if (presentations.get(details.tabId) === present) presentations.delete(details.tabId); }
    } catch (error) {
      console.error("Phishing URL Guard inference failure", error);
      if (current()) {
        await mutatePilot((study) => PhishingPilotStore.recordError(study, new Date().toISOString()), generation).catch(() => {});
        await sessionSet({ lastError: { message: String(error?.message ?? error), timestamp: Date.now() } });
      }
      return null;
    } finally {
      if (activeScans.get(key) === task) activeScans.delete(key);
    }
  })();
  activeScans.set(key, task);
  return task;
}

chrome.webNavigation.onBeforeNavigate.addListener((details) => { void inspectNavigation(details, "navigation"); });
chrome.webNavigation.onHistoryStateUpdated.addListener((details) => { void inspectNavigation(details, "history_state"); });
chrome.webNavigation.onReferenceFragmentUpdated.addListener((details) => { void inspectNavigation(details, "fragment"); });
chrome.tabs.onRemoved.addListener((tabId) => {
  navigations.delete(tabId);
  pageResults.delete(tabId);
  void chrome.storage.session.remove(pageResultKey(tabId));
});

async function refreshVisiblePolicies() {
  const tabs = await chrome.tabs.query({});
  for (const tab of tabs) {
    const url = tab.pendingUrl || tab.url;
    if (!validUrl(url)) continue;
    // Invalidate existing work before refreshing cached decisions.
    navigations.delete(tab.id);
    void inspectNavigation({ tabId: tab.id, frameId: 0, url }, "policy_change");
  }
}
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    void chrome.tabs.create({ url: chrome.runtime.getURL("pilot/pilot.html") });
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target === "offscreen") return false;
  const contentAction = ["GET_PAGE_ADVISORY", "RECORD_PILOT_ACTION", "ADVISORY_DISPLAYED", "TRUST_CURRENT_ADVISORY"].includes(message?.type);
  const contentSender = _sender.id === chrome.runtime.id && Number.isInteger(_sender.tab?.id)
    && _sender.frameId === 0 && validUrl(_sender.url);
  if (!isTrustedUI(_sender) && !(contentAction && contentSender)) {
    sendResponse({ ok: false, error: "This action is only available from the extension controls." });
    return false;
  }
  if (["SCORE_URL", "ALLOW_ONCE", "TRUST_HOST", "UNTRUST_HOST", "RECORD_PILOT_FEEDBACK"].includes(message?.type)
      && !validUrl(message.url)) {
    sendResponse({ ok: false, error: "An HTTP(S) URL is required." });
    return false;
  }
  if (message?.type === "ALLOW_ONCE") {
    if (!Number.isInteger(_sender.tab?.id)) {
      sendResponse({ ok: false, error: "Continue once requires the warning tab." });
      return false;
    }
    addAllowOnce(message.url, _sender.tab.id).then(() => {
      void recordPilotAction("warning_continue").catch(() => {});
      sendResponse({ ok: true });
    }, (error) => sendResponse({ ok: false, error: String(error) }));
    return true;
  }
  if (message?.type === "SCORE_URL") {
    const generation = measurementGeneration;
    Promise.all([scoreUrl(message.url), trustedHosts(), operationMode(), sitePolicy()]).then(async ([result, hosts, mode, catalog]) => {
      result.decision = PhishingBrowserPolicy.decide(result, hosts, { mode, sitePolicy: catalog });
      await recordPilotInference(result, "manual", generation).catch((error) => {
        console.error("Local manual-scan recording failure", error);
      });
      sendResponse({ ok: true, result });
    }, async (error) => {
      await mutatePilot((study) => PhishingPilotStore.recordError(study, new Date().toISOString()), generation).catch(() => {});
      sendResponse({ ok: false, error: String(error?.message ?? error) });
    });
    return true;
  }
  if (message?.type === "GET_PAGE_ADVISORY") {
    const tabId = _sender.tab?.id;
    const senderUrl = _sender.url ?? message.url;
    if (!Number.isInteger(tabId) || _sender.frameId !== 0 || !/^https?:\/\//i.test(senderUrl ?? "")) {
      sendResponse({ ok: false, error: "No eligible top-level page was found." });
      return false;
    }
    const key = `${tabId}:${senderUrl}`;
    (async () => {
      if (activeScans.has(key)) await activeScans.get(key);
      let cached = await getPageResult(tabId);
      const tab = await chrome.tabs.get(tabId);
      if ((tab.pendingUrl || tab.url) !== senderUrl) return null;
      if (!cached || cached.url !== senderUrl) {
        await inspectNavigation({ tabId, frameId: 0, url: senderUrl }, "advisory_recovery");
        cached = await getPageResult(tabId);
      }
      return cached?.url === senderUrl ? cached.result : null;
    })().then((result) => sendResponse({ ok: true, result }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "TRUST_CURRENT_ADVISORY") {
    const tabId = _sender.tab?.id;
    const senderUrl = _sender.url;
    if (!Number.isInteger(tabId) || _sender.frameId !== 0 || !validUrl(senderUrl)) {
      sendResponse({ ok: false, error: "No eligible current website was found." });
      return false;
    }
    (async () => {
      const cached = await getPageResult(tabId);
      if (cached?.url !== senderUrl || !PhishingBrowserPolicy.shouldShowAdvisory(cached.result?.decision)) {
        throw new Error("The current warning is no longer active.");
      }
      await recordFeedback({
        url: senderUrl, claim: "false_alarm", source: "allowed_website",
      }, _sender).catch(() => null);
      const result = await queueSettings(() => setHostTrust(senderUrl, true));
      await recordPilotAction("trusted_site").catch(() => {});
      await refreshVisiblePolicies();
      return result;
    })().then((result) => sendResponse({ ok: true, ...result }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "TRUST_HOST" || message?.type === "UNTRUST_HOST") {
    queueSettings(() => setHostTrust(message.url, message.type === "TRUST_HOST")).then(async (result) => {
      await recordPilotAction("trusted_site").catch(() => {});
      await refreshVisiblePolicies();
      sendResponse({ ok: true, ...result });
    }, (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "GET_TRUST_STATUS") {
    trustedHosts().then((hosts) => sendResponse({
      ok: true, host: PhishingBrowserPolicy.hostname(message.url),
      trusted: PhishingBrowserPolicy.isTrusted(message.url, hosts), trustedHosts: hosts,
    }));
    return true;
  }
  if (message?.type === "GET_STATUS") {
    Promise.all([sessionGet("lastResult"), sessionGet("lastError"), operationMode()]).then(
      ([lastResult, lastError, mode]) => sendResponse({ ok: true, lastResult, lastError, mode }),
    );
    return true;
  }
  if (message?.type === "GET_MODE") {
    operationMode().then((mode) => sendResponse({ ok: true, mode }));
    return true;
  }
  if (message?.type === "SET_MODE") {
    queueSettings(() => setOperationMode(message.mode)).then(async (mode) => {
      await refreshVisiblePolicies();
      sendResponse({ ok: true, mode });
    },
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "GET_PILOT_STATUS" || message?.type === "GET_PILOT_EXPORT") {
    pilotStatus().then((status) => sendResponse({ ok: true, ...status }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "SET_PILOT_PARTICIPATION") {
    setPilotParticipation(message.enabled).then((status) => sendResponse({ ok: true, ...status }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "RESET_PILOT_STUDY") {
    measurementGeneration += 1;
    queuePilot(async () => {
      const current = await pilotStatusNow();
      const participantId = current.study?.participantId ?? localParticipantId();
      const study = PhishingPilotStore.createStudy(participantId, new Date().toISOString(), pilotVersion);
      await clearDedupState();
      await chrome.storage.local.set({ pilotStudy: study });
      return { enabled: current.enabled, study, summary: PhishingPilotStore.summary(study) };
    }).then((status) => sendResponse({ ok: true, ...status }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "RECORD_PILOT_FEEDBACK") {
    recordFeedback(message, _sender).then((result) => {
      if (!result) sendResponse({ ok: false, error: "Local detection history is turned off." });
      else sendResponse({ ok: true, report: result.value, summary: result.summary });
    }, (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "RECORD_PILOT_ACTION") {
    if (message.action !== "advisory_dismiss") {
      sendResponse({ ok: false, error: "Unsupported page action." });
      return false;
    }
    recordPilotAction(message.action).then(() => sendResponse({ ok: true }),
      (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }
  if (message?.type === "ADVISORY_DISPLAYED") {
    (async () => {
      const cached = await getPageResult(_sender.tab.id);
      if (cached?.url !== _sender.url || !PhishingBrowserPolicy.shouldShowAdvisory(cached.result?.decision)) return;
      await mutatePilot((study) => {
        const record = study.assessmentRecords.find((entry) => entry.id === cached.result.assessmentId);
        if (record && !record.advisoryDisplayed) {
          record.advisoryDisplayed = true;
          study.counters.advisoryDisplays += 1;
        }
      });
    })().then(() => sendResponse({ ok: true }), (error) => sendResponse({ ok: false, error: String(error) }));
    return true;
  }
  return false;
});
