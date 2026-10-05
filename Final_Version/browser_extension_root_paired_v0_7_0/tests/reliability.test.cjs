const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const P = require('../pilot_store.js');
const root = path.resolve(__dirname, '..');
const clone = x => x === undefined ? x : JSON.parse(JSON.stringify(x));
const defer = () => { let resolve; const promise = new Promise(r => resolve = r); return { promise, resolve }; };

function area(backing) {
  return {
    async get(keys) {
      if (Array.isArray(keys)) return Object.fromEntries(keys.map(k => [k, clone(backing[k])]));
      return { [keys]: clone(backing[keys]) };
    },
    async set(values) { Object.assign(backing, clone(values)); },
    async remove(keys) { for (const key of [].concat(keys)) delete backing[key]; },
  };
}

function worker(local = { pilotParticipation: true, pilotStudy: P.createStudy('test', 'now', '0.5.7') }, session = {}) {
  const tabs = new Map(); const updates = []; const messages = []; let listener;
  const event = () => ({ listeners: [], addListener(fn) { this.listeners.push(fn); } });
  const chrome = {
    runtime: { id: 'test', getManifest: () => ({ version: '0.5.7' }),
      getURL: p => 'chrome-extension://test/' + p,
      getContexts: async () => [{}], onInstalled: event(),
      onMessage: { addListener(fn) { listener = fn; } },
      sendMessage: async m => ({ ok: true, result: {
        rawUrl: m.url, canonicalUrl: m.url.replace(/^https?:\/\//, ''),
        tier: m.url.includes('risk') ? 'block' : 'allow', score: m.url.includes('risk') ? 1 : 0,
        inferenceMilliseconds: 1,
      } }),
    },
    storage: { local: area(local), session: area(session) },
    offscreen: { createDocument: async () => {} },
    action: { setBadgeText: async () => {}, setBadgeBackgroundColor: async () => {} },
    tabs: {
      get: async id => { if (!tabs.has(id)) throw Error('Tab closed'); return clone(tabs.get(id)); },
      query: async () => [...tabs.values()].map(clone),
      update: async (id, value) => { updates.push({ id, ...value }); tabs.set(id, { id, ...value }); },
      sendMessage: async (id, value) => { messages.push({ id, ...value }); },
      create: async () => {}, onRemoved: event(),
    },
    webNavigation: { onBeforeNavigate: event(), onHistoryStateUpdated: event(), onReferenceFragmentUpdated: event() },
  };
  const fetch = async input => {
    const relative = String(input).replace('chrome-extension://test/', '');
    const body = fs.readFileSync(path.join(root, relative), 'utf8');
    return {ok: true, async text() { return body; }, async json() { return JSON.parse(body); }};
  };
  const ctx = vm.createContext({ chrome, console, crypto: webcrypto, TextEncoder, URL, URLSearchParams, fetch, setTimeout, clearTimeout });
  ctx.importScripts = (...files) => files.forEach(f => vm.runInContext(fs.readFileSync(path.join(root, f), 'utf8'), ctx));
  vm.runInContext(fs.readFileSync(path.join(root, 'background.js'), 'utf8'), ctx);
  async function nav(url, tabId = 1) {
    tabs.set(tabId, { id: tabId, url });
    return ctx.inspectNavigation({ tabId, frameId: 0, url }, 'navigation');
  }
  async function send(message, sender = { id: 'test', url: 'chrome-extension://test/popup/popup.html' }) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error('Message timeout: ' + message.type)), 3000);
      listener(message, sender, value => { clearTimeout(timer); resolve(value); });
    });
  }
  return { local, session, ctx, chrome, tabs, updates, messages, nav, send };
}

(async () => {
  // Older inference must not navigate, badge or cache over a newer page.
  {
    const w = worker(); const gate = defer(); const started = defer();
    const score = w.chrome.runtime.sendMessage;
    w.chrome.runtime.sendMessage = async m => { if (m.url.includes('slow')) { started.resolve(); await gate.promise; } return score(m); };
    const slow = w.nav('https://example.com/risk-slow'); await started.promise;
    await w.nav('https://example.com/new'); gate.resolve(); await slow;
    assert.equal(w.updates.length, 0);
    assert.equal((await w.ctx.getPageResult(1)).url, 'https://example.com/new');
    assert.equal(w.local.pilotStudy.counters.uniqueAutomaticUrls, 1);
  }
  // Status/export and increments must not lose writes under a forced interleaving.
  {
    const w = worker(); const gate = defer(); const started = defer(); const get = w.chrome.storage.local.get; let first = true;
    w.chrome.storage.local.get = async keys => { const snapshot = await get(keys); if (first) { first = false; started.resolve(); await gate.promise; } return snapshot; };
    const status = w.ctx.pilotStatus(); await started.promise;
    const change = w.ctx.recordPilotError(); gate.resolve(); await Promise.all([status, change]);
    assert.equal(w.local.pilotStudy.counters.inferenceErrors, 1);
    await Promise.all(Array.from({ length: 20 }, () => w.ctx.recordPilotError()));
    assert.equal((await w.ctx.pilotStatus()).study.counters.inferenceErrors, 21);
  }
  // Two query variants retain two linked reports; repeat presses update one and count submissions.
  {
    const w = worker();
    for (const n of [1, 2]) {
      await w.nav('https://example.com/path?private=' + n);
      assert.equal((await w.send({ type: 'RECORD_PILOT_FEEDBACK', claim: 'false_alarm', url: 'https://example.com/path?private=' + n })).ok, true);
    }
    await w.send({ type: 'RECORD_PILOT_FEEDBACK', claim: 'false_alarm', url: 'https://example.com/path?private=2#part' });
    const study = (await w.ctx.pilotStatus()).study;
    assert.equal(study.feedbackReports.length, 2);
    assert.equal(study.counters.feedbackSubmissions, 3);
    assert.ok(study.feedbackReports.every(r => study.assessmentRecords.some(a => a.id === r.assessmentId)));
    assert.ok(!JSON.stringify(study).includes('private='));
    assert.ok(!JSON.stringify(study).includes('pilotDedupSalt'));
    // Persistence over a service-worker restart, scheme and fragment deduplication.
    const restarted = worker(w.local, w.session);
    await restarted.nav('http://example.com/path?private=1#part', 2);
    assert.equal(w.local.pilotStudy.counters.uniqueAutomaticUrls, 2);
  }
  // Passive tier != actual interruption; trusted hosts do not increment block.
  {
    const w = worker(); w.local.operationMode = 'passive';
    await w.nav('https://example.com/risk');
    assert.equal(w.local.pilotStudy.decisionCounts.block, 1);
    assert.equal(w.local.pilotStudy.counters.navigationInterrupts, 0);
    w.local.trustedHosts = ['example.com'];
    await w.nav('https://example.com/risk-new');
    assert.equal(w.local.pilotStudy.decisionCounts.trusted, 1);
    w.local.trustedHosts = []; w.local.operationMode = 'normal';
    await w.nav('https://different.example/risk');
    assert.equal(w.local.pilotStudy.counters.navigationInterrupts, 1);
    assert.equal(w.local.pilotStudy.interventionRecords.length, 1);
  }
  // Trust while inference is pending is respected. Reset cannot absorb an old scan.
  for (const action of ['trust', 'reset', 'pause', 'close']) {
    const w = worker(); const gate = defer(); const started = defer(); const score = w.chrome.runtime.sendMessage;
    w.chrome.runtime.sendMessage = async m => { started.resolve(); await gate.promise; return score(m); };
    const pending = w.nav('https://example.com/risk'); await started.promise;
    if (action === 'trust') await w.send({ type: 'TRUST_HOST', url: 'https://example.com/risk' });
    if (action === 'reset') await w.send({ type: 'RESET_PILOT_STUDY' });
    if (action === 'pause') await w.send({ type: 'SET_PILOT_PARTICIPATION', enabled: false });
    if (action === 'close') { w.tabs.delete(1); w.chrome.tabs.onRemoved.listeners.forEach(fn => fn(1)); }
    gate.resolve(); await pending;
    if (action === 'trust' || action === 'close') assert.equal(w.updates.length, 0);
    if (action === 'reset' || action === 'pause') assert.equal(w.local.pilotStudy.counters.uniqueAutomaticUrls, 0);
  }
  // A visible caution/high-risk notice can allow only its current cached host and
  // records one local false-alarm report when the user opted into feedback.
  {
    const w = worker(); w.local.operationMode = 'passive';
    const url = 'https://notice.example/risk';
    await w.nav(url);
    const sender = { id: 'test', url, tab: { id: 1 }, frameId: 0 };
    const response = await w.send({ type: 'TRUST_CURRENT_ADVISORY' }, sender);
    assert.equal(response.ok, true);
    assert.deepEqual(w.local.trustedHosts, ['notice.example']);
    assert.equal(w.local.pilotStudy.feedbackReports.length, 1);
    assert.equal(w.local.pilotStudy.feedbackReports[0].claim, 'false_alarm');
    assert.equal(w.local.pilotStudy.feedbackReports[0].source, 'allowed_website');
    assert.equal(w.local.pilotStudy.counters.falseAlarmReports, 1);
  }
  // Content scripts cannot read exports, reset data, submit arbitrary feedback or trust hosts.
  {
    const w = worker(); const sender = { id: 'test', url: 'https://example.com/', tab: { id: 1 }, frameId: 0 };
    for (const type of ['GET_PILOT_EXPORT', 'RESET_PILOT_STUDY', 'TRUST_HOST', 'RECORD_PILOT_FEEDBACK']) {
      assert.equal((await w.send({ type, url: 'https://example.com/', claim: 'false_alarm' }, sender)).ok, false);
    }
    assert.equal((await w.send({ type: 'TRUST_HOST', url: 'javascript:alert(1)' })).ok, false);
    await w.ctx.addAllowOnce('https://example.com/risk', 1);
    assert.equal(await w.ctx.consumeAllowOnce('https://example.com/risk', 2), false);
    assert.equal(await w.ctx.consumeAllowOnce('https://example.com/risk', 1), true);
    assert.equal(await w.ctx.consumeAllowOnce('https://example.com/risk', 1), false);
  }
  // Schema-2 upgrade is explicit; previous measurements and reports survive in the archive.
  {
    const study = P.createStudy('retention', 'now', '0.5.7');
    for (let n = 0; n < 251; n++) P.addFeedback(study, {
      url: 'https://example.com/' + n, identityId: String(n), claim: 'false_alarm',
    }, 'now', String(n));
    assert.equal(study.feedbackReports.length, 250);
    assert.equal(study.counters.feedbackRecordsEvicted, 1);
    assert.equal(study.counters.feedbackSubmissions, 251);
    assert.equal(study.counters.falseAlarmReports, 250);
  }
  {
    const w = worker();
    await w.send({ type: 'SET_MODE', mode: 'passive' });
    await w.nav('https://example.com/risk');
    const sender = { id: 'test', url: 'https://example.com/risk', tab: { id: 1 }, frameId: 0 };
    await w.send({ type: 'ADVISORY_DISPLAYED' }, sender);
    await w.send({ type: 'ADVISORY_DISPLAYED' }, sender);
    assert.equal(w.local.pilotStudy.counters.advisoryDisplays, 1);
    assert.equal(w.local.pilotStudy.counters.navigationInterrupts, 0);
  }
  {
    const old = P.createStudy('existing-user', 'old-date', '0.5.1'); old.schemaVersion = 2;
    old.counters.uniqueAutomaticUrls = 202;
    old.feedbackReports.push({ reportedUrl: 'https://example.com/', claim: 'false_alarm' });
    const w = worker({ pilotParticipation: true, pilotStudy: old, pilotDedupFingerprintsV2: ['old'] });
    const updated = await w.ctx.pilotStatus();
    assert.equal(updated.study.schemaVersion, 7);
    assert.equal(updated.study.counters.uniqueAutomaticUrls, 0);
    assert.equal(updated.study.legacyStudy.counters.uniqueAutomaticUrls, 202);
    assert.equal(updated.study.legacyStudy.feedbackReports.length, 1);
    assert.equal(w.local.pilotDedupFingerprintsV2, undefined);
  }
  for (const oldSchema of [3, 4]) {
    const old = P.createStudy('previous-period', 'then', '0.5.' + oldSchema);
    old.schemaVersion = oldSchema;
    old.counters.uniqueAutomaticUrls = 77;
    const w = worker({pilotParticipation: true, pilotStudy: old});
    const current = await w.ctx.pilotStatus();
    assert.equal(current.study.schemaVersion, 7);
    assert.equal(current.study.counters.uniqueAutomaticUrls, 0);
    assert.equal(current.study.legacyStudy.counters.uniqueAutomaticUrls, 77);
  }
  console.log('PASS: navigation races, serialized storage, linked feedback, mode/trust, reset/pause, sender checks, continuation scope and migration.');
})().catch(error => { console.error(error); process.exitCode = 1; });
