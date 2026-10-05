const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const P = require('../browser_policy.js');
const preprocessing = require('../preprocessing.js');
const catalog = JSON.parse(fs.readFileSync(require.resolve('../policy/site_policy.json'), 'utf8'));
catalog.popular_domain_ranks = Object.fromEntries(fs.readFileSync(require.resolve('../policy/tranco_XN6XN_top250.csv'), 'utf8').trim().split(/\r?\n/).map(line => {
  const [rank, domain] = line.split(','); return [domain, Number(rank)];
}));
const psl = JSON.parse(fs.readFileSync(require.resolve('../models/psl_rules.json'), 'utf8'));
const extract = preprocessing.createPslExtractor(psl.rules);
catalog.registrableDomainForHost = host => { const x = extract(host); return [x.domain, x.suffix].filter(Boolean).join('.'); };
const decide = (result, trusted = [], options = {}) => P.decide(result, trusted, {...options, sitePolicy: catalog});
const good = [
  'https://www.google.com/search?q=login+password&state=abc',
  'https://google.com/',
  'https://accounts.google.com/v3/signin/identifier?continue=https%3A%2F%2Fexample.com',
  'https://accounts.google.com/o/oauth2/v2/auth?client_id=sample',
  'https://chatgpt.com/', 'https://chatgpt.com/c/1234-5678?model=example',
  'https://chat.openai.com/', 'https://auth.openai.com/log-in',
  'https://auth0.openai.com/authorize?state=sample',
  'https://mail.google.com/mail/u/0/#inbox',
  'https://docs.google.com/', 'https://drive.google.com/', 'https://sites.google.com/',
  'https://sheets.google.com/', 'https://slides.google.com/', 'https://forms.google.com/',
  'https://login.microsoftonline.com/common/oauth2/v2.0/authorize?state=sample',
];
good.push(...catalog.reviewed_rules.flatMap(rule => rule.hosts.map(host => `https://${host}/`)));
const excluded = [
  'http://chatgpt.com/', 'https://chatgpt.com:8443/',
  'https://user:pass@chatgpt.com/', 'https://chatgpt.com.attacker.invalid/',
  'https://accounts.google.com@attacker.invalid/', 'https://evil.chatgpt.com/',
  'https://chatgpt.com./', 'https://www.google.com/url?q=https%3A%2F%2Fattacker.invalid',
  'https://www.google.com/%75rl?q=https%3A%2F%2Fattacker.invalid',
  'https://chatgpt.com/share/123',
  'https://docs.google.com.attacker.invalid/document/d/example',
  'https://drive.google.com.attacker.invalid/file/d/example',
  'https://docs-google.com/document/d/example',
  'http://docs.google.com/document/d/example',
  'https://chatgpt.com/g/some-custom-gpt', 'https://chatgpt.com/c/../share/123',
  'https://chatgpt.com/c/%2fshare/123', 'https://chatgpt.com/unknown-path',
  'https://chatgpt.com\\@attacker.invalid/',
  'https://chаtgpt.com/', // Cyrillic a
];
for (const url of good) {
  assert.ok(P.servicePage(url, catalog), url);
  for (const tier of ['allow', 'secondary_review', 'caution', 'warning', 'block', 'unknown']) {
    for (const mode of ['normal', 'passive']) {
      const decision = decide({rawUrl: url, tier}, [], {mode});
      assert.equal(decision.reason, 'recognized_service', url);
      assert.equal(decision.modelTier, tier);
      assert.equal(decision.effectiveTier, 'allow');
      assert.equal(decision.interrupt, false);
      assert.equal(P.shouldShowAdvisory(decision), false, `No banner, including review: ${url}`);
    }
  }
}
const userContent = [
  'https://docs.google.com/document/d/example/edit?tab=t.0',
  'https://docs.google.com/forms/d/e/example/viewform?usp=sharing',
  'https://docs.google.com/presentation/d/e/example/pub',
  'https://docs.google.com/drawings/d/example/edit',
  'https://drive.google.com/drive/folders/example?usp=drive_link',
  'https://drive.google.com/file/d/example/view?usp=sharing',
  'https://sites.google.com/view/login',
];
const expectedSharedTier = {
  allow: 'allow', secondary_review: 'secondary_review', caution: 'secondary_review',
  warning: 'warning', block: 'block', unknown: 'secondary_review',
};
for (const url of userContent) {
  const service = P.servicePage(url, catalog);
  assert.ok(service, url);
  assert.equal(P.userControlledServiceContent(url, service), true, url);
  for (const [modelTier, effectiveTier] of Object.entries(expectedSharedTier)) {
    const decision = decide({rawUrl: url, tier: modelTier});
    const collaborativeDocument = P.googleCollaborativeDocumentRoute(url, service);
    assert.equal(decision.reason, collaborativeDocument ? 'reviewed_google_document' : 'reviewed_user_content', url);
    assert.equal(decision.modelTier, modelTier, url);
    const documentTier = {allow: 'allow', secondary_review: 'secondary_review', caution: 'secondary_review',
      warning: 'caution', block: 'warning', unknown: 'secondary_review'}[modelTier];
    assert.equal(decision.effectiveTier, collaborativeDocument ? documentTier : effectiveTier, url);
    const shouldInterrupt = !collaborativeDocument && ['warning', 'block'].includes(modelTier);
    assert.equal(decision.interrupt, shouldInterrupt,
      `Shared content should block only high-risk tiers in Normal mode: ${url}`);
    assert.equal(P.shouldShowAdvisory(decision), effectiveTier !== 'allow' && !shouldInterrupt, url);
    const passive = decide({rawUrl: url, tier: modelTier}, [], {mode: 'passive'});
    assert.equal(passive.interrupt, false, `Passive mode must not interrupt: ${url}`);
    const continued = decide({rawUrl: url, tier: modelTier}, [], {continuedOnce: true});
    assert.equal(continued.interrupt, false, `Continue once must not loop: ${url}`);
  }
}
for (const url of excluded) {
  assert.equal(P.servicePage(url, catalog), null, url);
  assert.equal(decide({rawUrl: url, tier: 'block'}).interrupt, true, url);
}
const view = P.queryView('https://example.com/path?a=secret&a=other&flag&empty=#fragment');
assert.equal(view.url, 'https://example.com/path?a=&a=&flag&empty=#fragment');
assert.equal(view.parameterCount, 4);
assert.equal(view.removedValueCharacters, 11);
assert.equal(P.queryView('https://example.com/path').applied, false);
assert.equal(decide({rawUrl: 'https://attacker.invalid/login', tier: 'block',
  queryView: {score: 0, tier: 'allow'}}).interrupt, true, 'normalizing values must not suppress an unknown host');
const correctedGoogleDocument = decide({
  rawUrl: 'https://docs.google.com/document/d/example/edit?tab=t.0&usp=sharing',
  tier: 'block', queryView: {applied: true, score: 0.1, tier: 'allow'},
});
assert.equal(correctedGoogleDocument.reason, 'reviewed_google_document_query_noise');
assert.equal(correctedGoogleDocument.effectiveTier, 'secondary_review');
assert.equal(correctedGoogleDocument.interrupt, false);
for (const unsafe of [
  'https://docs.google.com/document/d/example/edit?redirect=https://attacker.invalid',
  'https://docs.google.com/forms/d/e/example/viewform?usp=sharing',
  'https://docs.google.com/presentation/d/e/example/pub?usp=sharing',
]) {
  const decision = decide({rawUrl: unsafe, tier: 'block',
    queryView: {applied: true, score: 0.1, tier: 'allow'}});
  assert.equal(decision.interrupt, true, `Unsafe/public Google content must retain high-score blocking: ${unsafe}`);
}
assert.equal(P.servicePage(good[0]), null, 'catalog absence must fail closed to normal model policy');
assert.deepEqual(P.popularity('https://news.google.com/path', catalog), {domain: 'google.com', rank: 1, trusted: false});
assert.equal(P.popularity('https://unknown.invalid/', catalog), null);

// Use the production wrapper to verify secondary-score failure cannot discard
// a valid full score, and unavailable official-page scores remain explicit.
const ctx = vm.createContext({PhishingBrowserPolicy: P, performance, URL, console});
vm.runInContext(fs.readFileSync(require.resolve('../inference.js'), 'utf8'), ctx);
(async () => {
  const engine = new ctx.PhishingInferenceEngine();
  engine.scoreSingle = async url => {
    if (url.endsWith('?x=')) throw Error('diagnostic failure');
    return {rawUrl: url, score: 1, tier: 'block'};
  };
  const full = await engine.scoreUrl('https://attacker.invalid/login?x=secret');
  assert.equal(full.score, 1);
  assert.equal(full.modelViewCount, 1);
  assert.ok(full.queryView.error);
  engine.scoreSingle = async () => { throw Error('raw_url_too_long'); };
  const oversized = await engine.scoreUrl('https://accounts.google.com/login?state=' + 'x'.repeat(5000), {allowUnavailable: true});
  assert.equal(oversized.score, null);
  assert.equal(oversized.tier, 'unknown');
  assert.equal(P.shouldShowAdvisory(decide(oversized)), false);
  await assert.rejects(engine.scoreUrl('https://attacker.invalid/?state=' + 'x'.repeat(5000)));
  console.log(`PASS: ${good.length} fixed service URLs, ${userContent.length} risk-preserving shared-content URLs, ${excluded.length} excluded addresses, normalization and failure handling.`);
})().catch(error => {console.error(error); process.exitCode = 1;});
