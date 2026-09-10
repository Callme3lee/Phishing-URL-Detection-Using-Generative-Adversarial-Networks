const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');
const popup = read('popup/popup.html');
const activity = read('pilot/pilot.html');
const warning = read('warning/warning.html');
const demo = read('demo/demo.html');
const advisory = read('content/advisory.html');
const manifest = JSON.parse(read('manifest.json'));
const modelPolicy = JSON.parse(read('models/policy.json'));

assert.equal(manifest.version, '0.6.0');
assert.equal(manifest.name, 'Phishing URL Guard');
assert.equal(modelPolicy.primary_binary_rule.threshold, 0.9707667827606201);
assert.equal(modelPolicy.primary_binary_rule.threshold, modelPolicy.tiers.find(tier => tier.name === 'caution').lower_inclusive);
assert.equal(modelPolicy.tiers.find(tier => tier.name === 'block').lower_inclusive, 0.9998490214347839);
for (const phrase of ['Was this result correct?', '>Activity<', 'RISK SCORE', 'About this result', 'id="trust-hostname"']) {
  assert.ok(popup.includes(phrase), phrase);
}
for (const phrase of ['Your detection activity', 'Turn on local detection history?', 'Unique URLs checked', 'Export your activity']) {
  assert.ok(activity.includes(phrase), phrase);
}
for (const phrase of ['This URL may be dangerous', 'Continue once (do not allow)', 'Know and trust this website?', 'id="trust-hostname"']) {
  assert.ok(warning.includes(phrase), phrase);
}
assert.ok(!popup.includes('<dialog'), 'popup should not hide allow behind a confirmation dialog');
assert.ok(!warning.includes('<dialog'), 'warning should not show a confirmation dialog');
assert.ok(!warning.includes('<details class="advanced">'), 'permanent allow should be visible without an advanced menu');
assert.ok(advisory.includes('id="allow-hostname"'), 'caution/high-risk notices should show the exact hostname');
assert.ok(advisory.includes('Only this exact website name will be allowed:'), 'the permanent allow scope should be disclosed');
for (const text of [popup, warning]) {
  assert.ok(text.includes('Signals found in this URL'), 'suspicious-result UI should explain its diagnostic signals');
}
assert.ok(!advisory.includes('Signals found in this URL'), 'the in-page notice should keep detailed reasons inside the extension');
assert.ok(advisory.includes('Open the extension to see why this URL received this result.'));
assert.ok(demo.includes('Offline checker'));
for (const text of [popup, activity, warning, demo]) {
  assert.ok(!text.includes('Research prototype'));
  assert.ok(!text.includes('Local pilot'));
}
console.log('PASS: user-facing detection, activity, warning and privacy labels are present.');
