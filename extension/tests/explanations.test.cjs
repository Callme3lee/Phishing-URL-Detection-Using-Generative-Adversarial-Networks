const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createPreprocessor } = require('../preprocessing.js');
const i18n = require('../localization.js');

const root = path.resolve(__dirname, '..');
const read = relative => JSON.parse(fs.readFileSync(path.join(root, relative), 'utf8'));
const preprocessor = createPreprocessor({
  vocabulary: read('models/vocabulary.json'), scaler: read('models/scaler.json'),
  pslRules: read('models/psl_rules.json'), htmlEntities: read('models/html_entities.json'),
});

assert.deepEqual(preprocessor.prepare('https://example.com/').signals, []);
const ipCodes = preprocessor.prepare('http://192.0.2.10/update/password').signals.map(signal => signal.code);
assert.ok(ipCodes.includes('ip_hostname'));
assert.ok(ipCodes.includes('suspicious_terms'));
const complex = preprocessor.prepare(
  `https://secure-login-verify.example.com/account?token=${'x'.repeat(150)}`,
).signals;
assert.ok(complex.some(signal => signal.code === 'complex_query'));
assert.ok(complex.some(signal => signal.code === 'suspicious_terms'));
assert.ok(complex.length <= 4);
assert.ok(complex.every(signal => !JSON.stringify(signal).includes('token=')), 'signals must not retain the URL');
assert.match(i18n.signalText({ code: 'long_url', values: { count: 180 } }, 'en'), /180/);
assert.match(i18n.signalText({ code: 'long_url', values: { count: 180 } }, 'ar'), /180/);
assert.match(i18n.signalText({ code: 'combined_pattern' }, 'en'), /combined URL pattern/);
console.log('PASS: bounded, privacy-safe lexical diagnostics and Arabic signal labels are reproducible.');
