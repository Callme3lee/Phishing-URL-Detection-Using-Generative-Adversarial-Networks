const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const i18n = require('../localization.js');

const root = path.resolve(__dirname, '..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');

assert.equal(i18n.STORAGE_KEY, 'uiLanguage');
assert.equal(i18n.translate('Check this page', 'ar'), 'افحص هذه الصفحة');
assert.equal(i18n.translate('Check this page', 'en'), 'Check this page');
assert.equal(
  i18n.translate('{count} unique URLs · {reports} reports', 'ar', { count: 12, reports: 3 }),
  '12 رابط فريد · 3 بلاغ',
);
assert.match(i18n.translate('Very high-risk URL', 'ar'), /خطورة/);
assert.match(i18n.translate('Always allow this exact website', 'ar'), /السماح دائماً/);
assert.match(i18n.translate('Always allow', 'ar'), /السماح دائماً/);
assert.match(i18n.translate('Only this exact website name will be allowed:', 'ar'), /المحدد/);
assert.match(i18n.translate('Always allow this website', 'ar'), /السماح دائماً/);
assert.match(i18n.translate('Signals found in this URL', 'ar'), /إشارات/);
assert.match(i18n.signalText({ code: 'many_symbols' }, 'ar'), /الرموز/);
assert.match(i18n.translate('The URL is unusually long (180 characters)', 'ar'), /180/);
assert.match(i18n.translate('Shared content — use caution', 'ar'), /محتوى/);
assert.match(i18n.translate('Open the extension to see why this URL received this result.', 'ar'), /افتح الإضافة/);

for (const file of ['popup/popup.html', 'warning/warning.html', 'pilot/pilot.html', 'demo/demo.html']) {
  const html = read(file);
  assert.match(html, /data-language-toggle/, `${file} needs a language switch`);
  assert.match(html, /localization\.js/, `${file} needs the localization runtime`);
}
assert.match(read('content/advisory.html'), /localization\.js/);
assert.match(read('popup/popup.html'), /id="trust-hostname"/);
assert.match(read('warning/warning.html'), /Continue once \(do not allow\)/);
assert.match(read('warning/warning.html'), /id="trust-hostname"/);
assert.match(read('content/advisory.html'), /id="allow-hostname"/);

console.log('PASS: English/Arabic localization and clear exact-host allow actions are present.');
