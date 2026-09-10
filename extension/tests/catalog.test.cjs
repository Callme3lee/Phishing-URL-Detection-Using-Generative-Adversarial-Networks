const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const policy = JSON.parse(fs.readFileSync(path.join(root, 'policy/site_policy.json'), 'utf8'));
const csv = fs.readFileSync(path.join(root, 'policy/tranco_XN6XN_top250.csv'));
const rows = csv.toString('utf8').trim().split(/\r?\n/).map(line => line.split(','));
assert.equal(policy.schema_version, 1);
assert.equal(policy.policy_version, 'reviewed_site_catalog_v3');
assert.equal(policy.popularity_snapshot.list_id, 'XN6XN');
assert.equal(rows.length, 250);
assert.equal(crypto.createHash('sha256').update(csv).digest('hex'), policy.popularity_snapshot.shortlist_sha256);
const ranks = {};
rows.forEach(([rawRank, domain], index) => {
  const rank = Number(rawRank);
  assert.equal(rank, index + 1);
  assert.match(domain, /^[a-z0-9.-]+$/);
  assert.equal(ranks[domain], undefined);
  ranks[domain] = rank;
});
const ids = new Set(); const hosts = new Set();
for (const rule of policy.reviewed_rules) {
  assert.ok(!ids.has(rule.id)); ids.add(rule.id);
  assert.equal(ranks[rule.registrable_domain], rule.tranco_rank, rule.id);
  new RegExp(rule.allowed_path_pattern);
  if (rule.excluded_path_pattern) new RegExp(rule.excluded_path_pattern, 'i');
  assert.ok(rule.review_basis.length >= 20);
  for (const host of rule.hosts) {
    assert.ok(!hosts.has(host), host); hosts.add(host);
    assert.ok(host === rule.registrable_domain || host.endsWith('.' + rule.registrable_domain), `${host} vs ${rule.registrable_domain}`);
  }
}
for (const domain of ['cloudflare.com', 'gstatic.com', 'googleapis.com', 'amazonaws.com', 'akamai.net', 'fbcdn.net']) {
  assert.ok(ranks[domain], domain);
  assert.ok(!policy.reviewed_rules.some(rule => rule.registrable_domain === domain), `${domain} must remain popularity-only infrastructure`);
}
console.log(`PASS: frozen Tranco ${policy.popularity_snapshot.list_id} top-${rows.length}, ${ids.size} reviewed rules and ${hosts.size} exact hosts are internally consistent.`);
