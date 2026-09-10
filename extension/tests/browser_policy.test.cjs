const assert = require("node:assert/strict");
const policy = require("../browser_policy.js");

const result = (tier, rawUrl = "https://example.com/a/long/path?x=1") => ({ tier, rawUrl });

assert.equal(policy.decide(result("allow")).interrupt, false);
assert.equal(policy.decide(result("secondary_review")).interrupt, false);
assert.equal(policy.decide(result("caution")).interrupt, false);
assert.equal(policy.decide(result("warning")).interrupt, true, "high risk must interrupt in normal mode");
assert.equal(policy.decide(result("block")).interrupt, true, "very high risk must interrupt in normal mode");
const passiveWarning = policy.decide(result("warning"), [], { mode: "passive" });
assert.equal(passiveWarning.interrupt, false, "passive mode must not interrupt high risk");
const passiveBlock = policy.decide(result("block"), [], { mode: "passive" });
assert.equal(passiveBlock.interrupt, false, "passive mode must never interrupt");
assert.equal(passiveBlock.mode, "passive");
const continued = policy.decide(result("block"), [], { continuedOnce: true });
assert.equal(continued.interrupt, false, "continue-once must preserve the warning without redirecting again");
assert.equal(continued.continuedOnce, true);

const localIp = policy.decide(result("block", "http://192.168.1.1/admin"));
assert.equal(localIp.effectiveTier, "ip_local");
assert.equal(localIp.interrupt, false, "valid private IP addresses must not be model-blocked");
const publicIp = policy.decide(result("block", "https://8.8.8.8/"));
assert.equal(publicIp.effectiveTier, "ip_review");
assert.equal(publicIp.interrupt, false, "valid public IP addresses require passive review");
assert.equal(policy.ipAddress("http://999.8.8.8/"), null, "malformed IPv4 must not receive the IP exception");
assert.equal(policy.ipAddress("http://[::1]/").local, true);

const trusted = policy.decide(result("block"), ["example.com"]);
assert.equal(trusted.effectiveTier, "trusted");
assert.equal(trusted.interrupt, false);
assert.equal(policy.isTrusted("https://sub.example.com/", ["example.com"]), false, "trust must be exact-host only");
assert.equal(policy.hostname("https://Example.COM./x"), "example.com");

const sharedCatalog = {
  policy_version: policy.POLICY_VERSION,
  global_excluded_path_pattern: "(?:^|/)(?:url|redirect)(?:/|$)",
  reviewed_rules: [{id: "google-workspace", hosts: ["docs.google.com"],
    allowed_path_pattern: "^/", registrable_domain: "google.com", tranco_rank: 1}],
};
const sharedBlock = policy.decide(result("block", "https://docs.google.com/forms/d/e/test/viewform"), [], {sitePolicy: sharedCatalog});
assert.equal(sharedBlock.reason, "reviewed_user_content");
assert.equal(sharedBlock.effectiveTier, "block");
assert.equal(sharedBlock.interrupt, true);
const sharedWarning = policy.decide(result("warning", "https://docs.google.com/presentation/d/e/test/pub"), [], {sitePolicy: sharedCatalog});
assert.equal(sharedWarning.effectiveTier, "warning");
assert.equal(sharedWarning.interrupt, true);
const fixedDocs = policy.decide(result("block", "https://docs.google.com/"), [], {sitePolicy: sharedCatalog});
assert.equal(fixedDocs.effectiveTier, "allow");
assert.equal(fixedDocs.reason, "recognized_service");

assert.equal(policy.shouldShowAdvisory(policy.decide(result("allow"))), false);
assert.equal(policy.shouldShowAdvisory(policy.decide(result("secondary_review"))), true);
assert.equal(policy.shouldShowAdvisory(policy.decide(result("caution"))), true);
assert.equal(policy.shouldShowAdvisory(policy.decide(result("warning"))), false);
assert.equal(policy.shouldShowAdvisory(policy.decide(result("block"))), false, "an interrupted page must not also receive an advisory");
assert.equal(policy.shouldShowAdvisory(continued), true, "continued strict warnings must remain visible in-page");
assert.equal(policy.shouldShowAdvisory(passiveBlock), true, "passive strict warnings must remain visible in-page");
assert.equal(policy.shouldShowAdvisory(localIp), false);
assert.equal(policy.shouldShowAdvisory(publicIp), true);
assert.equal(policy.shouldShowAdvisory(trusted), false);

console.log("PASS: navigation policy, in-page advisories, IP review, and exact-host trust are enforced.");
