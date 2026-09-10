const assert = require("node:assert/strict");
const Pilot = require("../pilot_store.js");

assert.equal(Pilot.SCHEMA_VERSION, 6);

const sanitized = Pilot.sanitizeFeedbackUrl(
  "https://user:secret@Example.COM/login/reset?token=private&email=person%40example.com#account",
);
assert.equal(sanitized.url, "https://example.com/login/reset");
assert.equal(sanitized.queryValuesRemoved, true);
assert.equal(sanitized.queryParameterCount, 2);
assert.equal(sanitized.fragmentRemoved, true);
assert.ok(!sanitized.url.includes("secret"));
assert.ok(!sanitized.url.includes("private"));
assert.throws(() => Pilot.sanitizeFeedbackUrl("chrome://settings"), /HTTP/);

assert.equal(
  Pilot.canonicalAssessmentIdentity({ canonicalUrl: "example.com/login?x=1#first" }),
  "example.com/login?x=1",
);
assert.equal(
  Pilot.canonicalAssessmentIdentity({ canonicalUrl: "example.com/login?x=1#second" }),
  "example.com/login?x=1",
  "fragment-only changes must share one deduplication identity",
);
assert.notEqual(
  Pilot.canonicalAssessmentIdentity({ canonicalUrl: "example.com/login?x=1" }),
  Pilot.canonicalAssessmentIdentity({ canonicalUrl: "example.com/login?x=2" }),
  "different query-bearing model inputs remain distinct URLs",
);

const study = Pilot.createStudy("participant-test", "2026-08-28T10:00:00.000Z", "0.5.1");
assert.equal(study.schemaVersion, 6);
assert.equal(study.counters.uniqueAutomaticUrls, 0);

const result = {
  canonicalUrl: "example.com/account",
  tier: "warning",
  inferenceMilliseconds: 12.5,
  decision: { effectiveTier: "caution" },
};
Pilot.recordInference(study, result, "automatic", "2026-08-28T10:01:00.000Z");
Pilot.recordAutomaticAssessment(study, result, true, "navigation", "2026-08-28T10:01:00.000Z");
Pilot.recordAutomaticAssessment(study, result, false, "fragment", "2026-08-28T10:01:01.000Z");
Pilot.recordAutomaticAssessment(study, result, false, "advisory_recovery", "2026-08-28T10:01:02.000Z");
Pilot.recordInference(study, {
  canonicalUrl: "manual.example/path",
  tier: "allow",
  inferenceMilliseconds: 7.5,
  decision: { effectiveTier: "allow" },
}, "manual", "2026-08-28T10:02:00.000Z");
Pilot.recordError(study, "2026-08-28T10:03:00.000Z");
Pilot.recordAction(study, "advisory_dismiss", "2026-08-28T10:04:00.000Z");

Pilot.addFeedback(study, {
  claim: "false_alarm",
  url: "https://example.com/account?id=private#section",
  score: 0.8,
  modelTier: "warning",
  effectiveTier: "warning",
}, "2026-08-28T10:05:00.000Z", "report-1");

Pilot.addFeedback(study, {
  claim: "missed_phishing",
  url: "https://example.com/account?different=secret",
  score: 0.2,
  modelTier: "allow",
  effectiveTier: "allow",
}, "2026-08-28T10:06:00.000Z", "report-2");

const summary = Pilot.summary(study);
assert.equal(summary.uniqueAutomaticUrls, 1);
assert.equal(summary.automaticInferenceExecutions, 1);
assert.equal(summary.manualScans, 1);
assert.equal(summary.totalModelExecutions, 2);
assert.equal(summary.duplicateAssessmentsSuppressed, 2);
assert.equal(summary.decisionCounts.caution, 1);
assert.equal(summary.decisionCounts.allow, 0, "manual checks must not alter unique automatic decision counts");
assert.equal(summary.duplicateEventCounts.fragment, 1);
assert.equal(summary.duplicateEventCounts.advisory_recovery, 1);
assert.equal(summary.averageInferenceMilliseconds, 10);
assert.equal(summary.inferenceErrors, 1);
assert.equal(summary.falseAlarmReports, 0);
assert.equal(summary.missedPhishingReports, 1);
assert.equal(study.feedbackReports.length, 1);
assert.equal(study.feedbackReports[0].reportedUrl, "https://example.com/account");

const csv = Pilot.summaryCsv(study);
assert.match(csv, /"unique_automatic_urls","1"/);
assert.match(csv, /"automatic_inference_executions","1"/);
assert.match(csv, /"duplicate_assessments_suppressed","2"/);
assert.match(csv, /"unique_decision_caution","1"/);
assert.ok(!csv.includes("example.com/account"));

const legacy = {
  schemaVersion: 1,
  participantId: "legacy-user",
  extensionVersion: "0.5.0",
  createdAt: "2026-08-24T00:00:00.000Z",
  consentedAt: "2026-08-24T00:00:00.000Z",
  lastUpdatedAt: "2026-08-27T00:00:00.000Z",
  counters: {
    automaticScans: 1000,
    manualScans: 12,
    falseAlarmReports: 1,
    missedPhishingReports: 0,
  },
  decisionCounts: { allow: 900, warning: 100 },
  latency: { count: 1012, totalMilliseconds: 8096, minimumMilliseconds: 4, maximumMilliseconds: 30 },
  feedbackReports: [{ claim: "false_alarm", reportedUrl: "https://reported.example/" }],
};
const migration = Pilot.migrateStudy(legacy, "2026-08-28T12:00:00.000Z", "0.5.1");
assert.equal(migration.migrated, true);
assert.equal(migration.study.schemaVersion, 6);
assert.equal(migration.study.measurementStartedAt, "2026-08-28T12:00:00.000Z");
assert.equal(migration.study.counters.uniqueAutomaticUrls, 0);
assert.equal(migration.study.counters.automaticInferenceExecutions, 0);
assert.equal(migration.study.legacyV05Telemetry.counters.automaticScans, 1000);
assert.match(migration.study.legacyV05Telemetry.reliabilityNotice, /excluded/);
assert.equal(migration.study.feedbackReports.length, 0, "new period does not mix legacy feedback");
assert.equal(migration.study.legacyStudy.feedbackReports.length, 1, "deliberate feedback survives in the archive");
assert.equal(Pilot.summary(migration.study).legacyTelemetryPresent, true);

console.log("PASS: v0.6.0 unique-URL accounting, feedback privacy, and legacy migration are enforced.");
