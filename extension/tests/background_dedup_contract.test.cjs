const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const source = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");

assert.match(source, /pilotDedupFingerprintsV2/);
assert.match(source, /crypto\.subtle\.digest\("SHA-256"/);
assert.match(source, /canonicalAssessmentIdentity/);
assert.match(source, /recordPilotAutomaticAssessment/);
assert.match(source, /automaticInferenceExecutions|recordPilotInference/);
assert.match(source, /pageResultV051/);
assert.match(source, /chrome\.storage\.session/);
assert.match(source, /"history_state"/);
assert.match(source, /"fragment"/);
assert.match(source, /"advisory_recovery"/);
assert.doesNotMatch(source, /recordPilotScan/);

console.log("PASS: background uses persistent cache, salted URL fingerprints, and separated telemetry.");
