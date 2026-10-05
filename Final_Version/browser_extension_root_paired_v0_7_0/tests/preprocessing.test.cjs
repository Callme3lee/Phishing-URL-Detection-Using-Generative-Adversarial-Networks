const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createPreprocessor } = require("../preprocessing.js");

const root = path.resolve(__dirname, "..");
const read = (relative) => JSON.parse(fs.readFileSync(path.join(root, relative), "utf8"));
const preprocessor = createPreprocessor({
  vocabulary: read("models/vocabulary.json"),
  scaler: read("models/scaler.json"),
  pslRules: read("models/psl_rules.json"),
  htmlEntities: read("models/html_entities.json"),
});
const suite = read("tests/parity_fixtures.json");

function maximumDifference(left, right) {
  assert.equal(left.length, right.length);
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) maximum = Math.max(maximum, Math.abs(Number(left[index]) - Number(right[index])));
  return maximum;
}

for (const fixture of suite.fixtures) {
  const output = preprocessor.prepare(fixture.raw_url);
  assert.equal(output.canonicalUrl, fixture.canonical_url, fixture.raw_url);
  assert.equal(output.hostname, fixture.hostname, fixture.raw_url);
  assert.equal(output.scheme, fixture.scheme, fixture.raw_url);
  assert.deepEqual(Array.from(output.tokens, Number), fixture.tokens, `${fixture.raw_url}: token mismatch`);
  assert.ok(maximumDifference(output.rawLexical, fixture.raw_lexical) <= suite.feature_tolerance,
    `${fixture.raw_url}: raw lexical mismatch`);
  assert.ok(maximumDifference(output.lexical, fixture.scaled_lexical) <= suite.feature_tolerance,
    `${fixture.raw_url}: scaled lexical mismatch`);
}

console.log(`PASS: ${suite.fixtures.length} browser preprocessing fixtures match Python.`);
