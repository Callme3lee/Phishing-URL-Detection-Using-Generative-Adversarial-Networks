const examples = [
  ["Ordinary reserved domain", "https://example.com/"],
  ["Redirect-style query", "https://example.com/path?redirect=https%3A%2F%2Ftarget.example%2F"],
  ["Hosted login path", "https://foo.github.io/project/login"],
  ["Encoded payment wording", "https://example.com/%70%61%79%6d%65%6e%74?account=secure&verify=true"],
  ["Reserved IP login pattern", "http://192.0.2.10/update/password"],
];
const input = document.getElementById("url");
const button = document.getElementById("scan");
const resultElement = document.getElementById("result");
const tierElement = document.getElementById("tier");
const scoreElement = document.getElementById("score");
const actionElement = document.getElementById("action");
const latencyElement = document.getElementById("latency");
const statusGlyphElement = document.getElementById("status-glyph");
const buttonLabelElement = button.querySelector(".button-label");
const tr = (text, values = {}) => globalThis.PhishGuardI18n?.translate(text, undefined, values) ?? text;

const STATUS_GLYPHS = {
  trusted: "✓", allow: "✓", ip_local: "i", ip_review: "i",
  secondary_review: "?", caution: "!", warning: "!", block: "×", error: "!",
};

function setBusy(busy) {
  button.disabled = busy;
  buttonLabelElement.textContent = tr(busy ? "Scoring…" : "Score URL");
}

function display(result) {
  const decision = result.decision ?? { effectiveTier: result.tier, headline: result.tier, explanation: result.action };
  resultElement.className = `result ${decision.effectiveTier}`;
  statusGlyphElement.textContent = STATUS_GLYPHS[decision.effectiveTier] || "—";
  tierElement.textContent = tr(decision.headline);
  scoreElement.textContent = Number.isFinite(result.score) ? result.score.toFixed(4) : tr("Unavailable");
  actionElement.textContent = tr(decision.explanation);
  latencyElement.textContent = tr("{milliseconds} ms check time · processed on this device", { milliseconds: result.inferenceMilliseconds.toFixed(1) });
}
async function scan() {
  const url = input.value.trim(); if (!url) return;
  setBusy(true);
  try {
    const response = await chrome.runtime.sendMessage({ type: "SCORE_URL", url });
    if (response?.ok) display(response.result);
    else {
      resultElement.className = "result error";
      statusGlyphElement.textContent = "!";
      tierElement.textContent = tr("Unable to check URL");
      actionElement.textContent = tr(response?.error ?? "The URL check failed.");
    }
  } catch (error) {
    resultElement.className = "result error";
    statusGlyphElement.textContent = "!";
    tierElement.textContent = tr("Extension unavailable");
    actionElement.textContent = `Extension error: ${String(error?.message ?? error)}`;
  } finally {
    setBusy(false);
  }
}
for (const [label, url] of examples) {
  const choice = document.createElement("button"); choice.className = "example"; choice.type = "button";
  const title = document.createElement("span"); title.textContent = tr(label);
  const value = document.createElement("small"); value.textContent = url;
  choice.append(title, value); choice.addEventListener("click", () => { input.value = url; void scan(); });
  document.getElementById("examples").append(choice);
}
input.value = examples[0][1];
button.addEventListener("click", () => { void scan(); });
input.addEventListener("keydown", (event) => { if (event.key === "Enter") void scan(); });
