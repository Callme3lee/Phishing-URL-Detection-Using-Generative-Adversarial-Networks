const parameters = new URLSearchParams(location.hash.slice(1));
const destination = parameters.get("url") ?? "";
const score = Number(parameters.get("score"));
const tier = parameters.get("tier") ?? "warning";
let hostname = "Unknown host";
const tr = (text, values = {}) => globalThis.PhishGuardI18n?.translate(text, undefined, values) ?? text;
const reportButton = document.getElementById("report-false-alarm");
const reportStatus = document.getElementById("report-status");
try { hostname = new URL(destination).hostname; } catch (_) {}
document.getElementById("host").textContent = hostname;
document.getElementById("trust-hostname").textContent = hostname;
document.getElementById("trust-note-hostname").textContent = hostname;
document.getElementById("url").textContent = destination;
document.getElementById("score").textContent = Number.isFinite(score) ? score.toFixed(6) : tr("Unavailable");
document.getElementById("tier").textContent = tr(tier.replaceAll("_", " "));
const signalsElement = document.getElementById("signals");
let signals = [];
try { signals = JSON.parse(parameters.get("signals") ?? "[]"); } catch (_) {}
for (const signal of signals.length ? signals : [{ code: "combined_pattern", values: {} }]) {
  const item = document.createElement("li");
  item.textContent = globalThis.PhishGuardI18n?.signalText(signal) ?? signal.code;
  signalsElement.append(item);
}
document.getElementById("back").addEventListener("click", () => history.length > 1 ? history.back() : window.close());
document.getElementById("continue").addEventListener("click", async () => {
  if (!destination) return;
  const response = await chrome.runtime.sendMessage({ type: "ALLOW_ONCE", url: destination });
  if (response?.ok) location.replace(destination);
});
document.getElementById("trust").addEventListener("click", async (event) => {
  if (!destination) return;
  const button = event.currentTarget;
  button.disabled = true;
  try {
    await chrome.runtime.sendMessage({
      type: "RECORD_PILOT_FEEDBACK", claim: "false_alarm", url: destination,
      source: "allowed_website",
    }).catch(() => {});
    const response = await chrome.runtime.sendMessage({ type: "TRUST_HOST", url: destination });
    if (response?.ok) location.replace(destination);
    else reportStatus.textContent = tr(response?.error ?? "The website could not be added to your allowed list.");
  } catch (error) {
    reportStatus.textContent = String(error?.message ?? error);
  } finally {
    button.disabled = false;
  }
});

chrome.runtime.sendMessage({ type: "GET_PILOT_STATUS" }).then((response) => {
  reportButton.classList.toggle("hidden", !response?.ok || !response.enabled);
}).catch(() => {});

reportButton.addEventListener("click", async () => {
  reportButton.disabled = true;
  try {
    const response = await chrome.runtime.sendMessage({
      type: "RECORD_PILOT_FEEDBACK",
      claim: "false_alarm",
      url: destination,
      score,
      modelTier: tier,
      effectiveTier: "block",
    });
    reportStatus.textContent = response?.ok
      ? tr("Saved locally. Query values and fragments were removed.")
      : tr(response?.error ?? "The report could not be saved.");
  } catch (error) {
    reportStatus.textContent = String(error?.message ?? error);
  } finally { reportButton.disabled = false; }
});
