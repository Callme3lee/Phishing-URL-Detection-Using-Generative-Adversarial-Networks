const urlElement = document.getElementById("url");
document.getElementById("release-version").textContent = `v${chrome.runtime.getManifest().version}`;
const hostElement = document.getElementById("host");
const resultElement = document.getElementById("result");
const statusGlyphElement = document.getElementById("status-glyph");
const tierElement = document.getElementById("tier");
const scoreElement = document.getElementById("score");
const modelTierElement = document.getElementById("model-tier");
const actionElement = document.getElementById("action");
const signalsPanel = document.getElementById("signals-panel");
const signalsElement = document.getElementById("signals");
const latencyElement = document.getElementById("latency");
const scanButton = document.getElementById("scan");
const demoButton = document.getElementById("demo");
const trustButton = document.getElementById("trust");
const trustPanel = document.getElementById("trust-panel");
const trustHostname = document.getElementById("trust-hostname");
const trustNoteHostname = document.getElementById("trust-note-hostname");
const untrustButton = document.getElementById("untrust");
const modeSelect = document.getElementById("mode");
const modeNote = document.getElementById("mode-note");
const pilotStatusElement = document.getElementById("pilot-status");
const feedbackActions = document.getElementById("feedback-actions");
const feedbackNote = document.getElementById("feedback-note");
const pilotPageButton = document.getElementById("pilot-page");
const falseAlarmButton = document.getElementById("false-alarm");
const missedPhishingButton = document.getElementById("missed-phishing");
let currentUrl = "";
let currentResult = null;
let pilotEnabled = false;
const tr = (text, values = {}) => globalThis.PhishGuardI18n?.translate(text, undefined, values) ?? text;

const statusGlyphs = {
  trusted: "✓", allow: "✓", ip_local: "i", secondary_review: "?",
  ip_review: "i", caution: "!", warning: "!", block: "×", neutral: "·",
};

function setScanButton(busy) {
  scanButton.disabled = busy;
  scanButton.replaceChildren();
  const icon = document.createElement("span");
  icon.className = "button-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = busy ? "◌" : "⌁";
  const label = document.createElement("span");
  label.textContent = tr(busy ? "Checking privately…" : "Check this page");
  scanButton.append(icon, label);
}

function showMode(mode) {
  const passive = mode === "passive";
  modeSelect.value = passive ? "passive" : "normal";
  modeNote.textContent = tr(passive
    ? "Warnings only shows alerts but never pauses a page."
    : "Normal mode pauses only the highest-risk URLs.");
}

async function loadMode() {
  const response = await chrome.runtime.sendMessage({ type: "GET_MODE" });
  if (response?.ok) showMode(response.mode);
}

async function changeMode() {
  modeSelect.disabled = true;
  try {
    const response = await chrome.runtime.sendMessage({ type: "SET_MODE", mode: modeSelect.value });
    if (response?.ok) showMode(response.mode); else modeNote.textContent = response?.error ?? "Unable to change mode.";
  } finally { modeSelect.disabled = false; }
}

function display(result) {
  currentResult = result;
  const decision = result.decision ?? { effectiveTier: result.tier, headline: result.tier, explanation: result.action, modelTier: result.tier };
  resultElement.className = `result ${decision.effectiveTier}`;
  statusGlyphElement.textContent = statusGlyphs[decision.effectiveTier] ?? "·";
  tierElement.textContent = tr(decision.headline);
  scoreElement.textContent = Number.isFinite(result.score) ? result.score.toFixed(4) : tr("Unavailable");
  const normalized = result.queryView?.tier ? ` · without query values: ${result.queryView.tier.replaceAll("_", " ")}` : "";
  modelTierElement.textContent = `Full URL result: ${decision.modelTier.replaceAll("_", " ")}${normalized} · the score is not a safety percentage`;
  actionElement.textContent = tr(decision.explanation);
  latencyElement.textContent = tr("{milliseconds} ms check time · processed on this device", { milliseconds: result.inferenceMilliseconds.toFixed(1) });
  displaySignals(result, decision.effectiveTier);
  setTrustButtons(decision.effectiveTier === "trusted");
}

function displaySignals(result, effectiveTier) {
  const warned = ["secondary_review", "caution", "warning", "block"].includes(effectiveTier);
  signalsPanel.classList.toggle("hidden", !warned);
  signalsElement.replaceChildren();
  if (!warned) return;
  const signals = result.explanation?.signals ?? [];
  const displayed = signals.length ? signals : [{ code: "combined_pattern", values: {} }];
  for (const signal of displayed) {
    const item = document.createElement("li");
    item.textContent = globalThis.PhishGuardI18n?.signalText(signal) ?? signal.code;
    signalsElement.append(item);
  }
}

function setTrustButtons(trusted) {
  const scannable = /^https?:\/\//i.test(currentUrl);
  let currentHostname = tr("this website");
  try { currentHostname = new URL(currentUrl).hostname || currentHostname; } catch (_) {}
  trustHostname.textContent = currentHostname;
  trustNoteHostname.textContent = currentHostname;
  const tier = currentResult?.decision?.effectiveTier ?? currentResult?.tier;
  const warned = Boolean(tier && !["allow", "trusted", "ip_local"].includes(tier));
  trustPanel.classList.toggle("hidden", !scannable || trusted || !warned);
  untrustButton.classList.toggle("hidden", !scannable || !trusted);
}

async function refreshTrust() {
  if (!/^https?:\/\//i.test(currentUrl)) return setTrustButtons(false);
  const response = await chrome.runtime.sendMessage({ type: "GET_TRUST_STATUS", url: currentUrl });
  if (response?.ok) setTrustButtons(response.trusted);
}

async function changeTrust(type) {
  await refreshCurrentUrl();
  const response = await chrome.runtime.sendMessage({ type, url: currentUrl });
  if (!response?.ok) return void (actionElement.textContent = response?.error ?? "Unable to change trusted-site status.");
  await scan();
}

async function scan() {
  await refreshCurrentUrl();
  if (!currentUrl || !/^https?:\/\//i.test(currentUrl)) {
    actionElement.textContent = "Browser settings pages cannot be checked. Open a website or use the safe URL checker.";
    return;
  }
  setScanButton(true);
  const scannedUrl = currentUrl;
  try {
    const response = await chrome.runtime.sendMessage({ type: "SCORE_URL", url: scannedUrl });
    await refreshCurrentUrl();
    if (currentUrl !== scannedUrl) {
      actionElement.textContent = "The page changed during the check. Check the current page again.";
      return null;
    }
    if (response?.ok) {
      display(response.result);
      return response.result;
    }
    actionElement.textContent = response?.error ?? "The URL could not be checked.";
  } catch (error) {
    actionElement.textContent = `The check failed: ${String(error?.message ?? error)}`;
  } finally {
    setScanButton(false);
  }
  return null;
}

async function loadPilotStatus() {
  const response = await chrome.runtime.sendMessage({ type: "GET_PILOT_STATUS" });
  pilotEnabled = response?.ok && response.enabled;
  feedbackActions.classList.toggle("hidden", !pilotEnabled);
  pilotStatusElement.textContent = pilotEnabled
    ? tr("{count} unique URLs · {reports} reports", { count: response.summary?.uniqueAutomaticUrls ?? 0, reports: response.summary?.feedbackReports ?? 0 })
    : tr("Local history is off. Open Activity to turn it on.");
}

async function submitFeedback(claim) {
  if (!pilotEnabled) return void openPilotPage();
  await refreshCurrentUrl();
  if (!/^https?:\/\//i.test(currentUrl)) {
    feedbackNote.textContent = "Open an HTTP(S) page before reporting feedback.";
    return;
  }
  falseAlarmButton.disabled = true;
  missedPhishingButton.disabled = true;
  try {
    const result = currentResult ?? await scan();
    if (!result) throw new Error("The current URL could not be scored.");
    const response = await chrome.runtime.sendMessage({
      type: "RECORD_PILOT_FEEDBACK",
      claim,
      url: currentUrl,
      score: result.score,
      modelTier: result.tier,
      effectiveTier: result.decision?.effectiveTier ?? result.tier,
    });
    if (!response?.ok) throw new Error(response?.error ?? "Feedback could not be stored.");
    feedbackNote.textContent = claim === "false_alarm"
      ? "Saved locally as an incorrect warning. Private query values were removed."
      : "Saved locally as a missed phishing threat. Private query values were removed.";
    pilotStatusElement.textContent = `${response.summary.uniqueAutomaticUrls} unique URLs · ${response.summary.feedbackReports} reports`;
  } catch (error) {
    feedbackNote.textContent = String(error?.message ?? error);
  } finally {
    falseAlarmButton.disabled = false;
    missedPhishingButton.disabled = false;
  }
}

function openPilotPage() {
  void chrome.tabs.create({ url: chrome.runtime.getURL("pilot/pilot.html") });
}

async function refreshCurrentUrl() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const nextUrl = tab?.pendingUrl || tab?.url || "";
  if (nextUrl !== currentUrl) {
    currentResult = null;
    resultElement.className = "result neutral";
    tierElement.textContent = "Check the current page";
    scoreElement.textContent = "—";
    modelTierElement.textContent = "";
    latencyElement.textContent = "";
    signalsPanel.classList.add("hidden");
    signalsElement.replaceChildren();
  }
  currentUrl = nextUrl; urlElement.textContent = currentUrl || "No scannable URL.";
  try { hostElement.textContent = new URL(currentUrl).hostname || "Current browser page"; }
  catch (_) { hostElement.textContent = "Current browser page"; }
  if (!/^https?:\/\//i.test(currentUrl)) actionElement.textContent = "This browser page cannot be checked. Open a website or use the safe URL checker below.";
  await refreshTrust();
}
void refreshCurrentUrl();
void loadMode();
void loadPilotStatus();
modeSelect.addEventListener("change", () => { void changeMode(); });
scanButton.addEventListener("click", () => { void scan(); });
trustButton.addEventListener("click", async () => {
  trustButton.disabled = true;
  try {
    await chrome.runtime.sendMessage({
      type: "RECORD_PILOT_FEEDBACK", claim: "false_alarm", url: currentUrl,
      source: "allowed_website",
    }).catch(() => {});
    await changeTrust("TRUST_HOST");
  }
  finally { trustButton.disabled = false; }
});
untrustButton.addEventListener("click", () => { void changeTrust("UNTRUST_HOST"); });
demoButton.addEventListener("click", () => { void chrome.runtime.openOptionsPage(); });
pilotPageButton.addEventListener("click", openPilotPage);
falseAlarmButton.addEventListener("click", () => { void submitFeedback("false_alarm"); });
missedPhishingButton.addEventListener("click", () => { void submitFeedback("missed_phishing"); });
