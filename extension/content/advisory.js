const parameters = new URLSearchParams(location.hash.slice(1));
const tr = (text, values = {}) => globalThis.PhishGuardI18n?.translate(text, undefined, values) ?? text;
const tier = parameters.get("tier") ?? "caution";
const labels = {
  secondary_review: ["REVIEW SUGGESTED", "?"],
  caution: ["CAUTION ADVISED", "!"],
  warning: ["HIGH-RISK PATTERN", "!"],
  block: ["VERY HIGH-RISK PATTERN", "!"],
  ip_review: ["PUBLIC IP REVIEW", "i"],
};
const [riskLabel, icon] = labels[tier] ?? labels.caution;
document.body.className = tier;
document.getElementById("risk-label").textContent = tr(riskLabel);
document.getElementById("icon").textContent = icon;
document.getElementById("headline").textContent = tr(parameters.get("headline") ?? "Exercise caution on this site");
const hostname = parameters.get("host") ?? tr("Current destination");
document.getElementById("host").textContent = hostname;
document.getElementById("allow-hostname").textContent = hostname;
document.getElementById("allow-note-hostname").textContent = hostname;
document.getElementById("score").textContent = tr("Risk score {score}", { score: parameters.get("score") ?? tr("Unavailable") });
const allowButton = document.getElementById("allow-site");
const allowNote = document.getElementById("allow-note");
function restoreAllowLabel() {
  allowButton.replaceChildren();
  const prefix = document.createElement("span");
  prefix.textContent = tr("Always allow");
  const host = document.createElement("bdi");
  host.id = "allow-hostname";
  host.dir = "ltr";
  host.textContent = hostname;
  allowButton.append(prefix, " ", host);
}
const canAllow = ["caution", "warning", "block"].includes(tier);
allowButton.classList.toggle("hidden", !canAllow);
allowNote.classList.toggle("hidden", !canAllow);
allowButton.addEventListener("click", () => {
  allowButton.disabled = true;
  allowButton.textContent = tr("Allowing…");
  parent.postMessage({ type: "PHISHGUARD_TRUST_ADVISORY" }, "*");
});
window.addEventListener("message", (event) => {
  if (event.source !== parent || event.data?.type !== "PHISHGUARD_TRUST_RESULT") return;
  if (event.data.ok) return;
  allowButton.disabled = false;
  restoreAllowLabel();
  allowNote.textContent = tr(event.data.error ?? "The website could not be added to your allowed list.");
});
document.getElementById("close").addEventListener("click", () => {
  parent.postMessage({ type: "PHISHGUARD_DISMISS_ADVISORY" }, "*");
});
