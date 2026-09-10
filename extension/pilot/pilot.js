const consentCard = document.getElementById("consent-card");
const dashboard = document.getElementById("dashboard");
const consentMessage = document.getElementById("consent-message");
const agreeButton = document.getElementById("agree");
const declineButton = document.getElementById("decline");
const pauseButton = document.getElementById("pause");

let currentStudy = null;
let participationEnabled = false;
const tr = (text, values = {}) => globalThis.PhishGuardI18n?.translate(text, undefined, values) ?? text;

function setBusy(busy) {
  [agreeButton, declineButton, pauseButton].forEach((button) => { button.disabled = busy; });
}

function showStatus(status) {
  participationEnabled = status.enabled;
  currentStudy = status.study ?? null;
  consentCard.classList.toggle("hidden", status.enabled || status.hasStudy);
  dashboard.classList.toggle("hidden", !status.hasStudy);
  if (status.hasStudy) {
    document.getElementById("participation-label").textContent = tr(status.enabled
      ? "Local history is on" : "Local history is paused");
    document.querySelector(".live-dot").classList.toggle("paused", !status.enabled);
    pauseButton.textContent = tr(status.enabled ? "Pause history" : "Resume history");
  }
  if (!status.summary) return;
  const summary = status.summary;
  document.getElementById("participant-id").textContent = summary.participantId;
  document.getElementById("unique-urls").textContent = summary.uniqueAutomaticUrls.toLocaleString();
  document.getElementById("model-runs").textContent = summary.totalModelExecutions.toLocaleString();
  document.getElementById("duplicates-suppressed").textContent = summary.duplicateAssessmentsSuppressed.toLocaleString();
  document.getElementById("false-alarms").textContent = summary.falseAlarmReports.toLocaleString();
  document.getElementById("missed-phishing").textContent = summary.missedPhishingReports.toLocaleString();
  document.getElementById("average-latency").textContent = summary.averageInferenceMilliseconds === null
    ? "—" : `${summary.averageInferenceMilliseconds.toFixed(1)} ms`;
  document.getElementById("legacy-notice").classList.toggle("hidden", !summary.legacyTelemetryPresent);
  const counts = document.getElementById("decision-counts");
  counts.replaceChildren(...Object.entries(summary.decisionCounts).map(([tier, count]) => {
    const chip = document.createElement("span");
    chip.className = "decision-chip";
    const friendlyNames = {
      trusted: "allowed by you", allow: "no warning", ip_local: "local IP",
      secondary_review: "review suggested", ip_review: "public IP review",
      caution: "caution", warning: "high risk", block: "very high risk", unknown: "not available",
    };
    const label = document.createTextNode(`${tr(friendlyNames[tier] ?? tier.replaceAll("_", " "))} `);
    const value = document.createElement("strong");
    value.textContent = count.toLocaleString();
    chip.append(label, value);
    return chip;
  }));
  const measures = document.createElement("p");
  measures.textContent = tr("{paused} pages paused · {submissions} feedback submissions · {evicted} older reports removed after reaching the storage limit", {
    paused: summary.navigationInterrupts ?? 0, submissions: summary.feedbackSubmissions ?? 0, evicted: summary.feedbackRecordsEvicted ?? 0,
  });
  counts.appendChild(measures);
  const services = document.createElement("p");
  services.textContent = tr("{services} recognised service pages allowed quietly · {checks} model checks completed", {
    services: summary.servicePageDecisions ?? 0, checks: summary.modelViewEvaluations ?? 0,
  });
  counts.appendChild(services);
  if (summary.untrackedAssessments) {
    const notice = document.createElement("p");
    notice.textContent = tr("Local history is full. Export and erase it to begin a new record. URL protection continues.");
    counts.appendChild(notice);
  }
}

async function refresh() {
  const response = await chrome.runtime.sendMessage({ type: "GET_PILOT_STATUS" });
    if (!response?.ok) throw new Error(response?.error ?? tr("Unable to read local activity."));
  showStatus(response);
}

async function setParticipation(enabled) {
  setBusy(true);
  try {
    const response = await chrome.runtime.sendMessage({ type: "SET_PILOT_PARTICIPATION", enabled });
    if (!response?.ok) throw new Error(response?.error ?? tr("Unable to update local history settings."));
    showStatus({ ...response, hasStudy: Boolean(response.study) });
    consentMessage.textContent = tr(enabled
      ? "Local detection history is on." : "Local history is off. URL protection still works.");
  } catch (error) {
    consentMessage.textContent = String(error?.message ?? error);
  } finally { setBusy(false); }
}

function downloadFile(filename, contents, mimeType) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([contents], { type: mimeType }));
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function exportResults(format) {
  const response = await chrome.runtime.sendMessage({ type: "GET_PILOT_EXPORT" });
  if (!response?.ok || !response.study) return;
  const date = new Date().toISOString().slice(0, 10);
  const shortId = response.study.participantId.slice(0, 8);
  if (format === "json") {
    downloadFile(`phishguard-activity-${shortId}-${date}.json`, `${JSON.stringify(response.study, null, 2)}\n`, "application/json");
  } else {
    downloadFile(`phishguard-activity-summary-${shortId}-${date}.csv`, PhishingPilotStore.summaryCsv(response.study), "text/csv");
  }
}

async function resetStudy() {
  if (!confirm(tr("Erase all local activity counts, repeat-check data, and feedback reports? This cannot be undone."))) return;
  const response = await chrome.runtime.sendMessage({ type: "RESET_PILOT_STUDY" });
  if (response?.ok) showStatus({ ...response, hasStudy: true });
}

agreeButton.addEventListener("click", () => { void setParticipation(true); });
declineButton.addEventListener("click", () => { void setParticipation(false); });
pauseButton.addEventListener("click", () => { void setParticipation(!participationEnabled); });
document.getElementById("export-json").addEventListener("click", () => { void exportResults("json"); });
document.getElementById("export-csv").addEventListener("click", () => { void exportResults("csv"); });
document.getElementById("reset").addEventListener("click", () => { void resetStudy(); });
document.getElementById("open-demo").addEventListener("click", () => { void chrome.runtime.openOptionsPage(); });
void refresh().catch((error) => { consentMessage.textContent = String(error?.message ?? error); });
