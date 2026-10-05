(() => {
  "use strict";
  if (window.top !== window) return;

  const FRAME_ID = "__phishguard-caution-frame";
  const dismissed = new Set();
  let frame = null;
  let currentKey = "";
  let displayedUrl = "";

  function removeAdvisory(remember = false) {
    if (remember && currentKey) dismissed.add(currentKey);
    if (remember) {
      void chrome.runtime.sendMessage({ type: "RECORD_PILOT_ACTION", action: "advisory_dismiss" }).catch(() => {});
    }
    frame?.remove();
    frame = null;
    currentKey = "";
    displayedUrl = "";
  }

  function showAdvisory(result, scannedUrl) {
    if (scannedUrl !== location.href) return;
    const decision = result?.decision;
    if (!PhishingBrowserPolicy.shouldShowAdvisory(decision)) {
      removeAdvisory();
      return;
    }

    const key = `${scannedUrl}|${decision.effectiveTier}|${Number(result.score).toFixed(6)}`;
    if (dismissed.has(key) || currentKey === key) return;
    removeAdvisory();
    currentKey = key;
    displayedUrl = scannedUrl;

    const parameters = new URLSearchParams({
      tier: decision.effectiveTier,
      headline: decision.headline,
      host: PhishingBrowserPolicy.hostname(scannedUrl) || "Current destination",
      score: Number.isFinite(Number(result.score)) ? Number(result.score).toFixed(4) : "Unavailable",
    });
    frame = document.createElement("iframe");
    frame.id = FRAME_ID;
    frame.title = "Phishing URL Guard caution notification";
    frame.src = `${chrome.runtime.getURL("content/advisory.html")}#${parameters}`;
    frame.addEventListener("load", () => {
      if (scannedUrl === location.href) {
        void chrome.runtime.sendMessage({ type: "ADVISORY_DISPLAYED" }).catch(() => {});
      }
    }, { once: true });
    (document.documentElement || document.body).appendChild(frame);
  }

  window.addEventListener("message", (event) => {
    if (!frame || event.source !== frame.contentWindow) return;
    if (event.data?.type === "PHISHGUARD_DISMISS_ADVISORY") removeAdvisory(true);
    if (event.data?.type === "PHISHGUARD_TRUST_ADVISORY") {
      void chrome.runtime.sendMessage({ type: "TRUST_CURRENT_ADVISORY" }).then((response) => {
        if (response?.ok) removeAdvisory();
        else frame?.contentWindow?.postMessage({
          type: "PHISHGUARD_TRUST_RESULT", ok: false,
          error: response?.error ?? "The website could not be added to your allowed list.",
        }, "*");
      }).catch((error) => frame?.contentWindow?.postMessage({
        type: "PHISHGUARD_TRUST_RESULT", ok: false, error: String(error?.message ?? error),
      }, "*"));
    }
  });

  chrome.runtime.onMessage.addListener((message) => {
    if (message?.type === "PHISHGUARD_NAVIGATION" && displayedUrl !== location.href) removeAdvisory();
    if (message?.type === "PHISHGUARD_PAGE_RESULT") showAdvisory(message.result, message.url);
  });

  async function requestCurrentAdvisory() {
    if (displayedUrl !== location.href) removeAdvisory();
    const requestedUrl = location.href;
    try {
      const response = await chrome.runtime.sendMessage({ type: "GET_PAGE_ADVISORY", url: requestedUrl });
      if (requestedUrl === location.href && response?.ok && response.result) showAdvisory(response.result, requestedUrl);
    } catch (_) {
      // Fail open: an unavailable advisory must never interfere with the page.
    }
  }

  void requestCurrentAdvisory();
  window.addEventListener("pageshow", () => { void requestCurrentAdvisory(); });
})();
