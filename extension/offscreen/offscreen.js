const engine = new PhishingInferenceEngine();

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen" || message?.type !== "RUN_INFERENCE") return false;
  if (_sender.id !== chrome.runtime.id || _sender.tab) return false;
  engine.scoreUrl(message.url, { allowUnavailable: message.allowUnavailable === true }).then(
    (result) => sendResponse({ ok: true, result }),
    (error) => sendResponse({ ok: false, error: String(error?.message ?? error) }),
  );
  return true;
});
