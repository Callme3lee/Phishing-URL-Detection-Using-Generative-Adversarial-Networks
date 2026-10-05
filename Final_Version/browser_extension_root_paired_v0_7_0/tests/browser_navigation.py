"""Exercise real Chrome navigation/content scripts with locally fulfilled pages.
All test-tab HTTP requests are intercepted; no test destination is contacted.
This checks extension behaviour, not real-site content or phishing accuracy.
"""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import websocket

ROOT = Path(__file__).resolve().parents[1]
BLOCK_URL = "https://accounts.example.com:8443/verify/session?id=12345&next=wallet"
HTML = base64.b64encode(b"<!doctype html><html><head><title>Local navigation fixture</title></head><body><h1>Local navigation fixture</h1></body></html>").decode()

class CDP:
    def __init__(self, address):
        self.ws = websocket.create_connection(address, origin="http://localhost", timeout=35)
        self.serial = 0
        self.interceptions = 0

    def handle(self, item):
        if item.get("method") != "Fetch.requestPaused":
            return
        self.interceptions += 1
        request = item["params"]
        self.serial += 1
        params = {"requestId": request["requestId"], "responseCode": 200,
                  "responseHeaders": [{"name": "Content-Type", "value": "text/html"}], "body": HTML}
        if "local_redirect=1" in request["request"]["url"]:
            params = {"requestId": request["requestId"], "responseCode": 302,
                      "responseHeaders": [{"name": "Location", "value": BLOCK_URL}]}
        self.ws.send(json.dumps({"id": self.serial, "method": "Fetch.fulfillRequest", "params": params}))

    def call(self, method, params=None):
        self.serial += 1
        request_id = self.serial
        self.ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            item = json.loads(self.ws.recv())
            self.handle(item)
            if item.get("id") == request_id:
                assert "error" not in item, item
                return item.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        assert "exceptionDetails" not in result, result
        return result["result"].get("value")

profile = Path(tempfile.mkdtemp(prefix="urlguard-v056-navigation-"))
log = (profile / "chrome.log").open("w")
chrome = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
    "--no-first-run", "--disable-background-networking", "--remote-debugging-port=0",
    "--enable-unsafe-extension-debugging", "--remote-allow-origins=http://localhost",
    f"--user-data-dir={profile}", "about:blank"], stdout=log, stderr=log)
connections = []
try:
    deadline = time.monotonic() + 20
    while not (profile / "DevToolsActivePort").exists():
        assert chrome.poll() is None and time.monotonic() < deadline, str(profile)
        time.sleep(.1)
    port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
    base = f"http://127.0.0.1:{port}"
    browser = CDP(json.load(urllib.request.urlopen(base + "/json/version"))["webSocketDebuggerUrl"])
    connections.append(browser)
    extension = browser.call("Extensions.loadUnpacked", {"path": str(ROOT)})["id"]

    def target(url):
        tid = browser.call("Target.createTarget", {"url": url})["targetId"]
        items = json.load(urllib.request.urlopen(base + "/json"))
        page = CDP(next(x for x in items if x["id"] == tid)["webSocketDebuggerUrl"])
        connections.append(page)
        return page

    control = target(f"chrome-extension://{extension}/pilot/pilot.html")
    deadline = time.monotonic() + 10
    while not control.evaluate("Boolean(globalThis.chrome?.runtime?.getManifest)"):
        assert time.monotonic() < deadline
        time.sleep(.1)
    assert control.evaluate("chrome.runtime.getManifest().version") == "0.7.0"
    def message(data):
        reply = control.evaluate("chrome.runtime.sendMessage(" + json.dumps(data) + ")")
        assert reply["ok"], reply
        return reply
    message({"type": "SET_PILOT_PARTICIPATION", "enabled": True})
    page = target("about:blank")
    page.call("Page.enable")
    page.call("Fetch.enable", {"patterns": [{"urlPattern": "http://*/*"}, {"urlPattern": "https://*/*"}]})
    records = []

    def wait_result(url, condition):
        deadline = time.monotonic() + 25
        state = result = None
        while time.monotonic() < deadline:
            # Evaluating also drains and fulfils paused HTTP requests on this tab.
            state = page.evaluate("({url:location.href,loaded:document.readyState,fixture:document.title==='Local navigation fixture',banner:Boolean(document.getElementById('__phishguard-caution-frame'))})")
            result = message({"type": "GET_STATUS"}).get("lastResult")
            if result and result.get("rawUrl") == url and condition(state, result):
                # Allow queued content notifications to settle before the final check.
                time.sleep(.15)
                state = page.evaluate("({url:location.href,loaded:document.readyState,fixture:document.title==='Local navigation fixture',banner:Boolean(document.getElementById('__phishguard-caution-frame'))})")
                if condition(state, result):
                    return state, result
            time.sleep(.05)
        raise AssertionError({"url": url[:160], "page": state, "result": result})

    urls = [
        "https://www.google.com/search?q=login+password&token=test",
        "https://accounts.google.com/o/oauth2/v2/auth?client_id=sample&state=test",
        "https://docs.google.com/", "https://drive.google.com/", "https://sites.google.com/",
        "https://chatgpt.com/", "https://chatgpt.com/c/local-test?model=sample",
        "https://auth.openai.com/log-in?state=test", "https://auth0.openai.com/authorize?state=test",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?state=test",
        "https://www.facebook.com/login/?next=%2Fhome.php",
        "https://www.youtube.com/watch?v=local-fixture",
        "https://www.apple.com/shop/buy-iphone?model=test",
        "https://www.instagram.com/accounts/login/?next=%2Fdirect%2F",
        "https://x.com/home?lang=en", "https://www.linkedin.com/login?fromSignIn=true",
        "https://www.amazon.com/ap/signin?openid.return_to=test",
        "https://github.com/login?return_to=%2Fsettings",
        "https://www.bing.com/search?q=account+login",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://web.whatsapp.com/?lang=en", "https://login.yahoo.com/?done=test",
        "https://www.reddit.com/login/?dest=%2F",
        "https://accounts.google.com/login?state=" + "x" * 5000,
    ]
    for url in urls:
        page.call("Page.navigate", {"url": url})
        state, result = wait_result(url, lambda s, r: s["url"] == url and s["fixture"] and not s["banner"] and r["decision"].get("reason") == "recognized_service")
        badge = control.evaluate("chrome.action.getBadgeText({tabId:" + str(result["tabId"]) + "})")
        assert badge == "", badge
        records.append({"case": "service page", "url": url[:160], "full_tier": result["tier"],
                        "effective_tier": result["decision"]["effectiveTier"], "banner": state["banner"], "badge": badge})

    shared_urls = [
        "https://docs.google.com/document/d/local-test/edit?tab=t.0",
        "https://docs.google.com/forms/d/e/local-test/viewform?usp=sharing",
        "https://docs.google.com/presentation/d/e/local-test/pub",
        "https://docs.google.com/drawings/d/local-test/edit",
        "https://drive.google.com/drive/folders/local-test?usp=drive_link",
        "https://drive.google.com/file/d/local-test/view?usp=sharing",
        "https://sites.google.com/view/local-test",
    ]
    for url in shared_urls:
        page.call("Page.navigate", {"url": url})
        collaborative = "/document/d/" in url or "/drawings/d/" in url
        def shared_ready(state, result):
            decision = result["decision"]
            if collaborative:
                return (state["url"] == url and state["fixture"] and not decision["interrupt"]
                        and decision.get("reason") in {
                            "reviewed_google_document", "reviewed_google_document_query_noise"})
            expected_interrupt = result["tier"] in {"warning", "block"}
            page_matches = ("/warning/warning.html" in state["url"]
                            if expected_interrupt else state["url"] == url and state["fixture"])
            return (page_matches and decision["interrupt"] == expected_interrupt
                    and decision.get("reason") == "reviewed_user_content")
        state, result = wait_result(url, shared_ready)
        records.append({"case": "shared service content", "url": url[:160], "full_tier": result["tier"],
                        "effective_tier": result["decision"]["effectiveTier"], "banner": state["banner"]})

    # The very same model still warns on an excluded destination. Passive mode
    # must produce an actual iframe; normal mode must navigate to the warning.
    message({"type": "SET_MODE", "mode": "passive"})
    page.call("Page.navigate", {"url": BLOCK_URL})
    state, result = wait_result(BLOCK_URL, lambda s, r: s["banner"] and r["tier"] in {"warning", "block"})
    records.append({"case": "excluded host passive", "banner": state["banner"], "tier": result["tier"]})

    # Going back to a supported service must remove the actual warning UI.
    url = "https://chatgpt.com/c/return-from-warning"
    page.call("Page.navigate", {"url": url})
    state, result = wait_result(url, lambda s, r: s["url"] == url and s["fixture"] and not s["banner"])
    # SPA history changes must receive a fresh decision, including excluded paths.
    url = "https://chatgpt.com/share/local-fixture?account=verify&token=sample"
    page.evaluate("history.pushState({},''," + json.dumps(url) + ")")
    state, result = wait_result(url, lambda s, r: r["decision"].get("reason") != "recognized_service")
    records.append({"case": "SPA shared path excluded", "tier": result["tier"], "service_rule": result["decision"].get("serviceRule")})
    url = "https://chatgpt.com/c/return-after-spa"
    page.evaluate("history.pushState({},''," + json.dumps(url) + ")")
    state, result = wait_result(url, lambda s, r: not s["banner"] and r["decision"].get("reason") == "recognized_service")
    message({"type": "SET_MODE", "mode": "normal"})
    # A real HTTP 302 delivered locally from a recognised login page cannot
    # transfer service treatment to its destination.
    page.call("Page.navigate", {"url": "https://accounts.google.com/login?local_redirect=1"})
    state, result = wait_result(BLOCK_URL, lambda s, r: "/warning/warning.html" in s["url"] and r["decision"]["interrupt"])
    explanation_ui = page.evaluate("""({
      title: document.querySelector('.signals-card > strong')?.textContent,
      signals: [...document.querySelectorAll('#signals li')].map(item => item.textContent)
    })""")
    assert explanation_ui["title"] == "Signals found in this URL"
    assert len(explanation_ui["signals"]) >= 1
    allow_controls = page.evaluate("""(() => {
      const button = document.getElementById('trust');
      const continueLabel = document.getElementById('continue').textContent;
      return {
        permanentAllowLabel: button.textContent,
        continueLabel,
        visible: button.getClientRects().length > 0,
        hasAdvancedMenu: Boolean(document.querySelector('details.advanced')),
        hasDialog: Boolean(document.querySelector('dialog'))
      };
    })()""")
    assert allow_controls["visible"]
    assert "Always allow" in allow_controls["permanentAllowLabel"]
    assert "accounts.example.com" in allow_controls["permanentAllowLabel"]
    assert "do not allow" in allow_controls["continueLabel"]
    assert not allow_controls["hasAdvancedMenu"] and not allow_controls["hasDialog"]
    page.evaluate("document.getElementById('trust').click(); true")
    trusted_state, trusted_result = wait_result(
        BLOCK_URL,
        lambda s, r: s["url"] == BLOCK_URL and s["fixture"]
        and r["decision"]["effectiveTier"] == "trusted" and not s["banner"])
    records.append({"case": "login redirects to excluded host", "warning_page": True,
                    "tier": result["tier"], "explanation": explanation_ui,
                    "direct_allow": allow_controls,
                    "after_allow": trusted_result["decision"]["effectiveTier"]})
    study = message({"type": "GET_PILOT_STATUS"})
    assert study["study"]["schemaVersion"] == 7
    assert study["summary"]["servicePageDecisions"] >= len(urls)
    assert study["summary"]["navigationInterrupts"] >= 1
    assert study["summary"]["advisoryDisplays"] >= 1
    assert study["summary"]["falseAlarmReports"] == 1
    assert study["study"]["feedbackReports"][-1]["source"] == "allowed_website"
    assert len(study["study"]["feedbackReports"][-1]["explanationSignals"]) >= 1
    output = {"passed": True, "version": "0.7.0", "method": "Real Chrome + local HTTP response fixtures; no destination requests sent",
              "cases": records, "intercepted_requests": page.interceptions, "summary": study["summary"]}
    (ROOT / "tests" / "browser_navigation_results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
finally:
    for connection in connections:
        connection.ws.close()
    chrome.terminate()
    try:
        chrome.wait(timeout=5)
    except subprocess.TimeoutExpired:
        chrome.kill()
        chrome.wait()
    log.close()
