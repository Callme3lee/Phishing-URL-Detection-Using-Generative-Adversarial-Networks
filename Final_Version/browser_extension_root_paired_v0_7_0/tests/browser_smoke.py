"""Real Chrome smoke test in a disposable profile; never visits phishing pages.
Requires google-chrome and websocket-client. Leaves a temporary log for diagnosis.
"""
import json
import pathlib
import subprocess
import tempfile
import time
import urllib.request
import websocket

root = pathlib.Path(__file__).resolve().parents[1]
profile = pathlib.Path(tempfile.mkdtemp(prefix="urlguard-v053-chrome-"))
log = (profile / "chrome.log").open("w")
process = subprocess.Popen([
    "google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
    "--disable-features=DisableLoadExtensionCommandLineSwitch",
    "--no-first-run", "--disable-background-networking", "--remote-debugging-port=0", "--enable-unsafe-extension-debugging",
    "--remote-allow-origins=http://localhost", f"--user-data-dir={profile}",
    "about:blank",
], stdout=log, stderr=log)

try:
    deadline = time.monotonic() + 30
    targets = []
    pages = []
    installed = False
    while time.monotonic() < deadline:
        portfile = profile / "DevToolsActivePort"
        if portfile.exists():
            port = portfile.read_text().splitlines()[0]
            if not installed:
                version_info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version"))
                browser = websocket.create_connection(version_info["webSocketDebuggerUrl"], origin="http://localhost", timeout=10)
                browser.send(json.dumps({"id": 1, "method": "Extensions.loadUnpacked", "params": {"path": str(root)}}))
                while True:
                    reply = json.loads(browser.recv())
                    if reply.get("id") == 1:
                        assert "error" not in reply, reply
                        extension_id = reply["result"]["id"]
                        break
                browser.send(json.dumps({"id": 2, "method": "Target.createTarget", "params": {
                    "url": f"chrome-extension://{extension_id}/pilot/pilot.html"}}))
                while json.loads(browser.recv()).get("id") != 2:
                    pass
                browser.close()
                installed = True
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
            pages = [t for t in targets if t["type"] == "page" and "/pilot/pilot.html" in t["url"]]
            if pages:
                break
        if process.poll() is not None:
            raise RuntimeError(f"Chrome exited. See {profile}/chrome.log")
        time.sleep(0.2)
    assert pages, f"Extension page not loaded. Targets: {targets}. Log: {profile}/chrome.log"
    ws = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], origin="http://localhost", timeout=45)
    serial = 0

    def evaluate(expression):
        global serial
        serial += 1
        ws.send(json.dumps({"id": serial, "method": "Runtime.evaluate", "params": {
            "expression": expression, "awaitPromise": True, "returnByValue": True}}))
        while True:
            reply = json.loads(ws.recv())
            if reply.get("id") == serial:
                assert "exceptionDetails" not in reply.get("result", {}), reply
                return reply["result"]["result"].get("value")

    deadline = time.monotonic() + 10
    while not evaluate("Boolean(globalThis.chrome?.runtime?.getManifest)"):
        assert time.monotonic() < deadline, evaluate("({url:location.href,body:document.body?.innerText})")
        time.sleep(0.1)
    version = evaluate("chrome.runtime.getManifest().version")
    assert version == "0.7.0", version
    status = evaluate("chrome.runtime.sendMessage({type:'GET_PILOT_STATUS'})")
    assert status["ok"] and not status["enabled"], status
    assert evaluate("chrome.storage.local.set({uiLanguage:'ar'}).then(() => true)")
    evaluate("location.reload(); true")
    deadline = time.monotonic() + 10
    # Runtime.evaluate can briefly run in the old, cleared document while the
    # reload commits. Optional chaining keeps that normal transition from
    # becoming a false test failure.
    while evaluate("document.documentElement?.dir ?? ''") != "rtl":
        assert time.monotonic() < deadline
        time.sleep(.1)
    assert "نشاط الاكتشاف" in evaluate("document.body.innerText")
    assert evaluate("document.querySelector('[data-language-toggle]').textContent") == "English"
    score = evaluate("chrome.runtime.sendMessage({type:'SCORE_URL',url:'https://example.com/login?test=local'})")
    assert score["ok"] and isinstance(score["result"]["score"], (int, float)), score
    assert score["result"]["explanation"]["method"] == "hybrid_input_diagnostics_v1"
    assert isinstance(score["result"]["explanation"]["signals"], list)
    # The URL string is scored locally; this test does not navigate to it.
    assert evaluate("chrome.runtime.sendMessage({type:'SET_MODE',mode:'passive'})")["ok"]
    assert evaluate("chrome.runtime.sendMessage({type:'GET_MODE'})")["mode"] == "passive"
    print(json.dumps({"passed": True, "chrome_extension_version": version,
                      "checks": ["extension load", "consent off by default", "Arabic RTL localization", "language switch", "real offscreen ONNX inference", "diagnostic explanation", "mode persistence"],
                      "score": score["result"]["score"], "temporary_profile": str(profile)}, indent=2))
    ws.close()
finally:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()
