"""Validate the frozen assets, run regression tests and build an install-only ZIP."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
original = root.parent / "browser_extension_google_workspace_v0_5_7"
destination = root.parent / "releases"
destination.mkdir(exist_ok=True)
version = json.loads((root / "manifest.json").read_text())["version"]
evidence = {"version": version, "base": original.name, "frozen_asset_sha256": {}}
frozen_model_assets = [
    str(p.relative_to(root)) for p in sorted((root / "models").rglob("*"))
    if p.is_file() and p.name not in {"policy.json", "bundle_manifest.json"}
]
for relative in frozen_model_assets:
    data = (root / relative).read_bytes()
    assert data == (original / relative).read_bytes(), f"Frozen asset changed: {relative}"
    evidence["frozen_asset_sha256"][relative] = hashlib.sha256(data).hexdigest()
evidence["explanation_runtime_sha256"] = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in ["preprocessing.js", "inference.js", "localization.js"]
}

policy_path = root / "models" / "policy.json"
manifest_path = root / "models" / "bundle_manifest.json"
policy = json.loads(policy_path.read_text())
bundle_manifest = json.loads(manifest_path.read_text())
policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
expected_threshold = 0.9707667827606201
expected_block_threshold = 0.9998490214347839
assert policy["primary_binary_rule"]["threshold"] == expected_threshold
assert policy["tiers"][2]["name"] == "caution"
assert policy["tiers"][2]["lower_inclusive"] == expected_threshold
assert policy["tiers"][-1]["name"] == "block"
assert policy["tiers"][-1]["lower_inclusive"] == expected_block_threshold
assert bundle_manifest["portability_policy"]["browser_primary_threshold"] == expected_threshold
assert bundle_manifest["portability_policy"]["browser_policy_sha256"] == policy_sha256
assert bundle_manifest["preprocessing"]["policy_sha256"] == policy_sha256
evidence["recall_policy"] = {
    "policy_sha256": policy_sha256,
    "primary_threshold": expected_threshold,
    "automatic_block_threshold": expected_block_threshold,
    "selection_validation_fpr_cap": 0.01,
    "test_labels_used_for_selection": False,
}

checks = subprocess.run(["npm", "test"], cwd=root, text=True, capture_output=True)
evidence["regression_exit_code"] = checks.returncode
evidence["regression_output"] = checks.stdout + checks.stderr
print(evidence["regression_output"])
assert checks.returncode == 0, "Regression tests failed; no release built."
if "--browser" in sys.argv:
    smoke = subprocess.run([sys.executable, "tests/browser_smoke.py"], cwd=root, text=True, capture_output=True)
    evidence["browser_exit_code"] = smoke.returncode
    evidence["browser_output"] = smoke.stdout + smoke.stderr
    print(evidence["browser_output"])
    assert smoke.returncode == 0, "Browser smoke test failed; no release built."
    navigation = subprocess.run([sys.executable, "tests/browser_navigation.py"], cwd=root, text=True, capture_output=True)
    print(navigation.stdout + navigation.stderr)
    assert navigation.returncode == 0, "Actual browser navigation tests failed; no release built."
    evidence["browser_navigation"] = json.loads((root / "tests/browser_navigation_results.json").read_text())

archive = destination / f"Phishing_URL_Guard_v{version}.zip"
evidence["runtime_sha256"] = {}
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if not item.is_file() or any(part in {"node_modules", "tests", "__pycache__"} for part in relative.parts):
            continue
        if item.name in {"package.json", "package-lock.json", "RELEASE_NOTES_v0.5.1.md", "RELEASE_NOTES_v0.5.3.md", "RELEASE_NOTES_v0.5.5.md"}:
            continue
        bundle.write(item, str(relative))
        evidence["runtime_sha256"][str(relative)] = hashlib.sha256(item.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert json.loads(bundle.read("manifest.json"))["version"] == version
evidence["zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
evidence["zip_bytes"] = archive.stat().st_size
(destination / f"v{version}_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(f"Release: {archive} ({archive.stat().st_size:,} bytes)")
