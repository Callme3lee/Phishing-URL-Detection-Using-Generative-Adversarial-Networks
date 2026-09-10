#!/usr/bin/env python3
"""Verify externally restored project datasets against the public manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "artifacts_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    for artifact in payload["artifacts"]:
        print(f"[{artifact['id']}]")
        observed_valid_copy = False
        for relative in artifact["required_paths"]:
            path = ROOT / relative
            if not path.exists():
                print(f"  MISSING {relative}")
                failures += 1
                continue
            size_ok = path.stat().st_size == artifact["size_bytes"]
            digest = sha256(path)
            digest_ok = digest == artifact["sha256"]
            observed_valid_copy |= size_ok and digest_ok
            print(
                f"  {'OK' if size_ok and digest_ok else 'INVALID'} {relative} "
                f"size={path.stat().st_size} sha256={digest}"
            )
            failures += int(not (size_ok and digest_ok))
        if not observed_valid_copy and artifact.get("download_url"):
            print(f"  Download: {artifact['download_url']}")
    if failures:
        print(f"Artifact verification finished with {failures} missing or invalid path(s).")
        return 1
    print("All required artifact paths match the frozen manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
