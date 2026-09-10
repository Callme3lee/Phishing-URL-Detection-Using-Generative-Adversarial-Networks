#!/usr/bin/env python3
"""Fail when a public repository contains oversized or excluded artifacts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 100 * 1024 * 1024
BLOCKED_SUFFIXES = {".joblib", ".npy", ".sqlite", ".pt", ".pth", ".ckpt"}
BLOCKED_DIRS = {"node_modules", "__pycache__", ".pytest_cache", "DIscard"}
SKIPPED_DIRS = {".git"}
BLOCKED_NAMES = {"predictions.csv", "false_positives.csv", "false_negatives.csv"}
ALLOWED_EVIDENCE_FIXTURES = {
    Path("extension/tests/browser_onnx_calibration_predictions.npz"),
}


def main() -> int:
    problems: list[str] = []
    files = 0
    total_bytes = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        if any(part in BLOCKED_DIRS for part in relative.parts):
            problems.append(f"excluded directory: {relative}")
            continue
        if not path.is_file():
            continue
        files += 1
        size = path.stat().st_size
        total_bytes += size
        if size >= MAX_FILE_BYTES:
            problems.append(f"file is at least 100 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"excluded artifact type: {relative}")
        if (
            relative not in ALLOWED_EVIDENCE_FIXTURES
            and (path.name.lower() in BLOCKED_NAMES or "predictions" in path.name.lower())
        ):
            problems.append(f"row-level prediction file: {relative}")

    print(f"Scanned {files} files ({total_bytes / 1024 / 1024:.1f} MiB).")
    if problems:
        for problem in sorted(set(problems)):
            print(f"ERROR: {problem}")
        return 1
    print("Repository size and artifact-boundary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
