#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Interactive explainer: http://127.0.0.1:8765"
echo "Press Ctrl+C to stop."
python3 -m http.server 8765 --bind 127.0.0.1

