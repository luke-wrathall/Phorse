#!/usr/bin/env bash
# Run the phorse API and web dev server together. Ctrl-C stops both.
#   ./scripts/dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT

echo "starting API on http://127.0.0.1:8000 ..."
python -m server.wsgi &

echo "starting web on http://localhost:5173 ..."
npm --prefix web run dev &

wait
