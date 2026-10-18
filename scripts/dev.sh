#!/usr/bin/env bash
# Run the phorse API and web dev server together. Ctrl-C stops both.
#   ./scripts/dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT

# Local dev: no invite needed -- sign up at /join/<anything>.
export HORSE_REQUIRE_INVITE="${HORSE_REQUIRE_INVITE:-0}"

echo "starting API on http://127.0.0.1:8000 ..."
python -m server.wsgi &

echo "starting retrain scheduler (HORSE_RETRAIN_INTERVAL=${HORSE_RETRAIN_INTERVAL:-600}s) ..."
python -m server.scheduler &

echo "starting web on http://localhost:5173 ..."
npm --prefix web run dev &

wait
