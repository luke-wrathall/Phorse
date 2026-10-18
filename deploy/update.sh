#!/usr/bin/env bash
# Pull the latest code, install dependencies, rebuild the site, restart.
# Run on the server as root:   sudo bash /srv/phorse/deploy/update.sh
#
# Also used for the first install. Data under
# server/data/ and horse/data/ is gitignored, so a pull never touches it.
set -euo pipefail

APP=/srv/phorse
as_phorse() { sudo -u phorse -H bash -c "cd $APP && $1"; }

echo "== pulling code"
as_phorse "git pull --ff-only"

echo "== python dependencies"
[ -d "$APP/.venv" ] || as_phorse "python3 -m venv .venv"
# CPU-only torch first: the default PyPI build bundles ~3 GB of CUDA
# libraries this server has no GPU for. The second install then sees torch
# already satisfied and leaves it alone.
as_phorse ".venv/bin/pip install -q --upgrade pip"
as_phorse ".venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision"
as_phorse ".venv/bin/pip install -q -r requirements.txt -r server/requirements.txt"

echo "== building the site"
# Empty API base = same origin: the site calls /api/... on its own domain.
as_phorse "cd web && npm ci --no-audit --no-fund && PUBLIC_API_BASE= npm run build"

echo "== restarting"
systemctl restart phorse-api phorse-scheduler
systemctl --no-pager --lines=0 status phorse-api phorse-scheduler
echo "done."
