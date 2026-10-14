#!/usr/bin/env bash
# Nightly snapshot of everything that can't be rebuilt: the app database
# (accounts, images, ratings), the uploaded files, and head.pt (the current
# drift state). The training store and anchor cache are rebuilt
# automatically, so they're skipped. Keeps the last 14 days.
#
# Run by phorse-backup.timer. These copies sit on the same server, so they
# guard against mistakes, not against losing the server -- turn on
# DigitalOcean's droplet backups too (deploy/README.md).
set -euo pipefail

APP=/srv/phorse
# sqlite3.connect would create an empty (root-owned) app.db if it's missing.
[ -f "$APP/server/data/app.db" ] || { echo "no app.db yet, nothing to back up"; exit 0; }
DEST=/var/backups/phorse/$(date +%F)
mkdir -p "$DEST"

# sqlite's online backup API: a consistent copy even while the API and the
# scheduler are writing. A plain `cp` of a live database can be torn.
"$APP/.venv/bin/python" - "$APP/server/data/app.db" "$DEST/app.db" <<'EOF'
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
EOF

[ -d "$APP/server/data/uploads" ] && tar -czf "$DEST/uploads.tar.gz" -C "$APP/server/data" uploads
[ -f "$APP/horse/data/head.pt" ] && cp "$APP/horse/data/head.pt" "$DEST/"

find /var/backups/phorse -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
echo "backed up to $DEST"
