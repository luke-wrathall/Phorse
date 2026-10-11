# phorse

pictures of horses, as decided by everyone.

An image platform where every upload has to pass a classifier that says
"this is a horse" — and the classifier's idea of "horse" drifts over time
with community ratings. See `CLAUDE.md` for the full design.

## Pieces

| dir | what |
|---|---|
| `horse/` | the Stage 2 classifier — CLIP embed → trainable head → score/gate, the anchor+decay training, and `ingest` (app ratings → training store) |
| `server/` | Flask JSON API (`server/README.md`) — accounts, uploads, feed, ratings — plus `server.scheduler`, the automatic retrain loop |
| `web/` | SvelteKit frontend (`web/README.md`) — the actual site |
| `web/prototypes/` | static HTML mockups the UI was built from (reference only) |

**The drift loop** (what makes this more than a static classifier): people
rate images in the feed → `server.scheduler` retrains the head every
`HORSE_RETRAIN_INTERVAL` seconds (ingest ratings → train → hot-reload →
re-score every image) → the feed shifts. The anchor set keeps "horse" from
collapsing however far ratings drift.

## Run it locally

One-time setup:

```bash
pip install -r requirements.txt -r server/requirements.txt
npm --prefix web install
cp web/.env.example web/.env          # PUBLIC_API_BASE=http://localhost:8000
```

Then start everything together (API + web + retrain scheduler):

```bash
pwsh scripts/dev.ps1      # Windows
./scripts/dev.sh          # macOS / Linux / Git Bash
```

or in separate terminals:

```bash
python -m server.wsgi                 # API       → http://127.0.0.1:8000
python -m server.scheduler            # retrain loop (HORSE_RETRAIN_INTERVAL, default 600s)
npm --prefix web run dev              # web       → http://localhost:5173
```

Set `HORSE_RETRAIN_INTERVAL=30` for a fast loop while testing drift, or
`0` to disable the scheduler. `python -m server.retrain` runs one cycle on
demand. `HORSE_FAKE_EMBEDDER=1` swaps CLIP for a deterministic stand-in
(no model download) — handy for a quick local spin, not for real scoring.

> **Windows:** if PowerShell says `npm.ps1 cannot be loaded because running
> scripts is disabled`, either use `npm.cmd` in place of `npm`, or allow
> local scripts once with
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

The API uses insecure dev defaults unless you set `HORSE_SECRET_KEY` (and
the other `HORSE_*` vars — see `server/config.py`). The dev JWT secret only
works for `python -m server.wsgi` (localhost debug server); served by
gunicorn (`server.wsgi:app`) the API refuses to start without a real
`HORSE_SECRET_KEY` of 32+ characters. Uploads are capped at
`HORSE_MAX_UPLOAD_MB` (20) and `HORSE_MAX_UPLOAD_PIXELS` (40 million). It loads the trained
head from `horse/data/head.pt`; if that file is missing, uploads are
scored by an untrained head and nothing passes the gate — the scheduler
writes one on its first cycle (which also builds the anchor cache from
CIFAR-10 + PACS on first run, a one-time ~350 MB download).

## Data

Both live under gitignored dirs:

- `server/data/app.db` — accounts, uploaded-image metadata + embeddings, ratings
- `server/data/uploads/` — the image files
- `horse/data/` — the head checkpoint, anchor cache, training ratings db

Delete `server/data/` to start clean.
