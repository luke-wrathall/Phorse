# phorse

pictures of horses, as decided by everyone.

An image platform where every upload has to pass a classifier that says
"this is a horse" — and the classifier's idea of "horse" drifts over time
with community ratings. See `CLAUDE.md` for the full design.

## Pieces

| dir | what |
|---|---|
| `horse/` | the Stage 2 classifier — CLIP embed → trainable head → score/gate, plus the anchor+decay retrain loop |
| `server/` | Flask JSON API (`server/README.md`) — accounts, uploads, feed, ratings |
| `web/` | SvelteKit frontend (`web/README.md`) — the actual site |
| `web/prototypes/` | static HTML mockups the UI was built from (reference only) |

## Run it locally

One-time setup:

```bash
pip install -r requirements.txt -r server/requirements.txt
npm --prefix web install
cp web/.env.example web/.env          # PUBLIC_API_BASE=http://localhost:8000
```

Then start both servers together:

```bash
pwsh scripts/dev.ps1      # Windows
./scripts/dev.sh          # macOS / Linux / Git Bash
```

or in two terminals:

```bash
python -m server.wsgi                 # API  → http://127.0.0.1:8000
npm --prefix web run dev              # web  → http://localhost:5173
```

> **Windows:** if PowerShell says `npm.ps1 cannot be loaded because running
> scripts is disabled`, either use `npm.cmd` in place of `npm`, or allow
> local scripts once with
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

The API uses insecure dev defaults unless you set `HORSE_SECRET_KEY` (and
the other `HORSE_*` vars — see `server/config.py`). It loads the trained
head from `horse/data/head.pt`; if that file is missing, uploads are
scored by an untrained head and nothing passes the gate.

## Data

Both live under gitignored dirs:

- `server/data/app.db` — accounts, uploaded-image metadata + embeddings, ratings
- `server/data/uploads/` — the image files
- `horse/data/` — the head checkpoint, anchor cache, training ratings db

Delete `server/data/` to start clean.
