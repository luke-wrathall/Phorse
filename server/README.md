# server/ — Horse web API

JSON-only Flask API. No HTML (CLAUDE.md "Web app architecture" — the
frontend is the separate SvelteKit app in `web/`).

## Run

```bash
pip install -r server/requirements.txt          # web deps; model deps come from repo-root requirements.txt
export HORSE_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
python -m server.wsgi                            # http://127.0.0.1:8000
```

`HORSE_SECRET_KEY` is optional for `python -m server.wsgi` (localhost-only
debug server, falls back to a dev secret) but required — 32+ characters —
when a WSGI server such as gunicorn imports `server.wsgi:app`; it refuses to
start without one (`security.check_secret_key`).

First request that touches the model loads CLIP + the head checkpoint
(`horse/data/head.pt`); if that file is missing an untrained head is used
and responses carry `head_loaded: false`. The head is hot-reloaded when
the checkpoint changes on disk (an mtime check in `score_embedding`), so
the scheduler process's retrains take effect without an API restart.

Run the retrain loop alongside it:

```bash
python -m server.scheduler          # every HORSE_RETRAIN_INTERVAL s (default 600; 0 disables)
python -m server.retrain            # or a single cycle, on demand
```

`HORSE_FAKE_EMBEDDER=1` swaps CLIP for a deterministic hash-based stand-in
(`horse.embed.FakeEmbedder`) — no model download; used by the tests.

## Layout

| path | role |
|---|---|
| `__init__.py` | `create_app()` factory — config, CORS, blueprints, error handlers |
| `config.py` | env-driven config (`HORSE_*`) |
| `db.py` | sqlite connection-per-request + schema (`users`, `images`, `ratings`) |
| `security.py` | password hashing + JWT + `@require_auth` |
| `errors.py` | `ApiError` → JSON |
| `blueprints/` | HTTP layer only — parse request, call a service, jsonify |
| `services/` | business logic; `services/model.py` holds the CLIP/head singletons; `services/retrain.py` is one retrain cycle |
| `scheduler.py` | `python -m server.scheduler` — automatic retrain loop (separate process) |
| `retrain.py` | `python -m server.retrain` — one retrain cycle on demand |

## Endpoints

| method | path | auth | body → result |
|---|---|---|---|
| GET | `/api/health` | — | `{status}` |
| POST | `/api/auth/register` | — | `{username,password}` → `{token,user}` |
| POST | `/api/auth/login` | — | `{username,password}` → `{token,user}` |
| GET | `/api/auth/me` | Bearer | → `{user}` |
| POST | `/api/images` | Bearer | multipart `image` (+ optional `caption`) → `{id,score,passed,threshold,head_loaded}` |
| GET | `/api/images/mine` | Bearer | → `{items}` — the caller's uploads, passed or not |
| GET | `/api/images/<id>/file` | — | → image bytes |
| GET | `/api/feed` | optional | `?limit=&offset=` → `{items,next_offset}` — ranked by `HORSE_FEED_STRATEGY` (blend / contested / consensus / top, see `services/feed.py`); a token lets contested/consensus mix in horses you haven't rated |
| POST | `/api/images/<id>/ratings` | Bearer | `{rating:0..5}` → `{ok,rating_count,rating_mean}` |

## Databases

Two, on purpose:

- **`server/data/app.db`** (here) — users, uploaded-image metadata +
  cached embeddings, per-user ratings.
- **`horse/data/ratings.db`** (`horse.storage`) — the training store
  `horse.train` reads.

Each retrain cycle syncs app ratings (+ embeddings) into the training
store via `horse.ingest`, then retrains, reloads, and re-scores every
image (`images.score`). The gate decision (`images.passed`) is set once at
upload and not re-evaluated.

## Not done yet

No rate limiting, no image size/type limits beyond "Pillow can open it",
no refresh tokens, `HS256` shared secret only, no account recovery. The
retrain knobs (`LAMBDA_ANCHOR`, `LAMBDA_DECAY`, `COMMUNITY_WEIGHT_FLOOR`,
`FEED_MIX`) are untuned defaults. See `nubbins.md`.
