# server/ — Horse web API

JSON-only Flask API. No HTML (CLAUDE.md "Web app architecture" — the
frontend is the separate SvelteKit app in `web/`).

## Run

```bash
pip install -r server/requirements.txt          # web deps; model deps come from repo-root requirements.txt
export HORSE_SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
python -m server.wsgi                            # http://127.0.0.1:8000
```

First request that touches the model loads CLIP + the head checkpoint
(`horse/data/head.pt`); if that file is missing an untrained head is used
and responses carry `head_loaded: false`.

## Layout

| path | role |
|---|---|
| `__init__.py` | `create_app()` factory — config, CORS, blueprints, error handlers |
| `config.py` | env-driven config (`HORSE_*`) |
| `db.py` | sqlite connection-per-request + schema (`users`, `images`, `ratings`) |
| `security.py` | password hashing + JWT + `@require_auth` |
| `errors.py` | `ApiError` → JSON |
| `blueprints/` | HTTP layer only — parse request, call a service, jsonify |
| `services/` | business logic; `services/model.py` holds the CLIP/head singletons |

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
| GET | `/api/feed` | — | `?limit=&before=` → `{items,next_before}` (items carry `caption`) |
| POST | `/api/images/<id>/ratings` | Bearer | `{rating:0..5}` → `{ok,rating_count,rating_mean}` |

## Databases

Two, on purpose:

- **`server/data/app.db`** (here) — users, uploaded-image metadata +
  cached embeddings, per-user ratings.
- **`horse/data/ratings.db`** (`horse.storage`) — the training aggregate
  store `horse.train` reads.

Nothing bridges app ratings into the training store yet — tracked nubbin.

## Not done yet

Feed is a newest-first stub, not the consensus/contested/exploration
blend. No rate limiting, no image size/type limits beyond "Pillow can open
it", no refresh tokens, `HS256` shared secret only. See `nubbins.md`.
