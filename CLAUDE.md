# phorse — current state and rules

Read this first. The *reasoning* behind every decision below (experiments,
numbers, rejected alternatives, dates) is in `docs/decisions.md`; add to it
when a decision is made or reversed. `nubbins.md` (git-ignored) is the
running list of unfinished bits — keep it current.

## What this is

An Instagram-adjacent site where every upload must pass a classifier that
says "this is a horse". The classifier's idea of "horse" is not fixed: it
drifts with community ratings, so the platform's culture shapes its own
rules of entry. An art project run by one person; live as an invite-only
private beta at phorse.website (one Hetzner VM, `deploy/`).

**Status:** the drift loop is closed and runs on real CLIP in production.
What's left before calling the MVP done is tuning the knobs against real
vote data, which needs real users — so getting the site in front of people
is the current work. Known structural limits (see `docs/decisions.md`
2026-11-10 and nubbins): votes mostly memorize rated images rather than
generalize at today's volume; the loop can narrow "horse" but not widen it
(only admitted images get rated); decay doesn't pull back toward the
anchors at shipped settings.

## Layout

| dir | what |
|---|---|
| `horse/` | classifier: `embed.py` (frozen CLIP), `model.py` (head), `anchors.py`, `train.py` (loss), `storage.py` + `ingest.py` (training store), `tune.py` (offline knob sweeps), `check_anchors.py`, CLI tools `score/gate/rate.py` |
| `server/` | Flask JSON API (`blueprints/` routes, `services/` logic), `scheduler.py` (retrain loop process), `retrain.py` (one cycle), `admin.py` + `invites.py` (operator CLIs), `config.py` (every `HORSE_*` knob) |
| `web/` | SvelteKit SPA (adapter-static, `ssr = false`), one hand-written stylesheet `src/app.css` |
| `tests/` | pytest, runs under `HORSE_FAKE_EMBEDDER=1` (no CLIP download) |
| `deploy/` | Caddyfile, systemd units, `update.sh`, `backup.sh`, `manage.sh` |
| `docs/` | `decisions.md` (the log), screenshot |

Run locally: `scripts/dev.ps1` / `scripts/dev.sh` (API + web + scheduler,
invites and moderation off). Tests: `python -m pytest` (~6 min; two
failures that were known are fixed as of 2026-11-10). Frontend check:
`npm --prefix web run check`.

## Architecture

```
upload → embed (CLIP, frozen) → head → score
                                   ├─ gate: score >= GATE_THRESHOLD (2.5), decided once at upload
                                   ├─ content check (OpenAI moderation) on passing uploads only
                                   └─ feed ranking, re-scored every retrain
```

- **One head, one score** for both the gate and ranking. A separate gate
  model was rejected: the community must shape the rules of entry, not just
  the ranking. Anchors + prox are what stop collapse.
- **Stage 1 precheck** (loose, non-drifting noise filter) is not built;
  `services/safety.py` covers the moderation half of it, after the gate.
- **Gate threshold is fixed** at 2.5 for now; normalising against the score
  distribution is an open question for when real distributions exist.
- `images.passed` is **never re-evaluated** after upload (no re-gating).

## Model

`HorseHead`: CLIP ViT-B-32 embedding (512-d, L2-normalised) → 8 hidden
units → 1, regression (MSE) on a 0–5 "horse out of 5" scale. Embeddings are
computed once per image and cached (`images.embedding`); CLIP is never
re-run for training. Head size 8 was chosen from real votes: bigger heads
memorise each rated image; linear let anchors slip. Changing
`HORSE_HEAD_HIDDEN` makes the next cycle train a fresh head.

**Uploads are scored at 0/90/180/270°** (`services/images.py`): a rotation
is stored only if upright fails, the rotation passes, and it beats upright
by `UPLOAD_ROTATION_MARGIN`. The other three embeddings are kept in
`image_augments` (showing posts only) as training augmentations.

`HORSE_FAKE_EMBEDDER=1` swaps CLIP for a deterministic hash embedder.

## Training (`horse/train.py::train_head`)

```
L = λ_anchor · mean(anchor_sq_err)
  + Σ_i w_i · sq_err_i / max(Σ_i w_i, community_weight_floor)
  + prox_weight · ||θ − θ_prev||²            (warm start only)
w_i = exp(−λ_decay · age_days_i) · m_i       m_i = 1 for a vote, augment_weight for its rotations
```

Both sums pool **every rating on every image** — one platform-wide
numerator and denominator, deliberately not per-image (per-image would make
image *count* the cheap lever for gaming the head). Consequences to keep
straight:

- `λ_anchor` is a constant **ratio** between anchor pull and community
  pull, not diluted by volume.
- The floor is numerical safety near zero total weight; above it, inert.
- **Decay only re-weights votes against each other.** Uniform ageing leaves
  a weighted mean unchanged, so a quiet platform does not drift back toward
  the anchors until Σw < floor — years at the shipped half-life. Don't
  describe decay as "the anchor pulls stale images back"; it doesn't at
  production settings.
- Prox is a **speed limit, not a cap**: cycles converge to the same fixed
  point, so drift speed is per cycle and scales with the retrain interval.
- Augmentations count as fractions of a vote in numerator *and*
  denominator; they don't add leverage.

**Anchor set** (`horse/anchors.py::DEFAULT_SOURCES`): CIFAR-10 horses/
negatives, PACS (photo/painting/cartoon/sketch, matched negatives per
style), QuickDraw doodles (horse + 22 negative categories, zebra/donkey
left to the community), and an operator-curated folder. **Rule: every
style added as positives needs matched negatives in the same style**, or
the head learns "drawing = horse". Drawn horses are a first-class goal.
Check any anchor change with `python -m horse.check_anchors`, then
`python -m server.retrain --fresh`. The community never touches the anchors.

## Retrain loop (`server/services/retrain.py::run_retrain`)

`python -m server.scheduler` — a separate process, every
`HORSE_RETRAIN_INTERVAL` s (600), once on startup. One cycle: **ingest**
(app ratings + embeddings + augments → `horse/data/ratings.db`, idempotent
upsert) → **train** (warm start from `head.pt`, or `--fresh`) → **reload**
(API picks the new `head.pt` up by mtime) → **re-score** every image.
Cycles hold a file lock (`head.pt.lock`); `head.pt` is written atomically.
The scheduler only ever scores cached embeddings and must not load CLIP
(`services/model.py` loads head and embedder independently).

Ingest never deletes: `server/admin.py` mirrors deletions into the training
store by hand. Hidden posts' votes still train (nubbin).

**Tuning:** `python -m horse.tune --app-db tune/app.db --head tune/head.pt`
on a *copy* of the server's data: vote stats, model-vs-people per image,
and a knob sweep (fit, held-out fit, anchor health, unrated-image shift).
Snapshots live in git-ignored `tune/`.

### Knobs (`server/config.py`, all `HORSE_*`)

| knob | default | note |
|---|---|---|
| `LAMBDA_ANCHOR` | 0.1 | set from real data 2026-10-27 |
| `LAMBDA_DECAY` | 0.002 /day | ~350-day half-life; set from real data |
| `COMMUNITY_WEIGHT_FLOOR` | 3.0 | |
| `RETRAIN_PROX_WEIGHT` | 1.0 | |
| `RETRAIN_EPOCHS` / `RETRAIN_FRESH_EPOCHS` | 200 / 3000 | small heads need the long fresh run |
| `HEAD_HIDDEN` | 8 | set from real data |
| `AUGMENT_WEIGHT` | 0.25 | **a guess**, sweep with `horse.tune --augment-weight` |
| `GATE_THRESHOLD` | 2.5 | |
| `UPLOAD_ROTATION_MARGIN` | 0.5 | |
| `FEED_STRATEGY` | contested | blend / contested / consensus / top |
| `FEED_NEWEST_SHARE` | 0.5 | |
| `FEED_MIX` | 0.5,0.3,0.2 | blend only |
| `UPLOADS_PER_USER_DAY` / `UPLOADS_PER_IP_DAY` | 30 / 100 | |
| `SIGNUPS_PER_IP_HOUR`, `REPORTS_PER_REPORTER_DAY` | 20 / 20 | |
| `UPLOAD_BLOCKS_PER_USER_DAY`, `REPORT_AUTOHIDE` | 3 / 3 | |

Everything not marked "from real data" is a judgement call.

## Feed (`server/services/feed.py`)

Community-global, no per-user model. Two feeds, `GET /api/feed?feed=new|ranked`:

- **new** (front page): horses the viewer hasn't rated, ranked by
  `FEED_STRATEGY`. `contested` = most divisive (recency-weighted rating
  variance) interleaved with the newest; `consensus` = highest model score
  interleaved the same way; `top` = model score only; `blend` = the
  three-pool consensus/contested/exploration blend.
- **ranked** (`/ranked`, "most horse"): every horse by model score, with a
  rank, no rating UI.

"Highest" always means the **model's** score, never raw ratings — ratings
reach ranking only through the head. Mean and variance use the training
decay. Feed sessions (client id per page load, in-process memory, 1 h
TTL) keep already-served positions fixed. Loads the whole catalogue per
request (fine at MVP scale). The response exposes **no** variance / pool /
contested fields.

## Web app

Detached SPA + JSON API, same origin when deployed (Caddy routes
`/api/*`), cross-origin in dev (`PUBLIC_API_BASE`). Flask (CLIP is
CPU-bound; async buys nothing) + SvelteKit + `@tanstack/svelte-query`.
JWT in localStorage, sent as `X-Horse-Token` (`Authorization: Bearer` also
accepted). One gunicorn worker on purpose (in-memory feed sessions, one
CLIP copy).

**UI rules:** images lead, one column, rate in place with six 0–5 buttons;
deliberately plain old-web look, hand-written CSS, no framework. **Never
surface the drift mechanism** — no contested/consensus labels, variance,
or "state of horse" analytics. The emergent behaviour only works if the
app doesn't point at it.

## Access and moderation

- **Invite codes** gate signup (`python -m server.invites`, links
  `/join/<code>`); `/login` is log-in only. Feed and images need no login.
- **Reports** (`/report`, no login): `child_safety` / `intimate_image` hide
  the post at once, others at `REPORT_AUTOHIDE` distinct reporters; operator
  emailed if SMTP is set, never quoting the report.
- **Content check** (`services/safety.py`): OpenAI `omni-moderation-latest`
  on image + caption of uploads that *passed* the gate, fails closed, on by
  default. Cannot detect CSAM in images (hash matching via IWF planned, not
  built). Rejected uploads keep **no file**, only the row.
- **Admin CLI** (`python -m server.admin`, on the server via
  `deploy/manage.sh`): hide/restore/delete posts, quarantine (keeps the
  file out of serving and backups, refuses delete/purge without a flag —
  legal retention), ban/unban (`--purge`), set-password, pause
  uploads/signups, review links.
- Limits live in the db (IPs stored as keyed hashes). Invites stay on;
  opening signup fully needs the deferred work below first.

## Deferred (don't build unless asked)

Public signup hardening (dedup, email + account recovery, login rate
limiting, IWF hash matching, OSA documents), re-gating drifted content,
recursive aggregate update, vector DB, active learning, the public "state
of horse" trendline, per-user horse models (spitballed in
`docs/decisions.md`; feed-only if ever built, gate stays global).

## Working rules

- Keep the head small and retraining cheap; frozen backbone, cached
  embeddings, never re-run CLIP for training.
- Business logic in `services/`, routes stay thin. Plain sqlite3.
- Tests run under the fake embedder; real-CLIP behaviour is checked by hand
  (`horse.check_anchors`, `horse.tune`).
- When you change a knob default or a mechanism, update the table above,
  `server/config.py`'s comment, and `docs/decisions.md` with the date and
  the evidence.
- Stub or defer something → add a nubbin; finish one → strike it.
