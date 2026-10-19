# Project: "Horse" Platform — Stage 2 Classifier

## What this project is
An Instagram-adjacent social platform where every uploaded image must pass
a naive classifier identifying it as a "horse." The core creative premise:
the platform's concept of "horse" is not fixed — it evolves as a living
microcosm shaped by community ratings, drifting over time in ways that
reflect the culture of the user base. Currently building the classifier
system, starting with Stage 2 (see architecture below).

## Current focus
**Stage 2 MVP = the running site, locally, demonstrating the full drift
loop.** Revised (was: "embed → train → score pipeline, no UI"). MVP is
done when a person can, on a local instance: make an account → upload →
get gated → see a real feed → rate images → and those ratings measurably
move the classifier over a retrain cycle, with the anchor set keeping it
from collapsing.

**The core loop is now closed** (2026-09-03):
- ✅ community → training loop — `horse/ingest.py` bridges app ratings into
  the training store; `server/scheduler.py` retrains on an interval;
  `server/services/model.py` hot-reloads the head; images are re-scored
  each cycle. See "Retrain loop (built)".
- ✅ three-pool feed — `server/services/feed.py`, community-global. See
  "Recommendation / feed algorithm".
- ✅ anchor + decay math — `train.py` loss is now weighted-means with a
  constant `λ_anchor` ratio + a community-weight floor. See "Training /
  drift mechanism".
- ✅ end-to-end proof — `tests/test_core_loop.py` (drift down under
  ratings, anchor bounds it, decay pulls it back).

Manual real-CLIP pass end to end: ✅ done 2026-10-11 (upload, gate, rate,
retrain, re-score all work on real CLIP).

Remaining before calling MVP done: tune the `λ` / floor / mix knobs
against real score distributions (none exist yet) — which needs real users
rating, so it now depends on getting the site in front of people.

**Private beta deployment (2026-10-14)** — to get real users rating: one
Hetzner VM, invited users only. Config, services and scripts are in
`deploy/` (the local step-by-step guide was deleted once deployed). One
gunicorn worker on purpose (in-memory feed sessions, one CLIP copy).
Cambridge-only sign-on was considered and declined.

**Invite codes replaced the site-wide password (2026-10-18)** so the site
can be shared with a few hundred people (e.g. a link in the operator's
private Instagram story). Signup needs a code (`server/services/invites.py`,
`HORSE_REQUIRE_INVITE`, on by default; `scripts/dev.*` turn it off). A code
can be multi-use with a cap, is revocable, and each account records the
code it used (`users.invite_code`). Managed from the CLI (`python -m
server.invites create|list|revoke`), which prints a signup link
`<SITE_URL>/join/<code>`. Sign-up lives only on that page
(`web/src/routes/join/[code]`), taking the code from the URL, so invitees
never type it; `/login` is log-in only.
The feed and image files need no login, so anyone with the URL can look;
only invitees can post or rate. Moderation is by hand from the CLI
(`python -m server.admin`: delete a post, ban a user with optional
`--purge` of their posts + votes, reset a password, which also logs them
out). Deletes reach the training store too, since ingest never deletes.
On the server both CLIs run via `deploy/manage.sh`. Hosted moderation
APIs / CSAM hash matching were considered and put off as more than this
audience needs.
Invite-only signup is what makes deferring the items below acceptable for
now; **going public needs them revisited**, moderation first.

Still deferred (NOT MVP): public hosting, upload guards +
rate limiting + dedup, account recovery / email, Stage 1 precheck,
moderation, re-gating drifted content, the recursive aggregate
optimization.

---

## System architecture (two-stage, decided)

```
Image → Stage 1: Precheck (loose, stable, non-drifting) → reject / pass
                                                  │
                                                  ▼
                              Stage 2: Unified drift head → score
                                                  │
                                    ┌─────────────┴─────────────┐
                                    ▼                             ▼
                            gate decision                  feed ranking
                          (score ≥ threshold)              (raw score + variance)
```

**Stage 1 — loose precheck (not yet built):**
Filters out content that isn't even plausibly in the space of "horse
debate" at all (noise, screenshots of text, etc.), and folds in baseline
content moderation (NSFW/gore/spam via off-the-shelf classifier). Generous
threshold — false negatives (rejecting something that could've sparked fun
contested debate) are worse than false positives. Does NOT drift — this is
infrastructure, not culture.

**Stage 2 — unified drift head (current build target):**
One model, one score, used for BOTH the gate decision and the
rating/ranking signal. This was a deliberate correction from an earlier
draft that proposed fully separate gate and rating models — decided
against, because a decoupled gate would let the community shape "culture"
but not the actual rules of entry, which undercuts the point of the
project. The anchor+decay mechanism (below) is what prevents collapse,
so a separate stability mechanism isn't needed.

Open question, not yet decided: does the Stage 2 gate **threshold** stay
fixed, or normalize against the current score distribution (moving
target) if community rating inflation/deflation occurs over time? Leaning
toward starting with a fixed threshold for MVP simplicity, revisit once
real score distributions exist.

---

## Model design

**Pretrained backbone + trainable head**, not a CNN trained from scratch.

- CLIP image encoder (frozen) as feature extractor — turns any image into
  a fixed-length embedding (e.g. 512-d).
- Small trainable head on top of the embedding (a couple linear layers,
  `512 → 8 → 1`) outputs a continuous **horseness score** — trained
  via regression (MSE), not binary classification.
- **Head size: 8 hidden units (`HORSE_HEAD_HIDDEN`, 2026-10-19; was 64).**
  Chosen for interesting drift at low traffic — the site may never get much.
  A 64- or 32-hidden head memorizes each rated image, so votes never reach
  images nobody rated (or future uploads, which is what the gate judges).
  On the real votes (`horse.tune`, 74 votes / 19 images, 5-fold held-out
  over images, out to 72 cycles): 64 and 32 never beat their starting
  head on held-out images; 16 closed ~25% of the gap within 3 cycles but
  re-memorized by 72; **8 closed ~20% and kept it**; linear (`0`) got
  worse and let the doodle-horse anchors slip to ~82%. Anchors at 8 are
  as good as at 64 when both are fully trained (held-out doodle horses
  92%, non-horse sketches 35% vs 30%). Changing the size makes the next
  cycle train a fresh head (`RETRAIN_FRESH_EPOCHS`, 3000 — small heads
  need it; 200 from scratch left a linear head at 50% anchor accuracy).
- Compute embedding once per image, cache/reuse it — don't re-run CLIP
  per head or per retrain.
- Ratings are framed as regression targets ("horse out of 5" style, or a
  binary "🐴 horse" tap as a strong-signal vote), not class labels — this
  matches the degree-of-horseness concept, not fixed category membership.

## Training / drift mechanism (anchor + decay, decided)

Combined loss per retrain cycle (`horse/train.py::train_head`):
```
L = λ_anchor * mean(anchor_sq_err)
    + Σ w(t_i) * L_rating(image_i, rating_i) / max(Σ w(t_i), community_weight_floor)
w(t) = exp(-λ_decay * Δt)
```
Both `Σ`s pool every rating on every image together into one number each —
a single platform-wide numerator and denominator, not one pair per image.
That's load-bearing, not incidental: see `community_weight_floor` below.

- **Anchor loss**: fixed seed/reference set (e.g. CLIP zero-shot "a photo
  of a horse" scores, or curated reference images) included in every
  retrain with a small but CONSTANT (non-decaying) weight `λ_anchor`.
  Prevents total collapse of the horse-concept regardless of how far
  community ratings drift — acts like a rubber band, not a wall. Operator
  (you) can lightly curate/update the anchor set manually as an escape
  hatch if drift goes somewhere undesirable — the community does not
  control the anchor set.
- **Anchor set contents (2026-10-11)**: `horse/anchors.py::DEFAULT_SOURCES`
  = CIFAR-10 (32px photos) + **PACS** (photo / art painting / cartoon /
  sketch, 100 horse + 100 style-matched non-horse per style) +
  **QuickDraw** doodles (2026-10-19: 200 horse + 200 non-horse spread over
  22 categories, mostly other four-legged animals plus car/house/tree/
  bicycle/bird; zebra and donkey deliberately left to the community) + a
  curated folder (`horse/data/anchors_curated/{horse,not_horse,<number>}/`).
  QuickDraw was added after real MS-Paint doodle horses on the site scored
  1.4-2.3 and were turned away: held-out doodle horses passing went 28% ->
  85%, at the cost of non-horse doodles 0 -> 12% and non-horse sketches 8 ->
  20%. Accepted, because the gate is asymmetric: a false positive reaches
  the feed and can be voted down, a false negative is never seen again.
  (400 negatives was tried: 70% / 8%, too many real doodles lost.)
  **Drawn horses are a first-class goal**: most users have no horse to
  photograph, so hand/computer-drawn horses must be able to pass the gate.
  Rule: every style added as positives needs **matched negatives in the
  same style**, or the head learns "drawing = horse". Check any anchor
  change with `python -m horse.check_anchors --anchor-cache old.npz
  --anchor-cache new.npz` (held-out PACS pass rates per style + scores for
  `tests/test_horse/` and stored uploads) before swapping it in; then run
  `python -m server.retrain --fresh` (trains from scratch on the new
  anchors + every vote, instead of warm-starting), since the prox pull
  would otherwise absorb new anchors only slowly.
- **Community loss**: each rating weighted by recency,
  `w(t) = exp(-λ_decay * Δt)`. `λ_decay` is the main "how fast does
  culture move" knob.
- **Both terms are means, so `λ_anchor` is a constant *ratio*** between the
  anchor pull and the community pull for a well-rated image — it is NOT
  diluted as rating volume grows. (The earlier `Σ w·L_rating` sum form let
  the anchor be swamped once an image had more than a few dozen ratings;
  fixed 2026-09-03.) Trade-off accepted: vote *volume* no longer makes the
  head fit an image harder — the feed's exploration pool handles surfacing
  under-rated content instead.
- **`community_weight_floor`** (default 3.0): floors the single,
  platform-wide denominator (`max(Σ w(t_i), community_weight_floor)`,
  summed over every rating on every image pooled together) — **not** a
  per-image floor. Its job is numerical safety at low total effective
  vote volume (e.g. right at launch, or a platform-wide quiet spell):
  without it, a near-zero total weight could make the community term's
  magnitude/gradient blow up. Above that near-zero range it's inert — a
  plain weighted mean.
  "Stale image fades back toward the anchor" is real, but it's a
  **platform-relative** effect of the pooled mean itself, not a fixed
  per-image guarantee: an old vote's weight shrinks *relative to however
  much other, fresher weighted activity exists across the whole platform
  at that moment*, not against some fixed per-image vote-count threshold.
  A stale image on an otherwise-quiet platform fades slower than the same
  stale vote would on a busy one — that's expected, not a bug.
- **Deliberately not per-image.** A per-image version (group ratings by
  image, floor each image's own weight sum, then average the per-image
  means) was considered and rejected: once an image clears the floor,
  further votes on it stop buying any more influence over the total loss
  (each image becomes one equal term in an outer mean), while a *new*
  image only needs ~floor votes to earn a full additional term of its
  own — making image *count*, not vote *count*, the cheap lever for
  swaying the shared head. The pooled version has no such asymmetry:
  spreading N fake votes across many images vs. concentrating them on one
  contributes the same total weight to the shared denominator either way,
  so an attacker's leverage is capped by their share of *all* platform
  votes, not by how many images they touch — and that share shrinks on
  its own as real platform activity grows. Revisit only alongside real
  anti-abuse work (upload dedup/rate limiting are already deferred — see
  "Current focus" — and a per-image scheme would need equally careful
  anti-gaming design of its own, e.g. an outer weight capped/diminishing
  in each image's real vote mass rather than flat-equal-per-image).
- **Why exponential decay**: constant half-life (proportional fade
  regardless of rating age, no jarring cutoffs), cheap recursive update
  (`aggregate_new = aggregate_old * exp(-λΔt) + new_ratings`, no need to
  recompute over full history each cycle), and it's the standard default
  for "constant proportional fade" processes.
- **Per-cycle proximal pull (`RETRAIN_PROX_WEIGHT`, 2026-10-11)**: each
  retrain adds `prox_weight * ||θ - θ_prev||²` toward the warm-start
  head, so a cycle *nudges* rather than refits. Found with real CLIP: the
  64-hidden head has capacity to fit the anchors AND every rated image
  exactly, so without it one vote snapped an image to its exact rating in
  one cycle — and since both can be fit at once, no `λ` ratio changes
  that fixed point. The prox term is a **speed limit, not a cap**: it
  converges to the same fixed point over ~10 cycles at 1.0, so drift speed
  is per *cycle* and scales with `RETRAIN_INTERVAL_SECONDS`. Rejected
  alternative: pull toward a *fixed* anchors-only reference head (a true
  per-image cap, but it never loosens as the platform grows — a second,
  stiffer anchor that would bound how far culture can ever move). The
  head was later shrunk to 8 hidden units for the same reason (see "Model
  design"); prox stays as the speed limit.
- Alternatives considered but not chosen for the core mechanism: linear
  decay (hard cutoff, easier to explain to users), power-law decay (old
  ratings asymptotically linger, "cultural memory" feel), step-function
  "seasons" (discrete eras rather than smooth blending — flagged as
  possibly more legible/fun, worth revisiting for UI later, could be
  layered on top of the smooth exponential math under the hood as a
  presentation-layer hybrid).

## Retrain loop (built — this is what closes the MVP loop)

Automatic, on an interval. One cycle (`server/services/retrain.py::run_retrain`):

1. **Ingest** (`horse/ingest.py`) — copy every per-user rating from the app
   db (`server/data/app.db`) into the training store
   (`horse/data/ratings.db`), carrying each image's cached CLIP embedding
   across so training never re-runs CLIP. Full idempotent sync.
2. **Train** (`horse/train.py::train_from_store`) — retrain the head from
   the anchor set + those ratings, warm-started from the current `head.pt`.
3. **Reload** — `server/services/model.py` swaps the new checkpoint in.
   The API process (separate from the scheduler) picks it up via an mtime
   check on `head.pt` in `score_embedding`.
4. **Re-score** — recompute `images.score` for every stored image from its
   cached embedding, so the feed/ranking reflects the drift. The gate
   decision (`images.passed`) is **NOT** re-evaluated — admission is a
   point-in-time event. **Open question: should drift ever retroactively
   evict already-admitted content (re-gating)?** Currently no.

**Trigger**: `python -m server.scheduler` — a separate process (added to
`scripts/dev.*`), retrains every `HORSE_RETRAIN_INTERVAL` seconds (default
600; `0` disables), once immediately on startup. Deliberately not a thread
in the API (keeps retrain off the request path, no Flask-reloader
double-start) and not an HTTP endpoint. `python -m server.retrain` runs one
cycle on demand. Cycles never overlap: each holds an OS file lock
(`<HEAD_PATH>.lock`), so a manual run during a scheduled one is refused
(the scheduler just skips a busy tick), and `head.pt` is written
atomically (temp file + replace) so a killed save can't corrupt the
warm-start checkpoint (2026-10-19). One machine only; several would need
a job queue.

**Tuning against real data: `python -m horse.tune`** (2026-10-19) — read-only
report on a *copy* of the server's app.db (+ head.pt): vote stats, the
model's score vs people's mean per rated image, the uploads the gate turned
away, and a knob sweep that replays the loop from an anchors-only head
(fit, held-out-image fit, anchor health, unrated-image shift). Snapshots
live in git-ignored `tune/`. First real run (74 votes, 19 images, 6
people): the live head fits every rated image to within 0.08, and held-out
fit never beats the starting head at any `λ_anchor`/`prox` — votes are
memorized, not generalized. Led to the 8-hidden head (see "Model design");
`λ_anchor`/`prox` haven't been re-swept at that size yet.

Knobs (all `server/config.py`, `HORSE_*` env): `LAMBDA_ANCHOR` (0.25),
`LAMBDA_DECAY` (0.01), `COMMUNITY_WEIGHT_FLOOR` (3.0), `FEED_MIX`
("0.5,0.3,0.2"), `RETRAIN_INTERVAL_SECONDS` (600), `RETRAIN_EPOCHS` (200),
`RETRAIN_PROX_WEIGHT` (1.0), `RETRAIN_FRESH_EPOCHS` (3000), `HEAD_HIDDEN`
(8, set from real data), `FEED_STRATEGY` ("contested"), `FEED_NEWEST_SHARE`
(0.5). The rest aren't tuned against real data yet.

**`HORSE_FAKE_EMBEDDER=1`** swaps CLIP for a deterministic hash-based
stand-in (`horse/embed.py::FakeEmbedder`, via `get_embedder()`) — no
~600 MB download. Tests use it; also handy for running the app locally
without the model.

## Recommendation / feed algorithm (decided; built)

Community-global ranking — one feed for everyone, driven by aggregate
ratings, no per-user personalization. Built in
`server/services/feed.py::get_feed` (replaced the newest-first stub).

Feed ranking blends three pools (`_blended_score`, per image, independent
of the rest of the set so pagination is stable):
- **Consensus-horse**: high recency-weighted mean, low variance, enough
  votes — reinforces community identity.
- **Contested**: high recency-weighted rating variance — mirrors
  divisive/engagement-driven boosting on real platforms, drives
  debate/participation. This was a liked idea specifically because it
  mirrors real social media dynamics.
- **Exploration**: low total rating weight — ensures new/contested
  candidates get discovered rather than the same debated set recirculating
  forever.

Mean AND variance use the same recency weighting as training
(`w = exp(-λ_decay · Δt)`), so an image that was contested but has settled
drops out of the contested pool over time. Mix ratio is `config.FEED_MIX`
(`HORSE_FEED_MIX`, default 0.5/0.3/0.2) — a tunable parameter, possibly
community-votable later.

**Two feeds (2026-10-18)** — `GET /api/feed?feed=new|ranked`:
- **new** (front page `/`): only horses the viewer hasn't rated yet
  (logged out: every horse), ranked by `HORSE_FEED_STRATEGY` below
  (default now `contested`). A horse rated mid-scroll keeps its loaded
  slot for that page load; it's left out of later pages and reloads. A
  7-day "posted recently" window was tried first and dropped the same
  day in favour of unseen.
- **ranked** (`/ranked`, "most horse"): every gate-passing horse, most horse
  to least by model score (`top`), same for everyone, with a `#n` rank shown
  and nothing else — no rating buttons or vote averages (2026-10-19); you
  rate from the feed. The front page's nav label is "feed".
  This shows the model's ordering, but no variance / pool fields, so the "don't
  surface the mechanism" rule still holds.
Feed sessions are per (viewer, feed, session id).

**Strategies (2026-10-11, being compared)** — `HORSE_FEED_STRATEGY`
picks the *new* feed's ranking for everyone (server config switch, not a
UI choice):
- `blend` — the three-pool blend above.
- `contested` — most divisive first (`_contested_score`), interleaved with
  the newest horses (so fresh, unrated uploads still surface).
- `consensus` — highest model score first, interleaved the same way.
- `top` — straight highest model score, identical for everyone.

"Highest" = the **model's** horseness score (`images.score`, re-scored every
retrain), deliberately not user ratings — ratings reach the ranking only
through the drift head. "Unrated by the viewer" comes from the ratings table
— a horse scrolled past without rating stays "unseen" (no impressions
tracking). `HORSE_FEED_NEWEST_SHARE` (0.5) is the fraction of slots the
newest horses get in contested/consensus. The feed endpoint takes an
optional bearer token for the unseen filter; logged out, nothing counts as
rated.

**Feed sessions (no on-screen glitching):** the feed page sends a `session`
id generated once per page load. The server remembers which images it has
served to that (viewer, session); already-served positions are replayed
as-is (refetch after rating returns the same images in the same slots,
details live), and each *new* page is ranked fresh minus what's been served
— so new uploads, retrain re-scores and the viewer's own ratings only shape
content not yet loaded. Rated horses leave the "unrated" slots on the next
page load. In-process memory, 1h TTL. This is the first per-viewer element in the feed
— a filter on the viewer's own ratings, not a personalized model, so the
"no per-user personalization" spirit above still holds.

**MVP simplifications** (see `nubbins.md`): loads the whole gate-passing
catalogue + all ratings per request (O(catalogue) — fine at MVP scale);
offset pagination (`?limit=&offset=`), not keyset. The response exposes
**no** variance / pool / "contested" fields — the UI must not surface the
drift mechanism.

**Flagged risk**: contested-content boosting is a known driver of
toxicity/radicalization on real platforms. If this grows beyond an art
project, add a lightweight content-safety pass independent of the horse
classifier, so the "contested" pool doesn't become a magnet for abuse
(e.g. gore/NSFW gaming the ambiguity). Not needed for MVP but keep in mind.

---

## Spitballing — per-user "horse" models (NOT DECIDED, possible future feature)

> **Status: spitballing only.** Not on any roadmap, not an MVP concern,
> nothing here is a build instruction. Recorded so the idea isn't lost.
> If picked up, it's a post-MVP "second axis" feature that only makes
> sense once the collective mechanic is proven end to end and the real
> three-pool feed exists.

**The idea.** Alongside the single community "horse" concept, learn a
*personal* concept of "horse" per user from that user's own ratings. The
thematic point: the collective concept drifting over time is *cultural
evolution*; individual concepts diverging from each other at a single
moment is *filter bubbles / subcultures / rabbit-hole dynamics*. It's a
deliberate second mirror of real-platform pathology, complementary to the
contested pool (which already mirrors divisive-engagement boosting).

**Scope decision (spitball-level): FEED-ONLY.** The per-user model would
influence **feed ranking only**. The **gate stays global** — one shared,
community-shaped definition of what gets onto the platform. This keeps
"one platform, shared rules of entry" intact (consistent with the
unified-head rationale under "System architecture" — the gate is not
decoupled) and confines personalization to what each user *sees*, not
what *exists*. Personalized gating (what passes for you bounces for me)
was considered and set aside — it fragments "platform" into N platforms
and loses the single graspable art object. Could resurface as a later
opt-in experiment, but not the default.

**Feasibility.** Cheap, and the existing architecture already supports it
— frozen CLIP, cached embeddings, tiny trainable head. A per-user model
is just another small head (or better, see below) trained on cached
embeddings; CLIP is never re-run. The real constraint is **data
sparsity**: a head trained on one user's handful of ratings overfits into
noise. Options, in rough order of preference:
  1. **User-conditioned single model (preferred).** One head taking
     `[image_embedding, user_embedding]` → score, with a learned per-user
     vector (~8–32d) rather than a per-user network. Shares statistical
     strength across users; cold-start is automatic (new user = mean user
     embedding, then peels away as they rate); "your horse" is just the
     model conditioned on your vector. Least infra, degrades gracefully.
  2. **Hierarchical / partial pooling.** `personal = global +
     regularized_delta`, delta shrunk toward zero. A random-effects model
     in disguise. More granular, more knobs, more storage.
  3. **Faction models.** Cluster users into k "horse camps," train
     ~5–10 faction heads. Cheaper, overfitting-resistant, and factions
     are a fun emergent social object — but see UI rule below.

**Carried-over constraints if this is ever built:**
- **Anchor loss applies per-user too**, likely with a *stronger*
  `λ_anchor` than the collective model (far less personal data), or
  personal concepts collapse fast.
- **For the rabbit hole to actually form**, the loop must close: personal
  model → what you're shown → what you rate → personal model. If the
  per-user model only drives a display number, there's no rabbit hole,
  just a stat. It has to feed ranking (personal re-score of candidates,
  and/or personal exploration/contested mix) for real divergence to emerge.
- **Do not surface it.** Same rule as the drift mechanism — no "you are
  X% divergent from consensus" dashboard, no faction labels. Divergence
  is cheap to compute (distance between user vectors, or disagreement on
  a probe set) but it should be *felt* through the feed, not displayed.
- **Overlap with the contested pool:** community rating variance is
  partly per-user divergence aggregated. These two mechanisms measure
  overlapping things — don't double-count them in ranking.
- **Privacy note:** a per-user concept model is a fairly intimate
  behavioral fingerprint. Fine for an art project; worth acknowledging if
  scope ever grows.

---

## Web app architecture (decided)

**Detached client/server split from day one.** No server-rendered
prototype phase — the earlier two-phase plan (Flask + Jinja2 first, detach
later) was reconsidered and dropped. Building the SSR version only to throw
it away was wasted work, and keeping business logic "template-friendly just
in case" was a drag on the design. Go straight to the real architecture.

**Built and integrated** — `server/` (Flask JSON API) and `web/`
(SvelteKit) run as one app. See the root `README.md` to run both
(`scripts/dev.*`), plus `server/README.md` / `web/README.md` for detail.
Accounts, upload→gate, captions, the feed, ratings, and a `/me` page are
all wired to the real API.

**This app is part of the MVP** (see "Current focus" — revised). The core
loop is now closed (2026-09-03): site ratings reach the training store
via `horse/ingest.py`, `server/scheduler.py` retrains on an interval, the
API hot-reloads the head, images are re-scored, and the feed is the real
three-pool blend. `scripts/dev.*` now starts three processes (API + web +
scheduler). See "Retrain loop (built)".

**Feed UI direction (decided):** the "Stream" layout — social-app-first,
images lead, one column, rate in place. Rating (2026-10-19) is six big
bevelled 0–5 buttons, one tap, "not a horse" … "super horse", your own
vote shown pressed in (feed items carry `my_rating`). Chosen over a
no-default slider for one-tap mobile use; both mocked up in
`web/prototypes/rating.html`. 0–5 matches the model's scale (not-horse
anchors target 0, gate 2.5) — the old 1–5 buttons couldn't vote 0. Look is deliberately plain
old-web / "HTML energy", NOT modern-minimalist; hand-written CSS, no CSS
framework (Simple.css was tried and dropped). The UI must **not** surface
the drift mechanism — no "contested/consensus" labels, variance, or
"state of horse" analytics; the emergent behaviour only works if the app
doesn't point at it. Ported into the Svelte routes; all styling is one
global hand-written sheet at `web/src/app.css`. The `web/prototypes/*.html`
mockups it came from are kept as reference only.

- **Backend**: Python, serving a JSON API only — no HTML rendering. Flask
  (still the pick over FastAPI: CLIP inference is CPU-bound/blocking so
  async buys nothing; Flask over Django to avoid the upfront ceremony for
  a first webapp). Business logic (gate checks, rating writes, feed
  queries) lives in plain modules the API layer calls — no framework
  lock-in.
- **Frontend**: **Svelte (SvelteKit)**, consuming the JSON API. Chosen for
  the smallest learning curve from plain HTML/CSS/JS (operator is new to
  frontend frameworks), minimal boilerplate for a solo-maintained
  codebase, and strong built-ins (scoped styles, stores, transitions,
  routing) that suit a feed-driven app with a playful identity. Handles
  the smooth infinite-scroll feed (consensus/contested/exploration design
  above) via `@tanstack/svelte-query` + `IntersectionObserver` on a
  sentinel element against the paged API.
  - Accepted tradeoff: smaller ecosystem and a smaller AI-codegen training
    corpus than React. React + Vite + TanStack Query is the fallback if
    heavy reliance on AI-generated frontend code becomes the bottleneck.
  - Future mobile app is only a "maybe" — not weighted heavily, so the
    React Native code-sharing path was not a deciding factor.
  - Vue considered — sits between the two but doesn't win on any axis for
    this project.
- **Auth**: token-based (e.g. JWT) from the start, since the SPA and API
  are cross-origin. Session-cookie auth is off the table — it was only
  ever viable for the now-dropped SSR phase. (Careful cross-site cookie
  config — CORS, SameSite — is the fallback if token auth proves
  annoying, but token-first is the plan.)
  The client sends the JWT as **`X-Horse-Token`**, not `Authorization:
  Bearer` (still accepted, e.g. by tests): added while the deployment had
  a site-wide basic-auth password occupying `Authorization` (since
  replaced by invite codes), and kept. Deployed, the site and API
  share one origin (`PUBLIC_API_BASE` empty, Caddy routes `/api/*`); in
  local dev they're still cross-origin.

Note: the web app is part of the Stage 2 MVP now (see "Current focus" —
this was revised; the app was previously scoped as post-MVP). It's now
deployed as an invite-only private beta (see "Current focus");
public-facing hardening (upload guards, rate limiting, auth hardening)
stays deferred until it goes public.

Alternatives considered:
- **FastAPI**: rejected — async doesn't pay off (CLIP inference is
  CPU-bound/blocking regardless of framework).
- **Django**: rejected for now — more upfront ceremony (settings, apps,
  migrations framework) than Flask for a first webapp, though its
  built-in admin panel would help with moderation tooling if that need
  arrives sooner than planned (Stage 1/moderation is explicitly not in
  scope yet — see "Current focus").
- **Server-rendered / htmx / hybrid SSR**: rejected — full detachment is
  the target, so a hybrid or SSR-first step just adds throwaway work.
- **React**: strong option (largest ecosystem, best AI-codegen support,
  React Native path) but a steeper concept curve for a first frontend;
  kept as the documented fallback. **Vue**: no decisive advantage here.

---

## Explicitly deferred / rejected features

- **Active learning** (model-uncertainty-driven surfacing of images to
  rate): rejected — feels at odds with staying a straightforward social
  platform on the surface.
- **Aggregate horseness trendline** (public live metric of the platform's
  current "state of horse"): liked, deliberately saved as a post-MVP
  add-on feature.
- **Fully decoupled gate + rating models**: proposed early, then
  explicitly rejected in favor of the unified Stage 2 head described
  above — see rationale under "System architecture."

---

## Tech stack (decided)

- **Language**: Python.
- **CLIP encoder**: `open_clip_torch` (OpenCLIP by LAION) — chosen over
  `transformers`' CLIP or OpenAI's original `clip` repo for checkpoint
  flexibility and lightweight direct PyTorch integration.
- **Head + training loop**: plain PyTorch (`torch`), manual training loop,
  `torch.optim.Adam`, MSE loss. No training-loop-management framework
  (e.g. Lightning) needed at this scale.
- **Embedding/rating storage (MVP)**: SQLite or plain files
  (`.npy`/`.pt`), storing `(embedding, rating, timestamp, image_id)`
  records. NOT a vector DB yet — only reach for one (LanceDB, Chroma,
  pgvector) if/when similarity search over embeddings is actually needed.
- **Supporting libraries**: `Pillow` (image loading/preprocessing),
  `numpy` (array ops), `scikit-learn` (optional, for quick utilities like
  weighted averages/variance if not hand-rolled).

Suggested minimal stack for Stage 2 MVP:
```
open_clip_torch   # frozen image encoder
torch             # head model + training loop
Pillow            # image loading
numpy             # array ops
sqlite3 (stdlib)  # embedding/rating storage, no install needed
```

---

## Working notes for Claude Code

- **MVP target (revised): the running site, locally, closing the drift
  loop.** See "Current focus". The CLI pipeline and the web app both
  exist; the work is wiring site ratings → training store → retrain →
  live head reload, plus a real (community-global) feed algorithm. Track
  it in `nubbins.md` "MVP — core loop not closed".
- Keep the head model small and cheap to retrain — that's the whole point
  of the architecture (frozen backbone, tiny trainable head).
- Don't build Stage 1 (precheck/moderation) yet unless asked — current
  focus is Stage 2 only.
- Local-only is fine for the MVP — don't spend effort on deployment,
  hosting, upload guards, rate limiting, or account recovery yet (all
  explicitly deferred in "Current focus").
- Don't implement a vector DB, active learning, or the trendline feature
  unless asked — these are explicitly deferred.
- `nubbins.md` (git-ignored, repo root) is a running list of unfinished
  bits and extension points. Keep it current: add a nubbin whenever you
  stub or defer something, strike one when it's done.
