# Project: "Horse" Platform — Stage 2 Classifier

## What this project is
An Instagram-adjacent social platform where every uploaded image must pass
a naive classifier identifying it as a "horse." The core creative premise:
the platform's concept of "horse" is not fixed — it evolves as a living
microcosm shaped by community ratings, drifting over time in ways that
reflect the culture of the user base. Currently building the classifier
system, starting with Stage 2 (see architecture below).

## Current focus
Stage 2 MVP: a working embed → train → score pipeline. Minimal dependencies,
no UI, no moderation layer yet. Goal is to prove the core mechanic works
end to end before layering on anything else.

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
  e.g. `512 → 64 → 1`) outputs a continuous **horseness score** — trained
  via regression (MSE), not binary classification.
- Compute embedding once per image, cache/reuse it — don't re-run CLIP
  per head or per retrain.
- Ratings are framed as regression targets ("horse out of 5" style, or a
  binary "🐴 horse" tap as a strong-signal vote), not class labels — this
  matches the degree-of-horseness concept, not fixed category membership.

## Training / drift mechanism (anchor + decay, decided)

Combined loss per retrain cycle:
```
L = λ_anchor * L_anchor(seed_data) + Σ w(t_i) * L_rating(image_i, rating_i)
```

- **Anchor loss**: fixed seed/reference set (e.g. CLIP zero-shot "a photo
  of a horse" scores, or curated reference images) included in every
  retrain with a small but CONSTANT (non-decaying) weight `λ_anchor`.
  Prevents total collapse of the horse-concept regardless of how far
  community ratings drift — acts like a rubber band, not a wall. Operator
  (you) can lightly curate/update the anchor set manually as an escape
  hatch if drift goes somewhere undesirable — the community does not
  control the anchor set.
- **Community loss**: each rating weighted by recency,
  `w(t) = exp(-λ_decay * Δt)`. `λ_decay` is the main "how fast does
  culture move" knob.
- **Why exponential decay**: constant half-life (proportional fade
  regardless of rating age, no jarring cutoffs), cheap recursive update
  (`aggregate_new = aggregate_old * exp(-λΔt) + new_ratings`, no need to
  recompute over full history each cycle), and it's the standard default
  for "constant proportional fade" processes.
- Alternatives considered but not chosen for the core mechanism: linear
  decay (hard cutoff, easier to explain to users), power-law decay (old
  ratings asymptotically linger, "cultural memory" feel), step-function
  "seasons" (discrete eras rather than smooth blending — flagged as
  possibly more legible/fun, worth revisiting for UI later, could be
  layered on top of the smooth exponential math under the hood as a
  presentation-layer hybrid).

## Recommendation / feed algorithm (decided, not yet built)

Feed ranking blends three pools:
- **Consensus-horse**: high mean, low variance — reinforces community
  identity.
- **Contested**: high rating variance — mirrors divisive/engagement-driven
  boosting on real platforms, drives debate/participation. This was a
  liked idea specifically because it mirrors real social media dynamics.
- **Exploration**: low rating-count — ensures new/contested candidates get
  discovered rather than the same debated set recirculating forever.

Rating variance should decay with the same recency weighting as the mean
(so old contested images don't linger in the contested pool after
settling into consensus). Mixing ratio between pools (e.g. 50/30/20) is a
tunable parameter, possibly community-votable later.

**Flagged risk**: contested-content boosting is a known driver of
toxicity/radicalization on real platforms. If this grows beyond an art
project, add a lightweight content-safety pass independent of the horse
classifier, so the "contested" pool doesn't become a magnet for abuse
(e.g. gore/NSFW gaming the ambiguity). Not needed for MVP but keep in mind.

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
all wired to the real API. Feed ranking is still a newest-first
placeholder, not the three-pool blend — see `nubbins.md`.

**Feed UI direction (decided):** the "Stream" layout — social-app-first,
images lead, one column, rate in place. Look is deliberately plain
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

Note: this is still a Stage 2 concern — the current focus (embed → train →
score pipeline) has no UI yet. The web app comes after the core mechanic
is proven end to end.

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

- When scaffolding, start with the smallest possible working pipeline:
  load an image → CLIP embedding → head forward pass → score. Get this
  working before adding training loop, storage, or Stage 1.
- Keep the head model small and cheap to retrain — that's the whole point
  of the architecture (frozen backbone, tiny trainable head).
- Don't build Stage 1 (precheck/moderation) yet unless asked — current
  focus is Stage 2 only.
- Don't implement a vector DB, active learning, or the trendline feature
  unless asked — these are explicitly deferred.
- `nubbins.md` (git-ignored, repo root) is a running list of unfinished
  bits and extension points. Keep it current: add a nubbin whenever you
  stub or defer something, strike one when it's done.
