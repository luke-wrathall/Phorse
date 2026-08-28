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

Two-phase plan, not one framework choice:

- **Phase 1 — prototype**: server-rendered. Flask + Jinja2 templates, full
  page reloads, session-cookie auth (`Flask-Login` + `werkzeug.security`
  password hashing). Goal is the fastest path to a real, working
  accounts + uploads + feed slice to validate the core mechanic with real
  users before investing in a richer frontend.
- **Phase 2 — final app**: detach the frontend. A separate JS framework
  (React/Vue/Svelte — not yet chosen) consuming a JSON API served by the
  Python backend, replacing the Jinja-rendered pages. Decided specifically
  for smooth infinite-scroll feed interactions — considered a key part of
  the app's feel given the consensus/contested/exploration feed design
  above — and it also opens the door to a future mobile app reusing the
  same API.

Implication for Phase 1 build-out: keep business logic (gate checks,
rating writes, feed queries) separate from template-rendering code where
practical, so routes can migrate toward returning JSON in Phase 2 without
a full rewrite.

Open question, not yet decided: Phase 1's simple session-cookie auth
likely won't carry over cleanly to Phase 2 — a cross-origin SPA + API
split typically needs token-based auth (e.g. JWT) or careful cross-site
cookie config (CORS, SameSite). Revisit when Phase 2 actually starts.

Alternatives considered for Phase 1:
- **FastAPI**: rejected — its main strength (async) doesn't pay off here
  since CLIP inference is CPU-bound/blocking regardless of framework, and
  server-rendered HTML isn't its focus anyway.
- **Django**: rejected for now — more upfront ceremony (settings, apps,
  migrations framework) than Flask for a first webapp, though its
  built-in admin panel would help with moderation tooling if that need
  arrives sooner than planned (Stage 1/moderation is explicitly not in
  scope yet — see "Current focus").
- **htmx / hybrid SSR**: considered as a middle ground (partial-page
  updates without a separate JS app), but not chosen since Phase 1 is
  intentionally full server-rendered and Phase 2 goes fully detached
  rather than staying hybrid.

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
