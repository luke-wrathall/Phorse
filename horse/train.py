"""Retrain the Stage 2 drift head: anchor loss + recency-weighted community loss.

    L = lambda_anchor * mean(anchor_sq_err)
        + sum_i( w(t_i) * rating_sq_err_i ) / max(sum_i w(t_i), community_weight_floor)
        + prox_weight * ||theta - theta_prev||^2        (warm-start only)
    w(t) = exp(-lambda_decay * elapsed_days)

The anchor term is a plain mean over the fixed anchor set (anchors.py).
The community term is a recency-weighted mean of the per-vote squared
errors, pooled over every rating on every image -- one platform-wide
numerator and denominator, not one pair per image. So:

- `lambda_anchor` is a *constant ratio* between the anchor pull and the
  community pull -- it is not diluted by rating volume (an earlier version
  summed the community term, which let the anchor be swamped after a few
  dozen ratings).
- `community_weight_floor` floors that single pooled denominator. It is
  numerical safety for near-zero total vote weight (launch, a platform-wide
  quiet spell); above that it is inert. It is NOT a per-image threshold.
- Stale ratings fade toward the anchor *relative to* the platform's other,
  fresher weighted activity -- a property of the pooled mean, not a fixed
  per-image guarantee. Pooling is deliberate: it caps an attacker's
  leverage at their share of all platform votes, however many images they
  spread them over (see CLAUDE.md, "Deliberately not per-image").
- The prox term pulls toward the previous cycle's head, so each retrain
  nudges rather than refits. It is a speed limit, not a cap: repeated
  cycles converge to the same fixed point, just gradually.

This is the "one unified head, no separate gate/rating models" mechanism
decided in CLAUDE.md: gate.py and score.py both read the checkpoint this
produces.

Run:

    python -m horse.train [--ratings-db path] [--head-out path]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from horse.anchors import DEFAULT_CACHE_PATH, load_or_build_anchor_set
from horse.model import DEFAULT_HIDDEN_DIM, HorseHead
from horse.storage import all_ratings, connect

DEFAULT_RATINGS_DB = "horse/data/ratings.db"
DEFAULT_HEAD_OUT = "horse/data/head.pt"

# lambda_anchor is a ratio between two mean losses: 0.1 weights the anchor
# mean 1:10 against the community mean, whatever the vote volume. Small but
# constant "rubber band". (Was 0.25; lowered 2026-10-27 -- on the real votes
# 0.1 lets unrated images shift a bit more with anchors still at 99%.)
DEFAULT_LAMBDA_ANCHOR = 0.1
# Per day -- ~350-day half-life. Above the weight floor, decay only weighs
# old votes against new ones; it doesn't pull toward the anchors. So this sets
# how long the site remembers through a quiet spell, not how far it drifts.
# (Was 0.01, ~70 days; changed 2026-10-27.)
DEFAULT_LAMBDA_DECAY = 0.002
# Floor on the platform-wide sum of recency weights (not per image): keeps
# the community term from blowing up when total effective votes are near zero.
DEFAULT_COMMUNITY_WEIGHT_FLOOR = 3.0
DEFAULT_EPOCHS = 200
# A head trained from scratch (first cycle, --fresh, a head-size change) has
# no warm start to build on: 200 epochs leaves even the 64-hidden head at 97%
# anchor accuracy and a linear head at ~50%, so fresh starts train this long.
DEFAULT_FRESH_EPOCHS = 3000
DEFAULT_LR = 1e-3
# Proximal pull toward the warm-start head's weights: makes each retrain cycle
# a *nudge* rather than a refit. Without it a ~33k-param head memorizes a
# freshly rated image to its exact rating in one cycle (seen with real CLIP,
# 2026-10-11), so lambda_decay barely matters. 0 disables; ignored when not
# warm-starting. The retrain loop sets it via config.RETRAIN_PROX_WEIGHT (1.0).
DEFAULT_PROX_WEIGHT = 0.0

SECONDS_PER_DAY = 86400.0


def recency_weights(timestamps: np.ndarray, lambda_decay: float, now: float | None = None) -> np.ndarray:
    """w(t) = exp(-lambda_decay * elapsed_days) for each rating timestamp."""
    now = time.time() if now is None else now
    elapsed_days = (now - timestamps) / SECONDS_PER_DAY
    return np.exp(-lambda_decay * elapsed_days).astype(np.float32)


def train_head(
    anchor_embeddings: np.ndarray,
    anchor_targets: np.ndarray,
    rating_embeddings: np.ndarray,
    rating_targets: np.ndarray,
    rating_weights: np.ndarray,
    *,
    lambda_anchor: float = DEFAULT_LAMBDA_ANCHOR,
    community_weight_floor: float = DEFAULT_COMMUNITY_WEIGHT_FLOOR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    head: HorseHead | None = None,
    prox_weight: float = DEFAULT_PROX_WEIGHT,
    log_every: int | None = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> tuple[HorseHead, dict]:
    """Full-batch gradient descent on the combined anchor + community loss.

    Full-batch (no minibatching) is fine at this scale -- the whole point
    of the frozen-backbone design is that retraining the head is cheap.
    Returns (head, {"anchor_loss", "community_loss", "final_loss"}) with the
    loss values from the last epoch.

    `prox_weight` > 0 with a warm-start `head` adds
    prox_weight * ||theta - theta_prev||^2, anchoring this cycle to the
    previous one so repeated cycles drift gradually.
    """
    prev_params = (
        [p.detach().clone() for p in head.parameters()]
        if head is not None and prox_weight > 0
        else None
    )
    head = head if head is not None else HorseHead(hidden_dim=hidden_dim)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    anchor_x = torch.from_numpy(anchor_embeddings)
    anchor_y = torch.from_numpy(anchor_targets)

    has_ratings = rating_embeddings.shape[0] > 0
    if has_ratings:
        rating_x = torch.from_numpy(rating_embeddings)
        rating_y = torch.from_numpy(rating_targets)
        rating_w = torch.from_numpy(rating_weights)
        # one pooled denominator for all ratings, floored only for numerical
        # safety at near-zero total weight (see module docstring).
        weight_denom = torch.clamp(rating_w.sum(), min=community_weight_floor)

    anchor_loss = community_loss = total = torch.tensor(0.0)
    for epoch in range(epochs):
        optimizer.zero_grad()

        anchor_pred = head(anchor_x)
        anchor_loss = torch.mean((anchor_pred - anchor_y) ** 2)
        total = lambda_anchor * anchor_loss

        if has_ratings:
            rating_pred = head(rating_x)
            per_example = (rating_pred - rating_y) ** 2
            community_loss = torch.sum(rating_w * per_example) / weight_denom
            total = total + community_loss

        if prev_params is not None:
            prox = sum(torch.sum((p - q) ** 2) for p, q in zip(head.parameters(), prev_params))
            total = total + prox_weight * prox

        total.backward()
        optimizer.step()

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            community_val = float(torch.as_tensor(community_loss).detach())
            print(
                f"[train] epoch {epoch:4d}  anchor={anchor_loss.item():.4f}  "
                f"community={community_val:.4f}  total={total.item():.4f}"
            )

    head.eval()
    summary = {
        "anchor_loss": float(anchor_loss.detach()),
        "community_loss": float(torch.as_tensor(community_loss).detach()),
        "final_loss": float(total.detach()),
    }
    return head, summary


def load_ratings(db_path: str, embedding_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings, ratings, timestamps), each possibly empty."""
    conn = connect(db_path)
    records = all_ratings(conn)
    conn.close()
    if not records:
        return (
            np.zeros((0, embedding_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    embeddings = np.stack([r[1] for r in records]).astype(np.float32)
    ratings = np.array([r[2] for r in records], dtype=np.float32)
    timestamps = np.array([r[3] for r in records], dtype=np.float64)
    return embeddings, ratings, timestamps


def train_from_store(
    ratings_db: str = DEFAULT_RATINGS_DB,
    anchor_cache: str = str(DEFAULT_CACHE_PATH),
    head_out: str = DEFAULT_HEAD_OUT,
    *,
    head_in: str | None = None,
    lambda_anchor: float = DEFAULT_LAMBDA_ANCHOR,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
    community_weight_floor: float = DEFAULT_COMMUNITY_WEIGHT_FLOOR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    prox_weight: float = DEFAULT_PROX_WEIGHT,
    log_every: int | None = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    fresh_epochs: int = DEFAULT_FRESH_EPOCHS,
) -> dict:
    """One retrain cycle from the databases: anchors + ratings -> new head.pt.

    Warm-starts from `head_in` when it exists and is `hidden_dim` wide;
    otherwise (no head yet, or the head size was changed) trains a new
    head from scratch for `fresh_epochs`.

    Does NOT ingest -- the caller syncs app ratings into `ratings_db`
    first (see horse.ingest). Returns a summary dict (loss values + counts).
    """
    anchor_set = load_or_build_anchor_set(cache_path=anchor_cache)
    embedding_dim = anchor_set.embeddings.shape[1]

    rating_embeddings, rating_targets, timestamps = load_ratings(ratings_db, embedding_dim)
    rating_weights = (
        recency_weights(timestamps, lambda_decay)
        if len(timestamps)
        else timestamps.astype(np.float32)
    )

    head = HorseHead.load(head_in) if head_in and Path(head_in).exists() else None
    if head is not None and head.hidden_dim != hidden_dim:
        head = None  # the head size changed: start over at the new size

    trained, summary = train_head(
        anchor_set.embeddings,
        anchor_set.targets,
        rating_embeddings,
        rating_targets,
        rating_weights,
        lambda_anchor=lambda_anchor,
        community_weight_floor=community_weight_floor,
        epochs=epochs if head is not None else fresh_epochs,
        lr=lr,
        head=head,
        prox_weight=prox_weight,
        log_every=log_every,
        hidden_dim=hidden_dim,
    )
    # atomic write: the API process may be reading head.pt (mtime hot-reload)
    # while this runs in the scheduler process.
    tmp_out = f"{head_out}.tmp"
    trained.save(tmp_out)
    os.replace(tmp_out, head_out)

    summary.update(
        {"n_ratings": int(len(rating_targets)), "n_anchors": int(len(anchor_set.ids)),
         "head_out": head_out, "fresh": head is None}
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the Stage 2 drift head.")
    parser.add_argument("--ratings-db", default=DEFAULT_RATINGS_DB)
    parser.add_argument("--anchor-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--head-out", default=DEFAULT_HEAD_OUT)
    parser.add_argument("--head-in", default=None, help="Warm-start from a checkpoint instead of random init.")
    parser.add_argument("--lambda-anchor", type=float, default=DEFAULT_LAMBDA_ANCHOR)
    parser.add_argument("--lambda-decay", type=float, default=DEFAULT_LAMBDA_DECAY)
    parser.add_argument("--community-weight-floor", type=float, default=DEFAULT_COMMUNITY_WEIGHT_FLOOR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fresh-epochs", type=int, default=DEFAULT_FRESH_EPOCHS,
                        help="Epochs when there's no --head-in to warm-start from.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help="Head size: hidden units, 0 = linear.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--prox-weight", type=float, default=DEFAULT_PROX_WEIGHT,
                        help="Pull toward the --head-in weights (0 disables).")
    parser.add_argument("--log-every", type=int, default=50, help="Print loss every N epochs (0 to silence).")
    parser.add_argument("--rebuild-anchors", action="store_true", help="Force re-embedding the anchor set.")
    args = parser.parse_args()

    if args.rebuild_anchors:
        load_or_build_anchor_set(cache_path=args.anchor_cache, force_rebuild=True)

    summary = train_from_store(
        args.ratings_db,
        args.anchor_cache,
        args.head_out,
        head_in=args.head_in,
        lambda_anchor=args.lambda_anchor,
        lambda_decay=args.lambda_decay,
        community_weight_floor=args.community_weight_floor,
        epochs=args.epochs,
        fresh_epochs=args.fresh_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        prox_weight=args.prox_weight,
        log_every=args.log_every or None,
    )
    print(
        f"[train] {summary['n_ratings']} community ratings, {summary['n_anchors']} anchors "
        f"(lambda_anchor={args.lambda_anchor}, lambda_decay={args.lambda_decay}) "
        f"-> {args.head_out}  [anchor={summary['anchor_loss']:.4f} "
        f"community={summary['community_loss']:.4f}]"
    )


if __name__ == "__main__":
    main()
