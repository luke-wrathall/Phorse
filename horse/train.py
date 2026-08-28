"""Retrain the Stage 2 drift head: anchor loss + recency-weighted community loss.

    L = lambda_anchor * L_anchor(seed_data) + sum_i w(t_i) * L_rating(image_i, rating_i)
    w(t) = exp(-lambda_decay * elapsed_days)

L_anchor is the mean-squared error over the fixed anchor set (constant
weight, never decays -- see anchors.py). Each community rating is its
own per-example squared error, weighted by how recently it was made.
This is the "one unified head, no separate gate/rating models" mechanism
decided in CLAUDE.md: gate.py and score.py both read the same checkpoint
this produces.

Run:

    python -m horse.train [--ratings-db path] [--head-out path]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from horse.anchors import DEFAULT_CACHE_PATH, load_or_build_anchor_set
from horse.model import HorseHead
from horse.storage import all_ratings, connect

DEFAULT_RATINGS_DB = "horse/data/ratings.db"
DEFAULT_HEAD_OUT = "horse/data/head.pt"

DEFAULT_LAMBDA_ANCHOR = 1.0
DEFAULT_LAMBDA_DECAY = 0.01  # per day -- ~70-day half-life; the "how fast culture moves" knob
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-3

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
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    head: HorseHead | None = None,
) -> HorseHead:
    """Full-batch gradient descent on the combined anchor + community loss.

    Full-batch (no minibatching) is fine at this scale -- the whole point
    of the frozen-backbone design is that retraining the head is cheap.
    """
    head = head if head is not None else HorseHead()
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    anchor_x = torch.from_numpy(anchor_embeddings)
    anchor_y = torch.from_numpy(anchor_targets)

    has_ratings = rating_embeddings.shape[0] > 0
    if has_ratings:
        rating_x = torch.from_numpy(rating_embeddings)
        rating_y = torch.from_numpy(rating_targets)
        rating_w = torch.from_numpy(rating_weights)

    for _ in range(epochs):
        optimizer.zero_grad()

        anchor_pred = head(anchor_x)
        loss = lambda_anchor * torch.mean((anchor_pred - anchor_y) ** 2)

        if has_ratings:
            rating_pred = head(rating_x)
            per_example_loss = (rating_pred - rating_y) ** 2
            loss = loss + torch.sum(rating_w * per_example_loss)

        loss.backward()
        optimizer.step()

    head.eval()
    return head


def load_ratings(db_path: str, embedding_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings, ratings, timestamps), each possibly empty."""
    conn = connect(db_path)
    records = all_ratings(conn)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the Stage 2 drift head.")
    parser.add_argument("--ratings-db", default=DEFAULT_RATINGS_DB)
    parser.add_argument("--anchor-cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--head-out", default=DEFAULT_HEAD_OUT)
    parser.add_argument("--head-in", default=None, help="Warm-start from a checkpoint instead of random init.")
    parser.add_argument("--lambda-anchor", type=float, default=DEFAULT_LAMBDA_ANCHOR)
    parser.add_argument("--lambda-decay", type=float, default=DEFAULT_LAMBDA_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rebuild-anchors", action="store_true", help="Force re-embedding the anchor set.")
    args = parser.parse_args()

    anchor_set = load_or_build_anchor_set(cache_path=args.anchor_cache, force_rebuild=args.rebuild_anchors)
    embedding_dim = anchor_set.embeddings.shape[1]

    rating_embeddings, rating_targets, timestamps = load_ratings(args.ratings_db, embedding_dim)
    rating_weights = recency_weights(timestamps, args.lambda_decay) if len(timestamps) else timestamps.astype(np.float32)

    head = HorseHead.load(args.head_in) if args.head_in else None

    trained = train_head(
        anchor_set.embeddings,
        anchor_set.targets,
        rating_embeddings,
        rating_targets,
        rating_weights,
        lambda_anchor=args.lambda_anchor,
        epochs=args.epochs,
        lr=args.lr,
        head=head,
    )

    trained.save(args.head_out)
    print(
        f"[train] {len(rating_targets)} community ratings, {len(anchor_set.ids)} anchors "
        f"(lambda_anchor={args.lambda_anchor}, lambda_decay={args.lambda_decay}) -> {args.head_out}"
    )


if __name__ == "__main__":
    main()
