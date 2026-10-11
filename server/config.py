"""Runtime config, all overridable via environment variables.

Kept dead simple for the scaffold -- one class, read once at app creation.
Nothing here is secret-safe yet; HORSE_SECRET_KEY must be set for any real
deployment (JWTs are signed with it).
"""

from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


class Config:
    # Signs JWTs. Overridden in tests; MUST be set for real use.
    SECRET_KEY = os.environ.get("HORSE_SECRET_KEY", "dev-insecure-change-me")

    # App database: users, uploaded-image metadata, ratings. Separate from
    # horse/data/ratings.db (the training store) -- see server/README.md.
    DB_PATH = os.environ.get("HORSE_APP_DB", "server/data/app.db")
    UPLOAD_DIR = os.environ.get("HORSE_UPLOAD_DIR", "server/data/uploads")

    # Stage 2 head checkpoint + gate threshold (mirrors horse.gate defaults).
    HEAD_PATH = os.environ.get("HORSE_HEAD_PATH", "horse/data/head.pt")
    GATE_THRESHOLD = float(os.environ.get("HORSE_GATE_THRESHOLD", "2.5"))
    # Uploads are scored at 0/90/180/270 degrees; a rotation is kept (and
    # saved) only if it beats upright by this much. Set very high to disable.
    UPLOAD_ROTATION_MARGIN = float(os.environ.get("HORSE_UPLOAD_ROTATION_MARGIN", "0.5"))

    # Retrain loop (server.scheduler). The training store is the aggregate
    # db horse.train reads; app ratings are synced into it each cycle.
    TRAINING_DB = os.environ.get("HORSE_TRAINING_DB", "horse/data/ratings.db")
    ANCHOR_CACHE = os.environ.get("HORSE_ANCHOR_CACHE", "horse/data/anchor_cache.npz")
    RETRAIN_INTERVAL_SECONDS = int(os.environ.get("HORSE_RETRAIN_INTERVAL", "600"))
    LAMBDA_ANCHOR = float(os.environ.get("HORSE_LAMBDA_ANCHOR", "0.25"))
    LAMBDA_DECAY = float(os.environ.get("HORSE_LAMBDA_DECAY", "0.01"))
    COMMUNITY_WEIGHT_FLOOR = float(os.environ.get("HORSE_COMMUNITY_WEIGHT_FLOOR", "3.0"))
    # Per-cycle training: each retrain warm-starts from head.pt and is pulled
    # toward those weights, so a cycle nudges rather than refits. Drift speed
    # is therefore per *cycle* -- it scales with RETRAIN_INTERVAL_SECONDS.
    RETRAIN_EPOCHS = int(os.environ.get("HORSE_RETRAIN_EPOCHS", "200"))
    RETRAIN_PROX_WEIGHT = float(os.environ.get("HORSE_RETRAIN_PROX_WEIGHT", "1.0"))

    # Feed ranking strategy (services/feed.py):
    #   blend      -- three-pool blend below, community-global
    #   contested  -- most divisive horses, interleaved with ones you haven't rated
    #   consensus  -- highest-rated horses, interleaved with ones you haven't rated
    #   top        -- straight highest-rated, same for everyone
    FEED_STRATEGY = os.environ.get("HORSE_FEED_STRATEGY", "blend")
    # contested/consensus: fraction of feed slots given to horses the viewer
    # hasn't rated yet (newest first).
    FEED_UNSEEN_SHARE = float(os.environ.get("HORSE_FEED_UNSEEN_SHARE", "0.5"))

    # Feed pool mix (blend strategy): consensus, contested, exploration. Must sum > 0.
    FEED_MIX = tuple(
        float(x) for x in os.environ.get("HORSE_FEED_MIX", "0.5,0.3,0.2").split(",")
    )

    # JWT lifetime. Default 7 days.
    TOKEN_TTL_SECONDS = int(os.environ.get("HORSE_TOKEN_TTL", str(7 * 24 * 3600)))

    # Browser origins allowed to call /api/* (the SvelteKit dev server).
    CORS_ORIGINS = os.environ.get(
        "HORSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    TESTING = _bool("HORSE_TESTING", False)
