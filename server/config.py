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


# Fallback JWT secret for the localhost dev server only. server.wsgi refuses
# to serve under a real WSGI server (gunicorn) with it -- see
# security.check_secret_key.
DEV_SECRET_KEY = "dev-insecure-change-me"


class Config:
    # Signs JWTs. Overridden in tests; MUST be set for real use.
    SECRET_KEY = os.environ.get("HORSE_SECRET_KEY", DEV_SECRET_KEY)

    # Upload limits. MAX_CONTENT_LENGTH is Flask's own request-body cap (413
    # past it); MAX_UPLOAD_PIXELS stops a small, highly compressed file from
    # decoding into a huge bitmap (each upload is decoded and rotated 4x).
    MAX_CONTENT_LENGTH = int(float(os.environ.get("HORSE_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
    MAX_UPLOAD_PIXELS = int(os.environ.get("HORSE_MAX_UPLOAD_PIXELS", str(40_000_000)))

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
    # A head trained from scratch (the first cycle, `retrain --fresh`, or
    # after HEAD_HIDDEN changes) gets this many epochs instead.
    RETRAIN_FRESH_EPOCHS = int(os.environ.get("HORSE_RETRAIN_FRESH_EPOCHS", "3000"))
    # Head size: hidden units, 0 = a plain linear head. Smaller heads can't
    # memorize single images, so votes move the score of similar images too
    # (see horse.tune). Changing it makes the next cycle start a fresh head.
    HEAD_HIDDEN = int(os.environ.get("HORSE_HEAD_HIDDEN", "8"))

    # Two feeds (services/feed.py): "new" (the front page) is the horses you
    # haven't rated yet, ranked by FEED_STRATEGY; "ranked" (/ranked) is
    # every horse, most horse first by the model's score.
    # The new feed's ranking:
    #   contested  -- most divisive horses, interleaved with the newest
    #   consensus  -- most horse by the model's score, interleaved the same way
    #   blend      -- the three-pool blend below
    #   top        -- straight most horse
    FEED_STRATEGY = os.environ.get("HORSE_FEED_STRATEGY", "contested")
    # contested/consensus: fraction of feed slots given to the newest horses,
    # so fresh uploads surface before anyone has rated them.
    FEED_NEWEST_SHARE = float(os.environ.get("HORSE_FEED_NEWEST_SHARE", "0.5"))

    # Feed pool mix (blend strategy): consensus, contested, exploration. Must sum > 0.
    FEED_MIX = tuple(
        float(x) for x in os.environ.get("HORSE_FEED_MIX", "0.5,0.3,0.2").split(",")
    )

    # Signing up needs an invite code (services/invites.py). On unless
    # turned off, so a deploy can't open signups by forgetting to set it;
    # scripts/dev.* turn it off for local dev.
    REQUIRE_INVITE = _bool("HORSE_REQUIRE_INVITE", True)

    # JWT lifetime. Default 7 days.
    TOKEN_TTL_SECONDS = int(os.environ.get("HORSE_TOKEN_TTL", str(7 * 24 * 3600)))

    # Browser origins allowed to call /api/* (the SvelteKit dev server).
    CORS_ORIGINS = os.environ.get(
        "HORSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    # The site's public address, for links the CLI prints (invite links).
    # Defaults to the first CORS origin: the domain when deployed, the
    # SvelteKit dev server locally.
    SITE_URL = (os.environ.get("HORSE_SITE_URL") or CORS_ORIGINS[0]).rstrip("/")

    TESTING = _bool("HORSE_TESTING", False)
