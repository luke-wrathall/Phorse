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

    # JWT lifetime. Default 7 days.
    TOKEN_TTL_SECONDS = int(os.environ.get("HORSE_TOKEN_TTL", str(7 * 24 * 3600)))

    # Browser origins allowed to call /api/* (the SvelteKit dev server).
    CORS_ORIGINS = os.environ.get(
        "HORSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    TESTING = _bool("HORSE_TESTING", False)
