"""SQLite access for the web app: one connection per request, on `g`.

This is the *app* database (users, image metadata, per-user ratings) --
distinct from horse/data/ratings.db, which is the training aggregate store
that horse.train reads. Bridging app ratings into the training store is a
tracked nubbin, not done here.

Plain sqlite3 to match the rest of the project (CLAUDE.md "Tech stack").
Schema is created on first app start; there is no migration framework yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import Flask, current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    invite_code   TEXT,                   -- the code they signed up with
    banned        INTEGER NOT NULL DEFAULT 0,
    tokens_after  INTEGER                 -- tokens issued before this (unix s) are dead
);

CREATE TABLE IF NOT EXISTS images (
    id         TEXT PRIMARY KEY,          -- uuid4 hex
    owner_id   INTEGER NOT NULL REFERENCES users(id),
    filename   TEXT,                      -- original upload name, for display only
    ext        TEXT NOT NULL,             -- stored-file extension, incl. leading dot
    caption    TEXT,                      -- optional uploader caption
    embedding  BLOB NOT NULL,             -- float32 CLIP embedding, cached for retrain
    score      REAL NOT NULL,             -- Stage 2 head score at upload time
    passed     INTEGER NOT NULL,          -- gate decision at upload time (0/1)
    created_at REAL NOT NULL,
    hidden_at  REAL,                      -- taken off the site; NULL = showing
    hidden_reason TEXT,                   -- 'reported', 'operator', 'owner banned', 'quarantined'
    quarantined_at REAL                   -- suspected illegal: file moved to QUARANTINE_DIR
);
CREATE INDEX IF NOT EXISTS idx_images_feed ON images(passed, created_at DESC);

CREATE TABLE IF NOT EXISTS ratings (
    image_id   TEXT NOT NULL REFERENCES images(id),
    user_id    INTEGER NOT NULL REFERENCES users(id),
    rating     REAL NOT NULL,             -- 0-5 "horse out of 5"
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, user_id)       -- one live rating per user per image
);

-- The embeddings of a showing image at its other upload-time rotations
-- (services/images.py). Training augmentations: a vote on the image also
-- trains the head on these, at config AUGMENT_WEIGHT.
CREATE TABLE IF NOT EXISTS image_augments (
    image_id   TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    angle      INTEGER NOT NULL,          -- degrees from the stored orientation
    embedding  BLOB NOT NULL,             -- float32 CLIP embedding
    PRIMARY KEY (image_id, angle)
);

CREATE TABLE IF NOT EXISTS invites (
    code       TEXT PRIMARY KEY,          -- lowercase, what people type in
    max_uses   INTEGER,                   -- signups allowed; NULL = unlimited
    uses       INTEGER NOT NULL DEFAULT 0,
    note       TEXT,                      -- operator's reminder, e.g. "insta story"
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id     TEXT,                    -- NULL = a general complaint. No FK: outlives the post
    reason       TEXT NOT NULL,           -- services/reports.py REASONS
    details      TEXT,
    contact      TEXT,                    -- optional, for a reply
    reporter_key TEXT NOT NULL,           -- limits.client_key: tells reporters apart
    reporter_user_id INTEGER,             -- NULL = logged out
    created_at   REAL NOT NULL,
    resolved_at  REAL,                    -- NULL = still in the queue
    resolution   TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_open ON reports(resolved_at, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_once
    ON reports(image_id, reporter_key, reason) WHERE image_id IS NOT NULL;

-- Uploads the content check refused. The image itself isn't kept.
CREATE TABLE IF NOT EXISTS upload_blocks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    filename   TEXT,
    sha256     TEXT NOT NULL,             -- of the uploaded file
    categories TEXT NOT NULL,             -- comma-separated, from services/safety.py
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_events (
    kind       TEXT NOT NULL,             -- 'upload-user', 'signup', ...
    key        TEXT NOT NULL,             -- limits.client_key or a user id
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_events ON rate_events(kind, key, created_at);

CREATE TABLE IF NOT EXISTS pauses (
    what  TEXT PRIMARY KEY,               -- services/limits.py PAUSABLE
    since REAL NOT NULL
);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        path = Path(current_app.config["DB_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")  # API + scheduler write app.db concurrently
        g.db = conn
    return g.db


def close_db(_exc: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


# Columns added after the first schema shipped. `CREATE TABLE IF NOT EXISTS`
# won't add them to an existing db, and there's no migration framework yet
# (see nubbins.md), so we add them by hand, ignoring "duplicate column".
_ADD_COLUMNS = [
    "ALTER TABLE images ADD COLUMN caption TEXT",
    "ALTER TABLE users ADD COLUMN invite_code TEXT",
    "ALTER TABLE users ADD COLUMN banned INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE users ADD COLUMN tokens_after INTEGER",
    "ALTER TABLE images ADD COLUMN hidden_at REAL",
    "ALTER TABLE images ADD COLUMN hidden_reason TEXT",
    "ALTER TABLE images ADD COLUMN quarantined_at REAL",
]


def init_db() -> None:
    db = get_db()
    db.executescript(SCHEMA)
    for statement in _ADD_COLUMNS:
        try:
            db.execute(statement)
        except sqlite3.OperationalError:
            pass  # column already exists
    db.commit()


def init_app(app: Flask) -> None:
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
