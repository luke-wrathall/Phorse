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
    created_at    REAL NOT NULL
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
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_feed ON images(passed, created_at DESC);

CREATE TABLE IF NOT EXISTS ratings (
    image_id   TEXT NOT NULL REFERENCES images(id),
    user_id    INTEGER NOT NULL REFERENCES users(id),
    rating     REAL NOT NULL,             -- 0-5 "horse out of 5"
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, user_id)       -- one live rating per user per image
);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        path = Path(current_app.config["DB_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
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
