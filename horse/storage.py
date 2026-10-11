"""SQLite storage for community ratings: one row per (image, user) vote.

Deliberately plain sqlite3 (stdlib) -- see CLAUDE.md "Tech stack": not a
vector DB, no similarity search needed yet, just a durable place to
accumulate ratings between retrain cycles.

Two tables:

- ``ratings``    -- (image_id, user_id, rating, timestamp), one live row per
                    (image, user). Re-rating replaces that user's row and
                    refreshes its timestamp; the recency weighting in
                    train.py is keyed off each row's own timestamp, so every
                    vote is weighted individually (CLAUDE.md's
                    ``sum_i w(t_i) * L_rating``).
- ``embeddings`` -- (image_id, embedding), the image's CLIP embedding stored
                    once so training never has to re-run CLIP.

This is the *training* store. The web app keeps its own per-user ratings in
server/data/app.db; horse.ingest copies them here.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from horse.embed import EMBEDDING_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    image_id  TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    rating    REAL NOT NULL,
    timestamp REAL NOT NULL,
    PRIMARY KEY (image_id, user_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    image_id  TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def _migrate_legacy_schema(conn: sqlite3.Connection) -> None:
    """Drop the pre-per-user ``ratings`` table if we find it.

    The original schema was ``(image_id PRIMARY KEY, embedding, rating,
    timestamp)`` -- one row per image, no user dimension. There is no
    in-place migration (the embedding moved to its own table and the PK
    changed); the training store is regenerated from app.db by
    horse.ingest, so dropping an *empty* legacy table is safe. A non-empty
    one is left alone and the caller is told to delete the file.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ratings)")}
    if not cols or "user_id" in cols:
        return  # fresh db, or already the current schema
    (count,) = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()
    if count:
        raise RuntimeError(
            "horse/data/ratings.db uses the old single-rating-per-image schema "
            f"and has {count} rows. Delete the file and re-run horse.ingest to "
            "rebuild it in the new per-user format."
        )
    conn.execute("DROP TABLE ratings")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the ratings database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA busy_timeout = 10000")  # scheduler + API can hit this concurrently
    _migrate_legacy_schema(conn)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def add_rating(
    conn: sqlite3.Connection,
    image_id: str,
    user_id: str,
    embedding: np.ndarray,
    rating: float,
    timestamp: float | None = None,
) -> None:
    """Record (or replace) one user's rating of one image, plus its embedding.

    Re-rating the same (image_id, user_id) replaces that row and bumps its
    timestamp. The embedding is upserted once per image.
    """
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}")
    if timestamp is None:
        timestamp = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO ratings (image_id, user_id, rating, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (str(image_id), str(user_id), float(rating), float(timestamp)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (image_id, embedding) VALUES (?, ?)",
        (str(image_id), embedding.astype(np.float32).tobytes()),
    )
    conn.commit()


def all_ratings(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray, float, float]]:
    """Return every stored vote as (image_id, embedding, rating, timestamp).

    One tuple per (image, user) rating -- an image with N votes appears N
    times, each with that image's embedding. Rows whose embedding is
    missing (shouldn't happen via add_rating) are skipped.
    """
    rows = conn.execute(
        "SELECT r.image_id, e.embedding, r.rating, r.timestamp "
        "FROM ratings r JOIN embeddings e ON e.image_id = r.image_id"
    ).fetchall()
    records = []
    for image_id, blob, rating, timestamp in rows:
        embedding = np.frombuffer(blob, dtype=np.float32)
        records.append((image_id, embedding, rating, timestamp))
    return records
