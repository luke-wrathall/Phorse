"""SQLite storage for community ratings: (image_id, embedding, rating, timestamp).

Deliberately plain sqlite3 (stdlib) -- see CLAUDE.md "Tech stack": not a
vector DB, no similarity search needed yet, just a durable place to
accumulate ratings between retrain cycles. The embedding is stored
alongside the rating so training never has to re-run CLIP.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from horse.embed import EMBEDDING_DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    image_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    rating REAL NOT NULL,
    timestamp REAL NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the ratings database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def add_rating(
    conn: sqlite3.Connection,
    image_id: str,
    embedding: np.ndarray,
    rating: float,
    timestamp: float | None = None,
) -> None:
    """Record (or overwrite) one image's rating and its embedding.

    Re-rating the same image_id replaces the previous rating and bumps
    its timestamp -- the recency weighting in train.py is keyed off this.
    """
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"expected embedding shape ({EMBEDDING_DIM},), got {embedding.shape}")
    if timestamp is None:
        timestamp = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO ratings (image_id, embedding, rating, timestamp) VALUES (?, ?, ?, ?)",
        (image_id, embedding.astype(np.float32).tobytes(), float(rating), float(timestamp)),
    )
    conn.commit()


def all_ratings(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray, float, float]]:
    """Return every stored (image_id, embedding, rating, timestamp) record."""
    rows = conn.execute("SELECT image_id, embedding, rating, timestamp FROM ratings").fetchall()
    records = []
    for image_id, blob, rating, timestamp in rows:
        embedding = np.frombuffer(blob, dtype=np.float32)
        records.append((image_id, embedding, rating, timestamp))
    return records
