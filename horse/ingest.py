"""Sync community ratings from the web app into the training store.

The web app (server/) records per-user ratings in server/data/app.db as
people rate images in the feed. horse.train only reads
horse/data/ratings.db. This module copies the former into the latter,
carrying each image's cached CLIP embedding across so training never
re-runs CLIP.

Full idempotent sync: re-running reflects the app's current state (a
user changing their rating in the app overwrites their row here on the
next sync). This is the bridge that closes the drift loop -- app ratings
-> training store -> horse.train -> new head.

Run:

    python -m horse.ingest [--app-db server/data/app.db] [--training-db horse/data/ratings.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from horse.embed import EMBEDDING_DIM
from horse.storage import add_rating, connect

DEFAULT_APP_DB = "server/data/app.db"
DEFAULT_TRAINING_DB = "horse/data/ratings.db"


def ingest(
    app_db_path: str | Path = DEFAULT_APP_DB,
    training_db_path: str | Path = DEFAULT_TRAINING_DB,
) -> dict:
    """Copy every app rating (+ its image embedding) into the training store.

    Returns {"ratings_synced": int, "images": int}.
    """
    app_db_path = Path(app_db_path)
    if not app_db_path.exists():
        return {"ratings_synced": 0, "images": 0}

    app = sqlite3.connect(f"file:{app_db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = app.execute(
            """
            SELECT r.image_id, r.user_id, r.rating, r.created_at, i.embedding
            FROM ratings r
            JOIN images i ON i.id = r.image_id
            """
        ).fetchall()
    finally:
        app.close()

    conn = connect(training_db_path)
    images: set[str] = set()
    synced = 0
    try:
        for image_id, user_id, rating, created_at, blob in rows:
            embedding = np.frombuffer(blob, dtype=np.float32)
            if embedding.shape != (EMBEDDING_DIM,):
                continue  # skip a malformed/legacy embedding rather than abort
            add_rating(conn, str(image_id), str(user_id), embedding, float(rating), float(created_at))
            images.add(str(image_id))
            synced += 1
    finally:
        conn.close()

    return {"ratings_synced": synced, "images": len(images)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-db", default=DEFAULT_APP_DB)
    parser.add_argument("--training-db", default=DEFAULT_TRAINING_DB)
    args = parser.parse_args()

    summary = ingest(args.app_db, args.training_db)
    print(
        f"[ingest] {summary['ratings_synced']} ratings across {summary['images']} images "
        f"-> {args.training_db}"
    )


if __name__ == "__main__":
    main()
