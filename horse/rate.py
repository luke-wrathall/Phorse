"""Record a community rating for an image: embed once, store it.

This is the write side of the loop that train.py reads from -- each call
embeds the image (if needed) and stores one (image_id, user_id, rating,
timestamp) row plus the image's embedding in the ratings database.

Mostly a manual / debugging tool now that the web app is the real rating
surface (its votes reach this store via horse.ingest).

Run:

    python -m horse.rate path/to/image.jpg 4.5 [--user-id someone] [--db horse/data/ratings.db]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from horse.embed import get_embedder
from horse.storage import add_rating, connect

DEFAULT_DB_PATH = "horse/data/ratings.db"
DEFAULT_USER_ID = "cli"


def record_rating(
    image_path: str | Path,
    rating: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    image_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    embedder=None,
    embedding: np.ndarray | None = None,
) -> str:
    """Embed image_path (unless embedding is already known) and store the rating.

    Returns the image_id used. Re-rating the same (image_id, user_id)
    overwrites that vote and refreshes its timestamp.
    """
    image_id = image_id or Path(image_path).stem
    if embedding is None:
        embedder = embedder or get_embedder()
        embedding = embedder.embed_path(image_path)

    conn = connect(db_path)
    add_rating(conn, image_id, user_id, embedding, rating)
    return image_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a community rating for an image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("rating", type=float, help="Horseness rating, e.g. on a 0-5 scale.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to the ratings database.")
    parser.add_argument("--image-id", default=None, help="Defaults to the image's filename stem.")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Who cast this rating.")
    args = parser.parse_args()

    image_id = record_rating(
        args.image, args.rating, db_path=args.db, image_id=args.image_id, user_id=args.user_id
    )
    print(f"[rate] {image_id} = {args.rating} (by {args.user_id}) -> {args.db}")


if __name__ == "__main__":
    main()
