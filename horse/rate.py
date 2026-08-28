"""Record a community rating for an image: embed once, store it.

This is the write side of the loop that train.py reads from -- each call
embeds the image (if needed) and appends one (embedding, rating,
timestamp, image_id) row to the ratings database.

Run:

    python -m horse.rate path/to/image.jpg 4.5 [--db horse/data/ratings.db]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from horse.embed import ClipEmbedder
from horse.storage import add_rating, connect

DEFAULT_DB_PATH = "horse/data/ratings.db"


def record_rating(
    image_path: str | Path,
    rating: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    image_id: str | None = None,
    embedder: ClipEmbedder | None = None,
    embedding: np.ndarray | None = None,
) -> str:
    """Embed image_path (unless embedding is already known) and store the rating.

    Returns the image_id used. Rating an image_id that's already stored
    overwrites its previous rating and refreshes its timestamp.
    """
    image_id = image_id or Path(image_path).stem
    if embedding is None:
        embedder = embedder or ClipEmbedder()
        embedding = embedder.embed_path(image_path)

    conn = connect(db_path)
    add_rating(conn, image_id, embedding, rating)
    return image_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a community rating for an image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("rating", type=float, help="Horseness rating, e.g. on a 0-5 scale.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to the ratings database.")
    parser.add_argument("--image-id", default=None, help="Defaults to the image's filename stem.")
    args = parser.parse_args()

    image_id = record_rating(args.image, args.rating, db_path=args.db, image_id=args.image_id)
    print(f"[rate] {image_id} = {args.rating} -> {args.db}")


if __name__ == "__main__":
    main()
