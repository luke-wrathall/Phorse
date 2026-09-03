"""Submit a community rating.

    POST /api/images/<id>/ratings   (Bearer)  {rating: 0..5}  -> {ok, rating_count, rating_mean}

Re-rating the same image replaces your previous rating (one live rating
per user per image -- see the ratings table PK in server/db.py).
"""

from __future__ import annotations

import time

from flask import Blueprint, g, jsonify, request

from ..db import get_db
from ..errors import ApiError
from ..security import require_auth

bp = Blueprint("ratings", __name__, url_prefix="/api")


@bp.post("/images/<image_id>/ratings")
@require_auth
def rate(image_id: str):
    data = request.get_json(silent=True) or {}
    try:
        value = float(data["rating"])
    except (KeyError, TypeError, ValueError):
        raise ApiError('body must be {"rating": <number 0-5>}')
    if not 0.0 <= value <= 5.0:
        raise ApiError("rating must be between 0 and 5")

    db = get_db()
    if db.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is None:
        raise ApiError("image not found", 404)

    db.execute(
        "INSERT INTO ratings (image_id, user_id, rating, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(image_id, user_id) DO UPDATE SET "
        "rating = excluded.rating, created_at = excluded.created_at",
        (image_id, g.user_id, value, time.time()),
    )
    db.commit()

    agg = db.execute(
        "SELECT COUNT(*) AS n, AVG(rating) AS mean FROM ratings WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    return jsonify(ok=True, rating_count=agg["n"], rating_mean=agg["mean"])
