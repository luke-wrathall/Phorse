"""Business logic for uploads: embed -> score -> gate -> persist.

Kept out of the route handler (CLAUDE.md: "keep business logic separate
from the API layer"). The route just parses the request and calls
`upload_image`.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from flask import current_app
from PIL import Image
from werkzeug.datastructures import FileStorage

from horse.gate import passes_gate

from ..db import get_db
from ..errors import ApiError
from . import model

CAPTION_MAX = 280


def _clean_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    caption = caption.strip()[:CAPTION_MAX]
    return caption or None


def upload_image(file: FileStorage, owner_id: int, caption: str | None = None) -> dict:
    """Embed + score + gate one uploaded image, store it, return the verdict."""
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- report any bad-upload reason
        raise ApiError(f"could not read image: {exc}", 400)

    embedding = model.embed_image(image)
    score = model.score_embedding(embedding)
    threshold = current_app.config["GATE_THRESHOLD"]
    passed = passes_gate(score, threshold)

    image_id = uuid.uuid4().hex
    ext = (Path(file.filename or "").suffix or ".png").lower()
    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    image.save(upload_dir / f"{image_id}{ext}")

    db = get_db()
    db.execute(
        "INSERT INTO images "
        "(id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            image_id,
            owner_id,
            file.filename,
            ext,
            _clean_caption(caption),
            embedding.astype(np.float32).tobytes(),
            float(score),
            int(passed),
            time.time(),
        ),
    )
    db.commit()

    return {
        "id": image_id,
        "score": score,
        "passed": passed,
        "threshold": threshold,
        "head_loaded": model.head_loaded(),
    }


def get_image_meta(image_id: str) -> dict:
    row = get_db().execute(
        "SELECT id, ext, score, passed, created_at FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if row is None:
        raise ApiError("image not found", 404)
    return dict(row)


def list_own_images(owner_id: int, limit: int = 50) -> list[dict]:
    """Every image this user has uploaded, newest first -- passed or not."""
    rows = get_db().execute(
        """
        SELECT i.id,
               i.caption,
               i.passed,
               i.created_at,
               COUNT(r.rating) AS rating_count,
               AVG(r.rating)   AS rating_mean
        FROM images i
        LEFT JOIN ratings r ON r.image_id = i.id
        WHERE i.owner_id = ?
        GROUP BY i.id
        ORDER BY i.created_at DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "caption": row["caption"],
            "passed": bool(row["passed"]),
            "created_at": row["created_at"],
            "rating_count": row["rating_count"],
            "rating_mean": row["rating_mean"],
            "image_url": f"/api/images/{row['id']}/file",
        }
        for row in rows
    ]
