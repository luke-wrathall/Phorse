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
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

from horse.gate import passes_gate

from ..db import get_db
from ..errors import ApiError
from . import model

CAPTION_MAX = 280
# Counter-clockwise degrees tried at upload; 0 must come first (upright).
# 180 kept on purpose: an upside-down horse that scores its way in is funny.
ROTATIONS = (0, 90, 180, 270)


def _clean_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    caption = caption.strip()[:CAPTION_MAX]
    return caption or None


def _best_rotation(image: Image.Image) -> tuple[Image.Image, np.ndarray, float]:
    """Score the image at each of ROTATIONS; use a rotation only when it is
    what gets the image through the gate.

    Phone photos of drawings often arrive sideways with no EXIF tag, and the
    head reads a sideways horse as much less of a horse (a pen drawing:
    1.27 upright vs 2.90 rotated). A rotation is kept only if upright fails
    the gate, the rotation passes, and it beats upright by
    UPLOAD_ROTATION_MARGIN -- otherwise the image is stored as sent, so a
    drifted-down horse photo isn't saved upside down for a small gain.
    Raises false passes slightly (held-out PACS non-horse sketches
    7% -> 13%, other styles 0% -> 3%).
    """
    threshold = current_app.config["GATE_THRESHOLD"]
    margin = current_app.config["UPLOAD_ROTATION_MARGIN"]
    candidates = []
    for angle in ROTATIONS:
        rotated = image if angle == 0 else image.rotate(angle, expand=True)
        embedding = model.embed_image(rotated)
        candidates.append((rotated, embedding, model.score_embedding(embedding)))
    upright = candidates[0]
    best = max(candidates, key=lambda c: c[2])
    rescues = (
        not passes_gate(upright[2], threshold)
        and passes_gate(best[2], threshold)
        and best[2] >= upright[2] + margin
    )
    return best if rescues else upright


def upload_image(file: FileStorage, owner_id: int, caption: str | None = None) -> dict:
    """Embed + score + gate one uploaded image, store it, return the verdict.

    The stored file, embedding and score are all of the best-scoring
    rotation (see `_best_rotation`), so the feed shows it the right way up.
    """
    try:
        image = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- report any bad-upload reason
        raise ApiError(f"could not read image: {exc}", 400)

    image, embedding, score = _best_rotation(image)
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
