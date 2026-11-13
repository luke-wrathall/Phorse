"""Business logic for uploads: limits -> embed -> score -> gate -> content
check -> persist.

Kept out of the route handler (CLAUDE.md: "keep business logic separate
from the API layer"). The route just parses the request and calls
`upload_image`.

A rejected upload's file is never kept: nobody would ever see it, so
keeping it would only leave unreviewed pictures on the server and in its
backups. Its row stays, with the embedding and score, because tuning and
the uploader's page use those.
"""

from __future__ import annotations

import hashlib
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
from . import alerts, limits, model, safety

CAPTION_MAX = 280
# Counter-clockwise degrees tried at upload; 0 must come first (upright).
# 180 kept on purpose: an upside-down horse that scores its way in is funny.
ROTATIONS = (0, 90, 180, 270)


def _clean_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    caption = caption.strip()[:CAPTION_MAX]
    return caption or None


# The stored file is re-encoded from the decoded (EXIF-fixed, rotated, RGB)
# image, so its format is ours to pick -- never the uploader's extension.
# PIL chooses the encoder by extension, so a PNG called `horse.exe` used to
# crash the upload and `horse.pdf` was saved and served as a real PDF.
STORED_EXTENSIONS = {".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg", ".webp": ".webp"}
DEFAULT_STORED_EXTENSION = ".jpg"
STORED_MIMETYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}
JPEG_QUALITY = 90


def _stored_extension(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return STORED_EXTENSIONS.get(suffix, DEFAULT_STORED_EXTENSION)


def _best_rotation(
    image: Image.Image,
) -> tuple[tuple[Image.Image, np.ndarray, float], list[tuple[int, np.ndarray]]]:
    """Score the image at each of ROTATIONS; use a rotation only when it is
    what gets the image through the gate.

    Returns the chosen (image, embedding, score) and the other rotations'
    embeddings as (angle, embedding), with the angle relative to the chosen
    orientation. Those are kept as training augmentations (image_augments):
    each vote on the stored image also trains the head on them, at
    config AUGMENT_WEIGHT -- see horse.train.

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
    upright = 0
    best = max(range(len(candidates)), key=lambda i: candidates[i][2])
    rescues = (
        not passes_gate(candidates[upright][2], threshold)
        and passes_gate(candidates[best][2], threshold)
        and candidates[best][2] >= candidates[upright][2] + margin
    )
    chosen = best if rescues else upright
    others = [
        ((ROTATIONS[i] - ROTATIONS[chosen]) % 360, candidates[i][1])
        for i in range(len(candidates))
        if i != chosen
    ]
    return candidates[chosen], others


def _check_upload_limits(db, owner_id: int) -> None:
    cfg = current_app.config
    limits.require_open(db, "uploads")
    blocks = db.execute(
        "SELECT COUNT(*) FROM upload_blocks WHERE user_id = ? AND created_at > ?",
        (owner_id, time.time() - limits.DAY),
    ).fetchone()[0]
    if 0 < cfg["UPLOAD_BLOCKS_PER_USER_DAY"] <= blocks:
        raise ApiError("too many pictures broke the rules, try again tomorrow", 429)
    too_many = "that's a lot of horses for one day, try again tomorrow"
    limits.hit(db, "upload-user", str(owner_id), cfg["UPLOADS_PER_USER_DAY"], limits.DAY, too_many)
    limits.hit(
        db,
        "upload-ip",
        limits.client_key(limits.client_ip()),
        cfg["UPLOADS_PER_IP_DAY"],
        limits.DAY,
        too_many,
    )


def _record_block(db, file: FileStorage, owner_id: int, categories: list[str]) -> None:
    file.stream.seek(0)
    db.execute(
        "INSERT INTO upload_blocks (user_id, filename, sha256, categories, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            owner_id,
            file.filename,
            hashlib.sha256(file.stream.read()).hexdigest(),
            ",".join(categories),
            time.time(),
        ),
    )
    db.commit()
    alerts.notify(
        db,
        "upload blocked by the content check",
        f"categories: {', '.join(categories)}\n\n"
        "on the server: bash /srv/phorse/deploy/manage.sh admin blocks\n",
    )


def upload_image(file: FileStorage, owner_id: int, caption: str | None = None) -> dict:
    """Embed + score + gate one uploaded image, store it if it passed, return
    the verdict.

    The stored file, embedding and score are all of the best-scoring
    rotation (see `_best_rotation`), so the feed shows it the right way up.
    """
    db = get_db()
    _check_upload_limits(db, owner_id)
    try:
        opened = Image.open(file.stream)  # lazy: reads the header, not the pixels
    except Exception as exc:  # noqa: BLE001 -- report any bad-upload reason
        raise ApiError(f"could not read image: {exc}", 400)
    max_pixels = current_app.config["MAX_UPLOAD_PIXELS"]
    if opened.width * opened.height > max_pixels:
        raise ApiError(
            f"image too big -- {opened.width}x{opened.height} is over the "
            f"{max_pixels / 1_000_000:g} megapixel limit",
            413,
        )
    try:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- report any bad-upload reason
        raise ApiError(f"could not read image: {exc}", 400)

    (image, embedding, score), other_rotations = _best_rotation(image)
    threshold = current_app.config["GATE_THRESHOLD"]
    passed = passes_gate(score, threshold)
    caption = _clean_caption(caption)

    if passed:
        blocked = safety.blocked_categories(image, caption)
        if blocked:
            _record_block(db, file, owner_id, blocked)
            raise ApiError(
                "this one can't go up, it looks like it breaks the rules. "
                "if we got that wrong, tell us at /report",
                422,
            )

    image_id = uuid.uuid4().hex
    ext = _stored_extension(file.filename)
    if passed:
        upload_dir = Path(current_app.config["UPLOAD_DIR"])
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_options = {"quality": JPEG_QUALITY} if ext in (".jpg", ".webp") else {}
        image.save(upload_dir / f"{image_id}{ext}", **save_options)

    db.execute(
        "INSERT INTO images "
        "(id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            image_id,
            owner_id,
            file.filename,
            ext,
            caption,
            embedding.astype(np.float32).tobytes(),
            float(score),
            int(passed),
            time.time(),
        ),
    )
    if passed:
        # Only showing posts can be voted on, so only they need augmentations.
        db.executemany(
            "INSERT INTO image_augments (image_id, angle, embedding) VALUES (?, ?, ?)",
            [
                (image_id, angle, other.astype(np.float32).tobytes())
                for angle, other in other_rotations
            ],
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
        "SELECT id, ext, score, passed, hidden_at, created_at FROM images WHERE id = ?",
        (image_id,),
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
               i.hidden_at,
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
            # The uploader sees the same thing whatever the reason.
            "removed": row["hidden_at"] is not None,
            "created_at": row["created_at"],
            "rating_count": row["rating_count"],
            "rating_mean": row["rating_mean"],
            "image_url": (
                f"/api/images/{row['id']}/file"
                if row["passed"] and row["hidden_at"] is None
                else None
            ),
        }
        for row in rows
    ]
