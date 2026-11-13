"""Image upload + file serving.

    POST /api/images              (Bearer, multipart 'image' + optional 'caption')
    GET  /api/images/limits       -> {max_bytes, max_pixels}   so the client can pre-check
    GET  /api/images/mine         (Bearer)  -> {items: [...]}  the caller's own uploads
    GET  /api/images/<id>/file                                 -> the stored image bytes

Only images that are showing have a file to serve. `?review=<token>` (from
`admin review-link`) also opens one hidden post, for the operator to judge.
"""

from __future__ import annotations

import os

from flask import Blueprint, current_app, g, jsonify, request, send_from_directory

from ..errors import ApiError
from ..security import require_auth, review_token_allows
from ..services import images as images_service

bp = Blueprint("images", __name__, url_prefix="/api")


@bp.post("/images")
@require_auth
def upload():
    file = request.files.get("image")
    if file is None or not file.filename:
        raise ApiError("no image uploaded (multipart form field 'image')")
    caption = request.form.get("caption")
    return jsonify(images_service.upload_image(file, g.user_id, caption)), 201


@bp.get("/images/limits")
def upload_limits():
    # The server still enforces these; the client only checks to avoid
    # sending a file that is bound to be refused.
    cfg = current_app.config
    return jsonify(max_bytes=cfg["MAX_CONTENT_LENGTH"], max_pixels=cfg["MAX_UPLOAD_PIXELS"])


@bp.get("/images/mine")
@require_auth
def my_images():
    return jsonify(items=images_service.list_own_images(g.user_id))


@bp.get("/images/<image_id>/file")
def image_file(image_id: str):
    meta = images_service.get_image_meta(image_id)  # 404s if unknown
    reviewing = review_token_allows(request.args.get("review"), image_id)
    if not reviewing and (not meta["passed"] or meta["hidden_at"] is not None):
        raise ApiError("image not found", 404)
    # abspath: send_from_directory resolves a relative dir against the app
    # root (server/), not the cwd where UPLOAD_DIR is configured.
    # Explicit mimetype: the stored format is one of ours (services/images.py
    # STORED_EXTENSIONS), and Windows' mimetype table doesn't know .webp.
    response = send_from_directory(
        os.path.abspath(current_app.config["UPLOAD_DIR"]),
        f"{image_id}{meta['ext']}",
        mimetype=images_service.STORED_MIMETYPES.get(meta["ext"]),
    )
    if reviewing:
        response.headers["Cache-Control"] = "no-store"
    return response
