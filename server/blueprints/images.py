"""Image upload + file serving.

    POST /api/images              (Bearer, multipart 'image' + optional 'caption')
    GET  /api/images/mine         (Bearer)  -> {items: [...]}  the caller's own uploads
    GET  /api/images/<id>/file                                 -> the stored image bytes
"""

from __future__ import annotations

import os

from flask import Blueprint, current_app, g, jsonify, request, send_from_directory

from ..errors import ApiError
from ..security import require_auth
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


@bp.get("/images/mine")
@require_auth
def my_images():
    return jsonify(items=images_service.list_own_images(g.user_id))


@bp.get("/images/<image_id>/file")
def image_file(image_id: str):
    meta = images_service.get_image_meta(image_id)  # 404s if unknown
    # abspath: send_from_directory resolves a relative dir against the app
    # root (server/), not the cwd where UPLOAD_DIR is configured.
    return send_from_directory(
        os.path.abspath(current_app.config["UPLOAD_DIR"]), f"{image_id}{meta['ext']}"
    )
