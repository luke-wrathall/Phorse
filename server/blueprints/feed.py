"""Feed endpoint.

    GET /api/feed?limit=20&before=<timestamp>  -> {items: [...], next_before: <ts|null>}

`before` is the keyset cursor: pass back the previous response's
`next_before` to fetch the next page (null means end of feed).
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..errors import ApiError
from ..services import feed as feed_service

bp = Blueprint("feed", __name__, url_prefix="/api")


@bp.get("/feed")
def feed():
    try:
        limit = int(request.args.get("limit", feed_service.DEFAULT_LIMIT))
        before = request.args.get("before", type=float)
    except (TypeError, ValueError):
        raise ApiError("limit must be an integer and before a number")
    return jsonify(feed_service.get_feed(limit=limit, before=before))
