"""Feed endpoint.

    GET /api/feed?limit=20&offset=0  -> {items: [...], next_offset: <int|null>}

The feed is ranked (three-pool blend, see services/feed.py), so pagination
is by offset: pass back the previous response's `next_offset` for the next
page (null means end of feed).
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
        offset = int(request.args.get("offset", 0))
    except (TypeError, ValueError):
        raise ApiError("limit and offset must be integers")
    return jsonify(feed_service.get_feed(limit=limit, offset=offset))
