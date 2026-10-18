"""Feed endpoint.

    GET /api/feed?feed=new&limit=20&offset=0&session=<id>  -> {items: [...], next_offset: <int|null>}

`feed` is "new" (default: recent horses, ranked by config FEED_STRATEGY)
or "ranked" (every horse, most horse first) -- see services/feed.py. Both
are ranked, so pagination is by offset: pass back the previous
response's `next_offset` for the next page (null means end of feed).
`session` (optional, client-generated once per page load) keeps already
loaded items fixed in place for that page load -- see services/feed.py.

Auth is optional: with a bearer token, strategies that mix in "horses you
haven't rated" use the caller's ratings; logged out, nothing counts as rated.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..errors import ApiError
from ..security import optional_user_id
from ..services import feed as feed_service

bp = Blueprint("feed", __name__, url_prefix="/api")


@bp.get("/feed")
def feed():
    try:
        limit = int(request.args.get("limit", feed_service.DEFAULT_LIMIT))
        offset = int(request.args.get("offset", 0))
    except (TypeError, ValueError):
        raise ApiError("limit and offset must be integers")
    session = request.args.get("session") or None
    if session and len(session) > feed_service.SESSION_ID_MAX_LEN:
        raise ApiError("session id too long")
    kind = request.args.get("feed", "new")
    if kind not in feed_service.FEEDS:
        raise ApiError(f"feed must be one of {', '.join(feed_service.FEEDS)}")
    return jsonify(
        feed_service.get_feed(
            limit=limit, offset=offset, user_id=optional_user_id(), session=session, kind=kind
        )
    )
