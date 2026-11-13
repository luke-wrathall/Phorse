"""Report a post, or complain about anything else. No login needed.

    POST /api/reports   {reason, image_id?, details?, contact?}  -> {ok: true}

`reason` is one of services/reports.py REASONS. Without `image_id` it's a
general complaint, and `details` is required. The response never says
whether the report hid the post.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..db import get_db
from ..security import optional_user_id
from ..services import limits
from ..services import reports as reports_service

bp = Blueprint("reports", __name__, url_prefix="/api")


@bp.post("/reports")
def report():
    data = request.get_json(silent=True) or {}
    reports_service.submit_report(
        get_db(),
        reason=data.get("reason"),
        image_id=data.get("image_id") or None,
        details=data.get("details"),
        contact=data.get("contact"),
        user_id=optional_user_id(),
        ip=limits.client_ip(),
    )
    return jsonify(ok=True), 201
