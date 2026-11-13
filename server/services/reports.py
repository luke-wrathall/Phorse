"""Reports and complaints, from anyone, logged in or not.

A report about a post can hide it without waiting for the operator. Two
kinds hide it at once, because the harm of leaving them up for even a few
hours outweighs a wrongly hidden horse. Everything else hides after
REPORT_AUTOHIDE different people report the same post. Hiding is always
reversible (`python -m server.admin restore`), and every report goes to the
queue for a person to decide.
"""

from __future__ import annotations

import sqlite3
import time

from flask import current_app

from ..errors import ApiError
from . import alerts, limits

# The web form's list (web/src/routes/report/+page.svelte) must match.
REASONS = (
    "child_safety",
    "intimate_image",
    "illegal",
    "harmful",
    "hate",
    "copyright",
    "appeal",
    "other",
)
HIDE_AT_ONCE = ("child_safety", "intimate_image")
URGENT = HIDE_AT_ONCE + ("illegal",)

DETAILS_MAX = 2000
CONTACT_MAX = 200


def _clean(text: object, limit: int) -> str | None:
    if not isinstance(text, str):
        return None
    return text.strip()[:limit] or None


def submit_report(
    db: sqlite3.Connection,
    reason: object,
    image_id: object = None,
    details: object = None,
    contact: object = None,
    user_id: int | None = None,
    ip: str = "unknown",
) -> dict:
    """File a report. A second report of the same post by the same person
    changes nothing, and looks the same to them. Returns {"hidden": bool}."""
    if reason not in REASONS:
        raise ApiError(f"reason must be one of {', '.join(REASONS)}")
    details = _clean(details, DETAILS_MAX)
    contact = _clean(contact, CONTACT_MAX)
    if image_id is None and not details:
        raise ApiError("tell us what's wrong")
    if image_id is not None:
        if not isinstance(image_id, str):
            raise ApiError("post not found", 404)
        exists = db.execute(
            "SELECT 1 FROM images WHERE id = ? AND passed = 1", (image_id,)
        ).fetchone()
        if exists is None:
            raise ApiError("post not found", 404)

    cfg = current_app.config
    reporter_key = limits.client_key(f"user:{user_id}" if user_id is not None else f"ip:{ip}")
    limits.hit(
        db,
        "report",
        reporter_key,
        cfg["REPORTS_PER_REPORTER_DAY"],
        limits.DAY,
        "that's a lot of reports for one day, try again tomorrow",
    )
    filed = db.execute(
        "INSERT OR IGNORE INTO reports "
        "(image_id, reason, details, contact, reporter_key, reporter_user_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (image_id, reason, details, contact, reporter_key, user_id, time.time()),
    ).rowcount
    db.commit()
    if not filed:
        return {"hidden": False}

    hidden = image_id is not None and _maybe_hide(db, image_id, reason)
    open_count = db.execute("SELECT COUNT(*) FROM reports WHERE resolved_at IS NULL").fetchone()[0]
    what = f"post {image_id}" if image_id else "no specific post"
    alerts.notify(
        db,
        f"new report: {reason}",
        f"reason: {reason}\nabout: {what}\n"
        f"{'the post has been hidden' if hidden else 'still visible'}\n"
        f"{open_count} open in total.\n\n"
        "on the server: bash /srv/phorse/deploy/manage.sh admin reports\n",
        urgent=reason in URGENT,
    )
    return {"hidden": hidden}


def _maybe_hide(db: sqlite3.Connection, image_id: str, reason: str) -> bool:
    if reason == "appeal":  # someone asking for a post back
        return False
    if reason not in HIDE_AT_ONCE:
        threshold = current_app.config["REPORT_AUTOHIDE"]
        reporters = db.execute(
            "SELECT COUNT(DISTINCT reporter_key) FROM reports "
            "WHERE image_id = ? AND resolved_at IS NULL AND reason != 'appeal'",
            (image_id,),
        ).fetchone()[0]
        if threshold <= 0 or reporters < threshold:
            return False
    hid = db.execute(
        "UPDATE images SET hidden_at = ?, hidden_reason = 'reported' "
        "WHERE id = ? AND hidden_at IS NULL",
        (time.time(), image_id),
    ).rowcount
    db.commit()
    return bool(hid)


def resolve_reports(db: sqlite3.Connection, image_id: str, resolution: str) -> int:
    """Close every open report about a post."""
    closed = db.execute(
        "UPDATE reports SET resolved_at = ?, resolution = ? "
        "WHERE image_id = ? AND resolved_at IS NULL",
        (time.time(), resolution, image_id),
    ).rowcount
    db.commit()
    return closed


def resolve_report(db: sqlite3.Connection, report_id: int, resolution: str) -> bool:
    closed = db.execute(
        "UPDATE reports SET resolved_at = ?, resolution = ? WHERE id = ? AND resolved_at IS NULL",
        (time.time(), resolution, report_id),
    ).rowcount
    db.commit()
    return bool(closed)


def list_reports(db: sqlite3.Connection, include_resolved: bool = False, limit: int = 50) -> list:
    where = "" if include_resolved else "WHERE r.resolved_at IS NULL"
    return db.execute(
        f"""
        SELECT r.id, r.image_id, r.reason, r.details, r.contact, r.created_at,
               r.resolved_at, r.resolution,
               reporter.username AS reporter,
               owner.username AS owner,
               i.hidden_at, i.quarantined_at
        FROM reports r
        LEFT JOIN users reporter ON reporter.id = r.reporter_user_id
        LEFT JOIN images i ON i.id = r.image_id
        LEFT JOIN users owner ON owner.id = i.owner_id
        {where}
        ORDER BY r.created_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()


def has_open_child_safety_report(db: sqlite3.Connection, image_id: str) -> bool:
    return db.execute(
        "SELECT 1 FROM reports WHERE image_id = ? AND reason = 'child_safety' "
        "AND resolved_at IS NULL",
        (image_id,),
    ).fetchone() is not None
