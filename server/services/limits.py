"""Brakes against floods: rate limits, and pause switches for the whole site.

There's one moderator, so these keep a flood down to what one person can
look through. Both live in the app db, not in memory: limits survive a
restart, and `python -m server.admin pause` reaches the API at once.

IP addresses are stored only as keyed hashes (`client_key`), and rate
records are dropped after two days.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import time

from flask import current_app, request

from ..errors import ApiError

HOUR = 3600
DAY = 24 * HOUR
KEEP_SECONDS = 2 * DAY

PAUSABLE = ("uploads", "signups")


def client_ip() -> str:
    # Behind Caddy, ProxyFix (server/__init__.py) has already put the real
    # client address here.
    return request.remote_addr or "unknown"


def client_key(value: str) -> str:
    """A stable stand-in for an IP address or user id: tells two clients
    apart without storing who they are."""
    secret = current_app.config["SECRET_KEY"].encode()
    return hmac.new(secret, value.encode(), hashlib.sha256).hexdigest()


def allow(db: sqlite3.Connection, kind: str, key: str, limit: int, window_seconds: int) -> bool:
    """Record one event and return True, unless `key` already had `limit` of
    these within the window. A limit of 0 means no limit."""
    if limit <= 0:
        return True
    now = time.time()
    db.execute("DELETE FROM rate_events WHERE created_at < ?", (now - KEEP_SECONDS,))
    count = db.execute(
        "SELECT COUNT(*) FROM rate_events WHERE kind = ? AND key = ? AND created_at > ?",
        (kind, key, now - window_seconds),
    ).fetchone()[0]
    allowed = count < limit
    if allowed:
        db.execute(
            "INSERT INTO rate_events (kind, key, created_at) VALUES (?, ?, ?)", (kind, key, now)
        )
    db.commit()
    return allowed


def hit(
    db: sqlite3.Connection, kind: str, key: str, limit: int, window_seconds: int, message: str
) -> None:
    """`allow`, but past the limit the request fails with a 429."""
    if not allow(db, kind, key, limit, window_seconds):
        raise ApiError(message, 429)


def is_paused(db: sqlite3.Connection, what: str) -> bool:
    return db.execute("SELECT 1 FROM pauses WHERE what = ?", (what,)).fetchone() is not None


def set_paused(db: sqlite3.Connection, what: str, paused: bool) -> None:
    if paused:
        db.execute(
            "INSERT OR IGNORE INTO pauses (what, since) VALUES (?, ?)", (what, time.time())
        )
    else:
        db.execute("DELETE FROM pauses WHERE what = ?", (what,))
    db.commit()


def require_open(db: sqlite3.Connection, what: str) -> None:
    if is_paused(db, what):
        raise ApiError(f"{what} are paused for now, try again later", 503)
