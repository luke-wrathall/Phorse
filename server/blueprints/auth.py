"""Account endpoints: register, login, whoami.

    POST /api/auth/register   {username, password}  -> {token, user}
    POST /api/auth/login      {username, password}  -> {token, user}
    GET  /api/auth/me         (Bearer)              -> {user}
"""

from __future__ import annotations

import sqlite3
import time

from flask import Blueprint, g, jsonify, request

from ..db import get_db
from ..errors import ApiError
from ..security import hash_password, make_token, require_auth, verify_password

bp = Blueprint("auth", __name__, url_prefix="/api/auth")

MIN_USERNAME = 3
MIN_PASSWORD = 8


def _credentials() -> tuple[str, str]:
    data = request.get_json(silent=True) or {}
    return (data.get("username") or "").strip(), data.get("password") or ""


@bp.post("/register")
def register():
    username, password = _credentials()
    if len(username) < MIN_USERNAME or len(password) < MIN_PASSWORD:
        raise ApiError(
            f"username must be {MIN_USERNAME}+ chars and password {MIN_PASSWORD}+ chars"
        )

    db = get_db()
    try:
        cur = db.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), time.time()),
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise ApiError("username already taken", 409)

    user = {"id": cur.lastrowid, "username": username}
    return jsonify(token=make_token(user["id"]), user=user), 201


@bp.post("/login")
def login():
    username, password = _credentials()
    row = get_db().execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        raise ApiError("incorrect username or password", 401)

    user = {"id": row["id"], "username": row["username"]}
    return jsonify(token=make_token(user["id"]), user=user)


@bp.get("/me")
@require_auth
def me():
    row = get_db().execute(
        "SELECT id, username FROM users WHERE id = ?", (g.user_id,)
    ).fetchone()
    if row is None:
        raise ApiError("user no longer exists", 401)
    return jsonify(user=dict(row))
