"""Account endpoints: register, login, whoami.

    POST /api/auth/register   {username, password, invite}  -> {token, user}
    POST /api/auth/login      {username, password}  -> {token, user}
    GET  /api/auth/me         (Bearer)              -> {user}

`invite` is required when config REQUIRE_INVITE is on (see services/invites.py).
"""

from __future__ import annotations

import sqlite3
import time

from flask import Blueprint, current_app, g, jsonify, request

from ..db import get_db
from ..errors import ApiError
from ..security import (
    BANNED_MESSAGE,
    hash_password,
    make_token,
    require_auth,
    verify_password,
)
from ..services import invites

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

    invite = None
    if current_app.config["REQUIRE_INVITE"]:
        invite = invites.normalize((request.get_json(silent=True) or {}).get("invite"))
        if not invite:
            raise ApiError("you need an invite code to make an account", 403)

    db = get_db()
    try:
        if invite:
            invites.claim(db, invite)
        cur = db.execute(
            "INSERT INTO users (username, password_hash, created_at, invite_code) "
            "VALUES (?, ?, ?, ?)",
            (username, hash_password(password), time.time(), invite),
        )
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()  # also gives back the invite use
        raise ApiError("username already taken", 409)
    except ApiError:
        db.rollback()
        raise

    user = {"id": cur.lastrowid, "username": username}
    return jsonify(token=make_token(user["id"]), user=user), 201


@bp.post("/login")
def login():
    username, password = _credentials()
    row = get_db().execute(
        "SELECT id, username, password_hash, banned FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        raise ApiError("incorrect username or password", 401)
    if row["banned"]:
        raise ApiError(BANNED_MESSAGE, 403)

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
