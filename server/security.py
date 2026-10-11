"""Password hashing + JWT bearer-token auth.

Token-based from day one (CLAUDE.md "Web app architecture"): the SvelteKit
SPA and this API are cross-origin, so there is no session cookie. Clients
send `Authorization: Bearer <jwt>`; `@require_auth` decodes it and puts the
user id on `flask.g.user_id`.
"""

from __future__ import annotations

import time
from functools import wraps

import jwt
from flask import current_app, g, request
from werkzeug.security import check_password_hash, generate_password_hash

from .errors import ApiError


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def make_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + current_app.config["TOKEN_TTL_SECONDS"],
    }
    return jwt.encode(payload, current_app.config["SECRET_KEY"], algorithm="HS256")


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(
            token, current_app.config["SECRET_KEY"], algorithms=["HS256"]
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise ApiError("invalid or expired token", 401)


def optional_user_id() -> int | None:
    """The caller's user id if a valid bearer token is present, else None.
    For endpoints that work logged out but personalize when logged in."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    try:
        return decode_token(header[len("Bearer ") :].strip())
    except ApiError:
        return None


def require_auth(view):
    """Decorator: 401 unless a valid bearer token is present. Sets g.user_id."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise ApiError("missing bearer token", 401)
        g.user_id = decode_token(header[len("Bearer ") :].strip())
        return view(*args, **kwargs)

    return wrapper
