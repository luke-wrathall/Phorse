"""Invite codes: the only way to make an account when REQUIRE_INVITE is on.

A code can be shared by many people (e.g. one link in an Instagram story)
and stops working after `max_uses` signups or when revoked. Each account
remembers which code it came in on, so a leaked code can be revoked and
the accounts it let in found (`users.invite_code`).

Codes are created and revoked from the command line (`python -m
server.invites`), not over the API.
"""

from __future__ import annotations

import secrets
import sqlite3
import time

from ..errors import ApiError

# No 0/o, 1/l/i: codes get read off a phone screen and typed in.
_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
CODE_LENGTH = 8
MAX_CODE_LEN = 64


def normalize(code: str | None) -> str:
    return (code or "").strip().lower()


def _generate_code() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(CODE_LENGTH))


def create_invite(
    db: sqlite3.Connection,
    max_uses: int | None,
    note: str | None = None,
    code: str | None = None,
) -> str:
    """Store a new invite and return its code. `max_uses=None` = unlimited."""
    code = normalize(code) or _generate_code()
    if len(code) > MAX_CODE_LEN or not code.replace("-", "").isalnum():
        raise ValueError(f"codes are letters, digits and '-', at most {MAX_CODE_LEN} long")
    if max_uses is not None and max_uses < 1:
        raise ValueError("max_uses must be at least 1")
    try:
        db.execute(
            "INSERT INTO invites (code, max_uses, note, created_at) VALUES (?, ?, ?, ?)",
            (code, max_uses, note, time.time()),
        )
    except sqlite3.IntegrityError:
        raise ValueError(f"invite code {code!r} already exists")
    db.commit()
    return code


def list_invites(db: sqlite3.Connection) -> list[dict]:
    rows = db.execute(
        "SELECT code, max_uses, uses, note, revoked, created_at FROM invites "
        "ORDER BY created_at"
    ).fetchall()
    return [dict(row) for row in rows]


def revoke_invite(db: sqlite3.Connection, code: str) -> bool:
    """Stop a code working for new signups. Existing accounts are untouched."""
    cur = db.execute("UPDATE invites SET revoked = 1 WHERE code = ?", (normalize(code),))
    db.commit()
    return cur.rowcount > 0


def claim(db: sqlite3.Connection, code: str) -> None:
    """Use up one signup on `code`, or raise. Doesn't commit: call it in the
    same transaction as the user INSERT, so a failed signup (username
    taken) rolls the use back too."""
    cur = db.execute(
        "UPDATE invites SET uses = uses + 1 "
        "WHERE code = ? AND revoked = 0 AND (max_uses IS NULL OR uses < max_uses)",
        (code,),
    )
    if cur.rowcount == 0:
        # One message for unknown / revoked / used up: no probing for codes.
        raise ApiError("that invite has run out or isn't valid any more", 403)
