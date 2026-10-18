"""Operator moderation: delete posts, ban users, reset passwords.

Run from the command line (`python -m server.admin`), never over the API.

Deleting anything rated has to reach the training store too: ingest only
adds and updates rows there (horse/ingest.py), so a deleted post's ratings,
or a purged user's votes, would otherwise keep training the head.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time

from flask import current_app

from horse.storage import connect as connect_training_db

# Image ids are uuid4 hex. Also pulled out of a pasted image URL
# (.../api/images/<id>/file), which is what you can copy from the site.
_IMAGE_ID = re.compile(r"[0-9a-f]{32}")


def parse_image_id(text: str) -> str | None:
    match = _IMAGE_ID.search(text.lower())
    return match.group(0) if match else None


def find_user(db: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return db.execute(
        "SELECT id, username, banned FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()


def _forget_training_ratings(image_ids: list[str], user_id: int | None = None) -> None:
    """Drop ratings from the training store: every rating of `image_ids`, or
    only `user_id`'s ratings of them. (Training-store user ids are strings.)"""
    if not image_ids:
        return
    conn = connect_training_db(current_app.config["TRAINING_DB"])
    try:
        for image_id in image_ids:
            if user_id is None:
                conn.execute("DELETE FROM ratings WHERE image_id = ?", (image_id,))
                conn.execute("DELETE FROM embeddings WHERE image_id = ?", (image_id,))
            else:
                conn.execute(
                    "DELETE FROM ratings WHERE image_id = ? AND user_id = ?",
                    (image_id, str(user_id)),
                )
        conn.commit()
    finally:
        conn.close()


def delete_post(db: sqlite3.Connection, image_id: str) -> bool:
    """Delete a post: its row, its ratings (app + training store) and its file.
    Returns False if there's no such post."""
    row = db.execute("SELECT ext FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        return False
    db.execute("DELETE FROM ratings WHERE image_id = ?", (image_id,))
    db.execute("DELETE FROM images WHERE id = ?", (image_id,))
    db.commit()
    _forget_training_ratings([image_id])
    path = os.path.join(current_app.config["UPLOAD_DIR"], f"{image_id}{row['ext']}")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return True


def ban_user(db: sqlite3.Connection, user_id: int, purge: bool = False) -> dict:
    """Ban: can't log in, and their current logins stop working at once
    (security.check_account). With `purge`, also delete their posts and
    their ratings of everyone else's."""
    db.execute("UPDATE users SET banned = 1 WHERE id = ?", (user_id,))
    db.commit()
    removed = {"posts": 0, "ratings": 0}
    if purge:
        for (image_id,) in db.execute(
            "SELECT id FROM images WHERE owner_id = ?", (user_id,)
        ).fetchall():
            removed["posts"] += delete_post(db, image_id)
        rated = [
            r[0]
            for r in db.execute(
                "SELECT image_id FROM ratings WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        db.execute("DELETE FROM ratings WHERE user_id = ?", (user_id,))
        db.commit()
        _forget_training_ratings(rated, user_id)
        removed["ratings"] = len(rated)
    return removed


def unban_user(db: sqlite3.Connection, user_id: int) -> None:
    db.execute("UPDATE users SET banned = 0 WHERE id = ?", (user_id,))
    db.commit()


def set_password(db: sqlite3.Connection, user_id: int, password_hash: str) -> None:
    """Replace a password and log the account out everywhere (tokens issued
    before now stop working), in case it was set because of a break-in."""
    db.execute(
        "UPDATE users SET password_hash = ?, tokens_after = ? WHERE id = ?",
        (password_hash, int(time.time()), user_id),
    )
    db.commit()
