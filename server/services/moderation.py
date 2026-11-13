"""Operator moderation: hide, restore, quarantine and delete posts, ban users,
reset passwords.

Run from the command line (`python -m server.admin`), never over the API.

Hiding is the everyday tool: the post leaves the site and can come back.
Quarantine is for suspected child sexual abuse material. It moves the file
out of the uploads folder (so out of future backups too) but keeps it,
because after a report to the National Crime Agency the content and the
uploader's data must be kept for a year (SI 2026/268 reg. 8). So quarantined
posts can't be deleted or purged by accident.

Deleting anything rated has to reach the training store too: ingest only
adds and updates rows there (horse/ingest.py), so a deleted post's ratings,
or a purged user's votes, would otherwise keep training the head.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time

from flask import current_app

from horse.storage import connect as connect_training_db

from .reports import resolve_reports

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
                conn.execute("DELETE FROM augments WHERE image_id = ?", (image_id,))
            else:
                conn.execute(
                    "DELETE FROM ratings WHERE image_id = ? AND user_id = ?",
                    (image_id, str(user_id)),
                )
        conn.commit()
    finally:
        conn.close()


class QuarantinedError(Exception):
    """This would destroy material that has to be kept."""


def _forget_ratings(db: sqlite3.Connection, image_id: str) -> None:
    db.execute("DELETE FROM ratings WHERE image_id = ?", (image_id,))
    db.commit()
    _forget_training_ratings([image_id])


def delete_post(db: sqlite3.Connection, image_id: str, quarantined_too: bool = False) -> bool:
    """Delete a post: its row, its ratings (app + training store) and its file.
    Returns False if there's no such post. A quarantined post raises
    QuarantinedError unless `quarantined_too`."""
    row = db.execute(
        "SELECT ext, quarantined_at FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if row is None:
        return False
    if row["quarantined_at"] is not None and not quarantined_too:
        raise QuarantinedError(image_id)
    _forget_ratings(db, image_id)
    db.execute("DELETE FROM images WHERE id = ?", (image_id,))
    db.commit()
    resolve_reports(db, image_id, "post deleted")
    folder = "QUARANTINE_DIR" if row["quarantined_at"] is not None else "UPLOAD_DIR"
    path = os.path.join(current_app.config[folder], f"{image_id}{row['ext']}")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return True


def hide_post(db: sqlite3.Connection, image_id: str, reason: str = "operator") -> bool:
    """Take a post off the site, reversibly. False if there's no such post."""
    if db.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is None:
        return False
    db.execute(
        "UPDATE images SET hidden_at = ?, hidden_reason = ? WHERE id = ? AND hidden_at IS NULL",
        (time.time(), reason, image_id),
    )
    db.commit()
    return True


def restore_post(db: sqlite3.Connection, image_id: str, from_quarantine: bool = False) -> bool:
    """Put a hidden post back and close its reports. False if there's no such
    post. A quarantined post raises QuarantinedError unless `from_quarantine`;
    its votes were deleted when it was quarantined and don't come back."""
    row = db.execute(
        "SELECT ext, quarantined_at FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if row is None:
        return False
    if row["quarantined_at"] is not None:
        if not from_quarantine:
            raise QuarantinedError(image_id)
        name = f"{image_id}{row['ext']}"
        os.makedirs(current_app.config["UPLOAD_DIR"], exist_ok=True)
        try:
            os.replace(
                os.path.join(current_app.config["QUARANTINE_DIR"], name),
                os.path.join(current_app.config["UPLOAD_DIR"], name),
            )
        except FileNotFoundError:
            pass
    db.execute(
        "UPDATE images SET hidden_at = NULL, hidden_reason = NULL, quarantined_at = NULL "
        "WHERE id = ?",
        (image_id,),
    )
    db.commit()
    resolve_reports(db, image_id, "restored")
    return True


def quarantine_post(db: sqlite3.Connection, image_id: str) -> dict | None:
    """Hide a post for good, and move its file where it can't be served or
    backed up. Votes on it stop training the head. Returns what you need for
    a report, or None if there's no such post."""
    row = db.execute(
        """
        SELECT i.id, i.ext, i.filename, i.caption, i.created_at, u.username, u.invite_code,
               u.created_at AS user_created_at
        FROM images i JOIN users u ON u.id = i.owner_id
        WHERE i.id = ?
        """,
        (image_id,),
    ).fetchone()
    if row is None:
        return None
    folder = current_app.config["QUARANTINE_DIR"]
    os.makedirs(folder, mode=0o700, exist_ok=True)
    name = f"{image_id}{row['ext']}"
    target = os.path.join(folder, name)
    digest = None
    try:
        os.replace(os.path.join(current_app.config["UPLOAD_DIR"], name), target)
    except FileNotFoundError:
        pass
    if os.path.exists(target):
        os.chmod(target, 0o600)
        with open(target, "rb") as stored:
            digest = hashlib.sha256(stored.read()).hexdigest()
    now = time.time()
    db.execute(
        "UPDATE images SET hidden_at = ?, hidden_reason = 'quarantined', quarantined_at = ? "
        "WHERE id = ?",
        (now, now, image_id),
    )
    db.commit()
    _forget_ratings(db, image_id)
    resolve_reports(db, image_id, "quarantined")
    return {**dict(row), "sha256": digest, "path": target if digest else None}


def purge_rejected_files(db: sqlite3.Connection) -> int:
    """Delete the stored files of uploads the gate rejected (left over from
    before they stopped being kept). Their rows stay. Returns files removed."""
    rows = db.execute(
        "SELECT id, ext FROM images WHERE passed = 0 AND quarantined_at IS NULL"
    ).fetchall()
    removed = 0
    for row in rows:
        try:
            os.remove(os.path.join(current_app.config["UPLOAD_DIR"], f"{row['id']}{row['ext']}"))
            removed += 1
        except FileNotFoundError:
            pass
    return removed


BANNED_OWNER = "owner banned"


def ban_user(db: sqlite3.Connection, user_id: int, purge: bool = False) -> dict:
    """Ban: can't log in, their current logins stop working at once
    (security.check_account), and their posts come off the site. With
    `purge`, also delete their posts and their ratings of everyone else's.
    A purge raises QuarantinedError, before changing anything, if they have
    a quarantined post: their data has to be kept alongside it."""
    if purge and db.execute(
        "SELECT 1 FROM images WHERE owner_id = ? AND quarantined_at IS NOT NULL", (user_id,)
    ).fetchone():
        raise QuarantinedError(str(user_id))
    db.execute("UPDATE users SET banned = 1 WHERE id = ?", (user_id,))
    db.execute(
        "UPDATE images SET hidden_at = ?, hidden_reason = ? WHERE owner_id = ? AND hidden_at IS NULL",
        (time.time(), BANNED_OWNER, user_id),
    )
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
    """Let them back in, and bring back the posts the ban took down. Posts
    hidden for any other reason stay hidden."""
    db.execute("UPDATE users SET banned = 0 WHERE id = ?", (user_id,))
    db.execute(
        "UPDATE images SET hidden_at = NULL, hidden_reason = NULL "
        "WHERE owner_id = ? AND hidden_reason = ?",
        (user_id, BANNED_OWNER),
    )
    db.commit()


def set_password(db: sqlite3.Connection, user_id: int, password_hash: str) -> None:
    """Replace a password and log the account out everywhere (tokens issued
    before now stop working), in case it was set because of a break-in."""
    db.execute(
        "UPDATE users SET password_hash = ?, tokens_after = ? WHERE id = ?",
        (password_hash, int(time.time()), user_id),
    )
    db.commit()
