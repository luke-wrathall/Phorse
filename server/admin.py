"""Moderation from the command line (see services/moderation.py):

    python -m server.admin users                  every account, newest first
    python -m server.admin posts [USERNAME]       recent posts (all, or one user's)
    python -m server.admin delete-post ID_OR_URL  a post id, or its image URL from the site
    python -m server.admin ban USERNAME [--purge] --purge also deletes their posts + ratings
    python -m server.admin unban USERNAME
    python -m server.admin set-password USERNAME  prompts for the new password

On the server: sudo bash /srv/phorse/deploy/manage.sh admin <command> ...
Deleted posts are still in the nightly backups until those age out.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime

from . import create_app
from .blueprints.auth import MIN_PASSWORD
from .db import get_db
from .security import hash_password
from .services import moderation


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _list_users(db) -> None:
    rows = db.execute(
        """
        SELECT u.username, u.created_at, u.invite_code, u.banned,
               (SELECT COUNT(*) FROM images i WHERE i.owner_id = u.id) AS posts,
               (SELECT COUNT(*) FROM ratings r WHERE r.user_id = u.id) AS ratings
        FROM users u ORDER BY u.created_at DESC
        """
    ).fetchall()
    for r in rows:
        banned = "  BANNED" if r["banned"] else ""
        print(f"{r['username']:<24} joined {_date(r['created_at'])}  "
              f"invite {r['invite_code'] or '-':<12} {r['posts']} posts  "
              f"{r['ratings']} ratings{banned}")
    if not rows:
        print("no accounts yet")


def _list_posts(db, username: str | None, limit: int) -> int:
    where, params = "", ()
    if username:
        user = moderation.find_user(db, username)
        if user is None:
            return _fail(f"no user {username!r}")
        where, params = "WHERE i.owner_id = ?", (user["id"],)
    rows = db.execute(
        f"""
        SELECT i.id, i.caption, i.passed, i.created_at, u.username
        FROM images i JOIN users u ON u.id = i.owner_id {where}
        ORDER BY i.created_at DESC LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    for r in rows:
        shown = "" if r["passed"] else "  (didn't pass the gate)"
        print(f"{r['id']}  {_date(r['created_at'])}  {r['username']:<20} "
              f"{r['caption'] or ''}{shown}")
    if not rows:
        print("no posts")
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m server.admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("users", help="list every account")
    posts = sub.add_parser("posts", help="list recent posts")
    posts.add_argument("username", nargs="?")
    posts.add_argument("--limit", type=int, default=30)
    delete = sub.add_parser("delete-post", help="delete a post and its ratings")
    delete.add_argument("post", help="post id, or the image's URL")
    ban = sub.add_parser("ban", help="stop a user logging in")
    ban.add_argument("username")
    ban.add_argument("--purge", action="store_true", help="also delete their posts and ratings")
    unban = sub.add_parser("unban", help="let a banned user back in")
    unban.add_argument("username")
    passwd = sub.add_parser("set-password", help="give a user a new password")
    passwd.add_argument("username")
    args = parser.parse_args(argv)

    app = create_app()
    with app.app_context():
        db = get_db()
        if args.cmd == "users":
            _list_users(db)
            return 0
        if args.cmd == "posts":
            return _list_posts(db, args.username, args.limit)
        if args.cmd == "delete-post":
            image_id = moderation.parse_image_id(args.post)
            if image_id is None or not moderation.delete_post(db, image_id):
                return _fail(f"no post {args.post!r}")
            print(f"deleted post {image_id}")
            return 0

        user = moderation.find_user(db, args.username)
        if user is None:
            return _fail(f"no user {args.username!r}")
        if args.cmd == "ban":
            removed = moderation.ban_user(db, user["id"], purge=args.purge)
            extra = (f", deleted {removed['posts']} posts and {removed['ratings']} ratings"
                     if args.purge else "")
            print(f"banned {user['username']}{extra}")
        elif args.cmd == "unban":
            moderation.unban_user(db, user["id"])
            print(f"unbanned {user['username']}")
        elif args.cmd == "set-password":
            password = getpass.getpass(f"new password for {user['username']}: ")
            if len(password) < MIN_PASSWORD:
                return _fail(f"passwords are {MIN_PASSWORD}+ characters")
            if getpass.getpass("again: ") != password:
                return _fail("those didn't match")
            moderation.set_password(db, user["id"], hash_password(password))
            print(f"password changed for {user['username']}; they're logged out everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
