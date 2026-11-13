"""Moderation from the command line (see services/moderation.py):

    python -m server.admin status                 pauses, open reports, what's set up
    python -m server.admin reports [--all]        the report queue
    python -m server.admin resolve REPORT_ID      close a report without doing anything else
    python -m server.admin blocks                 uploads the content check refused
    python -m server.admin users                  every account, newest first
    python -m server.admin posts [USERNAME]       recent posts (all, or one user's)
    python -m server.admin review-link ID_OR_URL  a 15-minute link to see a hidden post
    python -m server.admin hide ID_OR_URL         take a post off the site, reversibly
    python -m server.admin restore ID_OR_URL      put it back and close its reports
    python -m server.admin quarantine ID_OR_URL   suspected child abuse material: see below
    python -m server.admin delete-post ID_OR_URL  a post id, or its image URL from the site
    python -m server.admin ban USERNAME [--purge] also hides their posts; --purge deletes
                                                  their posts + ratings
    python -m server.admin unban USERNAME
    python -m server.admin set-password USERNAME  prompts for the new password
    python -m server.admin pause uploads|signups|all
    python -m server.admin unpause uploads|signups|all
    python -m server.admin purge-rejected-files   delete rejected uploads' leftover files

On the server: sudo bash /srv/phorse/deploy/manage.sh admin <command> ...
Deleted posts are still in the nightly backups until those age out.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime, timezone

from flask import current_app

from . import create_app
from .blueprints.auth import MIN_PASSWORD
from .db import get_db
from .security import hash_password, make_review_token
from .services import limits, moderation
from .services import reports as reports_service


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


def _post_state(row) -> str:
    if row["quarantined_at"] is not None:
        return "  QUARANTINED"
    if row["hidden_at"] is not None:
        return f"  HIDDEN ({row['hidden_reason']})"
    return "" if row["passed"] else "  (didn't pass the gate)"


def _status(db) -> None:
    cfg = current_app.config
    for what in limits.PAUSABLE:
        print(f"{what:<16} {'PAUSED' if limits.is_paused(db, what) else 'open'}")
    open_reports = db.execute(
        "SELECT COUNT(*) FROM reports WHERE resolved_at IS NULL"
    ).fetchone()[0]
    print(f"{'open reports':<16} {open_reports}")
    hidden = db.execute("SELECT COUNT(*) FROM images WHERE hidden_at IS NOT NULL").fetchone()[0]
    print(f"{'hidden posts':<16} {hidden}")
    if cfg["MODERATION_PROVIDER"] == "off":
        check = "OFF"
    elif cfg["MODERATION_PROVIDER"] == "openai" and not cfg["OPENAI_API_KEY"]:
        check = "NO KEY: every upload that passes the gate is being refused"
    else:
        check = cfg["MODERATION_PROVIDER"]
    print(f"{'content check':<16} {check}")
    alerts_on = cfg["ALERT_EMAIL"] and cfg["SMTP_HOST"]
    print(f"{'alert emails':<16} {'to ' + cfg['ALERT_EMAIL'] if alerts_on else 'OFF'}")


def _list_reports(db, include_resolved: bool, limit: int) -> None:
    rows = reports_service.list_reports(db, include_resolved, limit)
    for r in rows:
        state = "no post"
        if r["image_id"]:
            if r["owner"] is None:
                state = "post deleted"
            elif r["quarantined_at"] is not None:
                state = "quarantined"
            else:
                state = "hidden" if r["hidden_at"] is not None else "SHOWING"
        print(f"#{r['id']}  {_date(r['created_at'])}  {r['reason']}  [{state}]")
        if r["image_id"]:
            print(f"    post     {r['image_id']}, by {r['owner'] or '?'}")
        print(f"    from     {r['reporter'] or 'someone logged out'}")
        if r["contact"]:
            print(f"    contact  {r['contact']}")
        if r["details"]:
            print(f"    says     {r['details']}")
        if r["resolved_at"]:
            print(f"    closed   {_date(r['resolved_at'])}: {r['resolution']}")
    if not rows:
        print("no open reports" if not include_resolved else "no reports")


def _list_blocks(db, limit: int) -> None:
    rows = db.execute(
        """
        SELECT b.created_at, b.categories, b.filename, b.sha256, u.username
        FROM upload_blocks b JOIN users u ON u.id = b.user_id
        ORDER BY b.created_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for r in rows:
        print(f"{_date(r['created_at'])}  {r['username']:<20} {r['categories']}  "
              f"{r['filename'] or ''}  sha256 {r['sha256'][:16]}")
    if not rows:
        print("nothing blocked")


def _review_link(db, image_id: str, yes: bool) -> int:
    row = db.execute(
        "SELECT quarantined_at, passed FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if row is None:
        return _fail(f"no post {image_id!r}")
    if row["quarantined_at"] is not None or not row["passed"]:
        return _fail("there's no file to look at: it's quarantined, or was rejected at the gate")
    if reports_service.has_open_child_safety_report(db, image_id) and not yes:
        return _fail(
            "this post is reported as child abuse material. Opening it means viewing it, and\n"
            "viewing or saving such an image can be an offence. If you can't rule the report\n"
            "out from its text, `quarantine` it unseen. To open it anyway, add --yes, and never\n"
            "download or screenshot it."
        )
    token = make_review_token(image_id)
    print(f"{current_app.config['SITE_URL']}/api/images/{image_id}/file?review={token}")
    print("works for 15 minutes. don't forward it.")
    return 0


def _print_quarantine_record(record: dict) -> None:
    uploaded = datetime.fromtimestamp(record["created_at"], timezone.utc)
    print(f"quarantined post {record['id']}. Not viewable, not in new backups, kept on disk.\n")
    print("for your report:")
    print(f"  uploaded      {uploaded:%d/%m/%Y %H:%M:%S} UTC")
    print(f"  by            {record['username']} (invite: {record['invite_code'] or '-'})")
    print(f"  filename      {record['filename'] or '-'}")
    print(f"  caption       {record['caption'] or '-'}")
    print(f"  file sha256   {record['sha256'] or 'no file on disk'}")
    print(f"  file          {record['path'] or '-'}")
    print(f"""
what to do now:
  1. Don't open, copy or download it. Never pull it to your own computer.
  2. Find the uploader's IP (it's in the access log, look near the time above):
       zgrep -h '"method":"POST"' /var/log/caddy/phorse.log* | grep '"uri":"/api/images"'
  3. Register at the National Crime Agency's Child Sexual Exploitation & Abuse Industry
     Reporting Portal and send a report. Ofcom's page on the duty explains how:
     https://www.ofcom.org.uk/online-safety/illegal-and-harmful-content/roadmap-to-regulation/
     If a child is in immediate danger, call 999.
  4. Keep the report's reference number for 5 years. Keep the file, the report and
     the uploader's data for 1 year from sending it. Don't purge this user.
  5. As root, remove the older copies in the nightly backups:
       rm -f /var/backups/phorse/*/uploads.tar.gz""")


def _list_posts(db, username: str | None, limit: int) -> int:
    where, params = "", ()
    if username:
        user = moderation.find_user(db, username)
        if user is None:
            return _fail(f"no user {username!r}")
        where, params = "WHERE i.owner_id = ?", (user["id"],)
    rows = db.execute(
        f"""
        SELECT i.id, i.caption, i.passed, i.hidden_at, i.hidden_reason, i.quarantined_at,
               i.created_at, u.username
        FROM images i JOIN users u ON u.id = i.owner_id {where}
        ORDER BY i.created_at DESC LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    for r in rows:
        print(f"{r['id']}  {_date(r['created_at'])}  {r['username']:<20} "
              f"{r['caption'] or ''}{_post_state(r)}")
    if not rows:
        print("no posts")
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m server.admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="pauses, open reports, what's set up")
    reports = sub.add_parser("reports", help="the report queue")
    reports.add_argument("--all", action="store_true", help="include closed reports")
    reports.add_argument("--limit", type=int, default=50)
    resolve = sub.add_parser("resolve", help="close a report without doing anything else")
    resolve.add_argument("report_id", type=int)
    resolve.add_argument("--note", default="no action", help="why, for your own records")
    blocks = sub.add_parser("blocks", help="uploads the content check refused")
    blocks.add_argument("--limit", type=int, default=50)
    review = sub.add_parser("review-link", help="a 15-minute link to see a hidden post")
    review.add_argument("post", help="post id, or the image's URL")
    review.add_argument("--yes", action="store_true",
                        help="open a post reported as child abuse material anyway")
    hide = sub.add_parser("hide", help="take a post off the site, reversibly")
    hide.add_argument("post", help="post id, or the image's URL")
    restore = sub.add_parser("restore", help="put a hidden post back, close its reports")
    restore.add_argument("post", help="post id, or the image's URL")
    restore.add_argument("--from-quarantine", action="store_true",
                         help="only if a quarantined post was a mistake")
    quarantine = sub.add_parser("quarantine", help="suspected child abuse material")
    quarantine.add_argument("post", help="post id, or the image's URL")
    for name in ("pause", "unpause"):
        switch = sub.add_parser(name, help=f"{name} uploads, signups or both, at once")
        switch.add_argument("what", choices=(*limits.PAUSABLE, "all"))
    sub.add_parser("purge-rejected-files", help="delete rejected uploads' leftover files")
    sub.add_parser("users", help="list every account")
    posts = sub.add_parser("posts", help="list recent posts")
    posts.add_argument("username", nargs="?")
    posts.add_argument("--limit", type=int, default=30)
    delete = sub.add_parser("delete-post", help="delete a post and its ratings")
    delete.add_argument("post", help="post id, or the image's URL")
    delete.add_argument("--quarantined-too", action="store_true",
                        help="once a quarantined post's retention period is over")
    ban = sub.add_parser("ban", help="stop a user logging in, hide their posts")
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
        if args.cmd == "status":
            _status(db)
            return 0
        if args.cmd == "reports":
            _list_reports(db, args.all, args.limit)
            return 0
        if args.cmd == "resolve":
            if not reports_service.resolve_report(db, args.report_id, args.note):
                return _fail(f"no open report #{args.report_id}")
            print(f"closed report #{args.report_id}")
            return 0
        if args.cmd == "blocks":
            _list_blocks(db, args.limit)
            return 0
        if args.cmd == "users":
            _list_users(db)
            return 0
        if args.cmd == "posts":
            return _list_posts(db, args.username, args.limit)
        if args.cmd in ("pause", "unpause"):
            for what in limits.PAUSABLE if args.what == "all" else (args.what,):
                limits.set_paused(db, what, args.cmd == "pause")
                print(f"{what}: {'paused' if args.cmd == 'pause' else 'open'}")
            return 0
        if args.cmd == "purge-rejected-files":
            removed = moderation.purge_rejected_files(db)
            print(f"deleted {removed} files. Copies stay in the nightly backups until those age out.")
            return 0
        if args.cmd in ("review-link", "hide", "restore", "quarantine", "delete-post"):
            image_id = moderation.parse_image_id(args.post)
            if image_id is None:
                return _fail(f"no post {args.post!r}")
            if args.cmd == "review-link":
                return _review_link(db, image_id, args.yes)
            if args.cmd == "quarantine":
                record = moderation.quarantine_post(db, image_id)
                if record is None:
                    return _fail(f"no post {args.post!r}")
                _print_quarantine_record(record)
                return 0
            try:
                if args.cmd == "hide":
                    found = moderation.hide_post(db, image_id)
                elif args.cmd == "restore":
                    found = moderation.restore_post(db, image_id, args.from_quarantine)
                else:
                    found = moderation.delete_post(db, image_id, args.quarantined_too)
            except moderation.QuarantinedError:
                return _fail(
                    f"post {image_id} is quarantined. It may have to be kept for a year "
                    f"after a report, so this needs "
                    f"{'--from-quarantine' if args.cmd == 'restore' else '--quarantined-too'}."
                )
            if not found:
                return _fail(f"no post {args.post!r}")
            done = {"hide": "hid", "restore": "restored", "delete-post": "deleted"}[args.cmd]
            print(f"{done} post {image_id}")
            return 0

        user = moderation.find_user(db, args.username)
        if user is None:
            return _fail(f"no user {args.username!r}")
        if args.cmd == "ban":
            try:
                removed = moderation.ban_user(db, user["id"], purge=args.purge)
            except moderation.QuarantinedError:
                return _fail(
                    f"{user['username']} has a quarantined post, and their data may have to "
                    "be kept for a year after a report. Ban them without --purge."
                )
            extra = (f", deleted {removed['posts']} posts and {removed['ratings']} ratings"
                     if args.purge else ", hid their posts")
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
