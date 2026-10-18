"""Manage invite codes (signups need one -- see services/invites.py):

    python -m server.invites create --uses 300 --note "insta story"
    python -m server.invites create --code horsegirls --uses 50
    python -m server.invites list
    python -m server.invites revoke <code>

`create` and `list` print each code's signup link (HORSE_SITE_URL, else
the first HORSE_CORS_ORIGINS entry): people who open it just pick a
username and password, the code is filled in for them.

On the server: sudo bash /srv/phorse/deploy/manage.sh invites <command> ...
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from flask import current_app

from . import create_app
from .db import get_db
from .services import invites


def invite_link(code: str) -> str:
    return f"{current_app.config['SITE_URL']}/join/{code}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m server.invites")
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="make a new invite code")
    create.add_argument("--uses", type=int, help="signups allowed (default: unlimited)")
    create.add_argument("--note", help="reminder of where you shared it")
    create.add_argument("--code", help="pick the code yourself (default: random)")

    sub.add_parser("list", help="show every code and how much it's been used")

    revoke = sub.add_parser("revoke", help="stop a code working for new signups")
    revoke.add_argument("code")

    args = parser.parse_args(argv)
    app = create_app()
    with app.app_context():
        db = get_db()
        if args.cmd == "create":
            try:
                code = invites.create_invite(db, args.uses, note=args.note, code=args.code)
            except ValueError as err:
                print(f"error: {err}", file=sys.stderr)
                return 1
            print(invite_link(code))
        elif args.cmd == "list":
            rows = invites.list_invites(db)
            if not rows:
                print("no invite codes yet")
            for row in rows:
                limit = "unlimited" if row["max_uses"] is None else row["max_uses"]
                status = "  REVOKED" if row["revoked"] else ""
                made = datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d")
                print(f"{row['code']:<20} {row['uses']}/{limit} used  {made}  "
                      f"{row['note'] or ''}{status}")
                if not row["revoked"]:
                    print(f"    {invite_link(row['code'])}")
        elif args.cmd == "revoke":
            if not invites.revoke_invite(db, args.code):
                print(f"error: no invite code {args.code!r}", file=sys.stderr)
                return 1
            print(f"revoked {invites.normalize(args.code)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
