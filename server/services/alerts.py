"""Email the operator when something needs a look.

Off unless HORSE_ALERT_EMAIL and HORSE_SMTP_HOST are set. Sent on a
background thread, so a slow mail server can't hold up a request. Alerts
are rate limited, since reports can be spammed; anything not emailed is
still in the queue (`python -m server.admin reports`).

The email never quotes what people wrote, and never includes an image.
"""

from __future__ import annotations

import smtplib
import ssl
import sqlite3
import threading
from email.message import EmailMessage

from flask import Flask, current_app

from . import limits

ALERTS_PER_HOUR = 6
URGENT_ALERTS_PER_HOUR = 20


def _send(app: Flask, subject: str, body: str) -> None:
    cfg = app.config
    message = EmailMessage()
    message["Subject"] = f"[phorse] {subject}"
    message["From"] = cfg["SMTP_USER"] or cfg["ALERT_EMAIL"]
    message["To"] = cfg["ALERT_EMAIL"]
    message.set_content(body)
    try:
        with smtplib.SMTP(cfg["SMTP_HOST"], cfg["SMTP_PORT"], timeout=15) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            if cfg["SMTP_USER"]:
                smtp.login(cfg["SMTP_USER"], cfg["SMTP_PASSWORD"])
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        app.logger.error("couldn't send alert %r: %s", subject, exc)


def notify(
    db: sqlite3.Connection, subject: str, body: str, urgent: bool = False
) -> threading.Thread | None:
    """Send an alert in the background. Returns its thread, or None if alerts
    are off or this one was over the hourly limit."""
    cfg = current_app.config
    if not (cfg["ALERT_EMAIL"] and cfg["SMTP_HOST"]):
        return None
    kind, per_hour = ("alert-urgent", URGENT_ALERTS_PER_HOUR) if urgent else ("alert", ALERTS_PER_HOUR)
    if not limits.allow(db, kind, "operator", per_hour, limits.HOUR):
        return None
    thread = threading.Thread(
        target=_send, args=(current_app._get_current_object(), subject, body), daemon=True
    )
    thread.start()
    return thread
