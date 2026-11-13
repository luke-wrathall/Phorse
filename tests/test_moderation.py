"""The moderation layer: rejected uploads aren't kept, reports hide posts,
quarantine keeps what has to be kept, flood limits, pause switches, and the
automated content check (with OpenAI faked: no network here)."""

from __future__ import annotations

import io
import json
import os
import smtplib
import time
import urllib.error

import pytest

from conftest import HORSE_COLORS, auth, register, solid_png, upload_png
from horse.ingest import ingest
from horse.storage import connect as connect_training_db
from server.admin import main as admin_cli
from server.db import get_db
from server.services import alerts, limits, safety


@pytest.fixture
def cli(app, monkeypatch):
    monkeypatch.setattr("server.admin.create_app", lambda: app)
    return admin_cli


@pytest.fixture
def post(client, fixed_score):
    """bob's image, past the gate."""
    fixed_score["value"] = 4.0
    token = register(client, "bob")
    return {"bob": token, "id": upload_png(client, token, solid_png(HORSE_COLORS[0]))["id"]}


def feed_ids(client) -> list[str]:
    return [i["id"] for i in client.get("/api/feed").get_json()["items"]]


def report(client, image_id=None, reason="harmful", ip="1.1.1.1", headers=None, **fields):
    return client.post(
        "/api/reports",
        json={"reason": reason, "image_id": image_id, **fields},
        headers={"X-Forwarded-For": ip, **(headers or {})},
    )


def files_in(folder) -> list[str]:
    return os.listdir(folder) if os.path.isdir(folder) else []


# --- rejected uploads ------------------------------------------------------


def test_rejected_upload_keeps_no_file(app, client, fixed_score):
    token = register(client, "alice")
    fixed_score["value"] = 0.0

    result = upload_png(client, token, solid_png(HORSE_COLORS[0]), name="x.png")

    assert result["passed"] is False
    assert files_in(app.config["UPLOAD_DIR"]) == []
    assert client.get(f"/api/images/{result['id']}/file").status_code == 404
    mine = client.get("/api/images/mine", headers=auth(token)).get_json()["items"]
    assert [(m["id"], m["passed"], m["image_url"]) for m in mine] == [(result["id"], False, None)]


def test_a_rejected_uploads_old_file_is_not_served_and_can_be_purged(app, client, fixed_score, cli):
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    kept = upload_png(client, token, solid_png(HORSE_COLORS[0]))["id"]
    gone = upload_png(client, token, solid_png(HORSE_COLORS[1]))["id"]
    with app.app_context():  # as if rejected before files stopped being kept
        get_db().execute("UPDATE images SET passed = 0 WHERE id = ?", (gone,))
        get_db().commit()

    assert client.get(f"/api/images/{gone}/file").status_code == 404
    assert cli(["purge-rejected-files"]) == 0
    assert files_in(app.config["UPLOAD_DIR"]) == [f"{kept}.png"]
    assert cli(["purge-rejected-files"]) == 0, "safe to run twice"


def test_only_showing_posts_can_be_voted_on(client, fixed_score, post, cli):
    alice = register(client, "alice")
    fixed_score["value"] = 0.0
    rejected = upload_png(client, alice, solid_png(HORSE_COLORS[1]))["id"]

    def vote(image):
        return client.post(f"/api/images/{image}/ratings", json={"rating": 5}, headers=auth(alice))

    assert vote(rejected).status_code == 404, "uploader could train the head to let it in"
    assert vote(post["id"]).status_code == 200
    assert cli(["hide", post["id"]]) == 0
    assert vote(post["id"]).status_code == 404


# --- reports ---------------------------------------------------------------


def test_report_works_logged_out_and_reaches_the_queue(client, post, cli, capsys):
    resp = report(client, post["id"], details="this isn't ok", contact="me@example.com")
    assert resp.status_code == 201
    assert resp.get_json() == {"ok": True}, "doesn't reveal whether the post was hidden"

    assert cli(["reports"]) == 0
    out = capsys.readouterr().out
    assert post["id"] in out and "this isn't ok" in out and "me@example.com" in out
    assert "SHOWING" in out and "someone logged out" in out


def test_three_different_reporters_hide_a_post_and_restore_brings_it_back(client, post, cli, capsys):
    for n, ip in enumerate(["1.1.1.1", "1.1.1.1", "2.2.2.2"]):
        report(client, post["id"], ip=ip)
    assert post["id"] in feed_ids(client), "the same person twice counts once"

    report(client, post["id"], ip="3.3.3.3")
    assert feed_ids(client) == []
    assert client.get(f"/api/images/{post['id']}/file").status_code == 404
    mine = client.get("/api/images/mine", headers=auth(post["bob"])).get_json()["items"]
    assert mine[0]["removed"] is True and mine[0]["image_url"] is None

    assert cli(["restore", post["id"]]) == 0
    assert feed_ids(client) == [post["id"]]
    assert client.get(f"/api/images/{post['id']}/file").status_code == 200
    capsys.readouterr()
    assert cli(["reports"]) == 0
    assert "no open reports" in capsys.readouterr().out


@pytest.mark.parametrize("reason", ["child_safety", "intimate_image"])
def test_the_worst_reports_hide_at_once(client, post, reason):
    report(client, post["id"], reason=reason)
    assert feed_ids(client) == []


def test_a_serious_report_counts_after_a_lesser_one_from_the_same_person(client, post):
    report(client, post["id"], reason="other")
    report(client, post["id"], reason="child_safety")
    assert feed_ids(client) == []


def test_an_appeal_never_hides_anything(client, post):
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"):
        report(client, post["id"], reason="appeal", ip=ip)
    assert feed_ids(client) == [post["id"]]


def test_report_validation(client, post):
    assert report(client, post["id"], reason="nope").status_code == 400
    assert report(client, "no-such-post").status_code == 404
    assert report(client, 123).status_code == 404
    assert report(client, details="   ").status_code == 400, "a general complaint needs words"
    assert report(client, details="your terms are confusing", reason="other").status_code == 201


def test_a_rejected_upload_cant_be_reported(client, fixed_score):
    token = register(client, "alice")
    fixed_score["value"] = 0.0
    rejected = upload_png(client, token, solid_png(HORSE_COLORS[0]))["id"]
    assert report(client, rejected).status_code == 404


def test_resolve_closes_one_report_and_leaves_the_post(client, post, cli, capsys):
    report(client, post["id"])
    assert cli(["resolve", "1", "--note", "it's fine"]) == 0
    assert feed_ids(client) == [post["id"]]
    assert cli(["resolve", "1"]) == 1, "already closed"
    capsys.readouterr()
    assert cli(["reports", "--all"]) == 0
    assert "it's fine" in capsys.readouterr().out


def test_deleting_a_post_closes_its_reports(client, post, cli, capsys):
    report(client, post["id"])
    assert cli(["delete-post", post["id"]]) == 0
    capsys.readouterr()
    assert cli(["reports"]) == 0
    assert "no open reports" in capsys.readouterr().out


# --- review links ----------------------------------------------------------


def test_review_link_opens_only_that_hidden_post(app, client, post, cli, capsys):
    assert cli(["hide", post["id"]]) == 0
    capsys.readouterr()
    assert cli(["review-link", post["id"]]) == 0
    url = capsys.readouterr().out.splitlines()[0]
    token = url.split("review=")[1]

    resp = client.get(f"/api/images/{post['id']}/file?review={token}")
    assert resp.status_code == 200 and resp.headers["Cache-Control"] == "no-store"
    resp.close()

    other = upload_png(client, post["bob"], solid_png(HORSE_COLORS[1]))["id"]
    assert cli(["hide", other]) == 0
    assert client.get(f"/api/images/{other}/file?review={token}").status_code == 404
    assert client.get(f"/api/images/{post['id']}/file?review=rubbish").status_code == 404
    assert client.get("/api/auth/me", headers={"X-Horse-Token": token}).status_code == 401, (
        "a review link is not a login"
    )


def test_review_link_refuses_a_child_safety_report_without_yes(client, post, cli, capsys):
    report(client, post["id"], reason="child_safety")
    assert cli(["review-link", post["id"]]) == 1
    assert "child abuse" in capsys.readouterr().err
    assert cli(["review-link", post["id"], "--yes"]) == 0


# --- quarantine ------------------------------------------------------------


def training_ratings(app) -> list[str]:
    conn = connect_training_db(app.config["TRAINING_DB"])
    try:
        return [r[0] for r in conn.execute("SELECT image_id FROM ratings").fetchall()]
    finally:
        conn.close()


def test_quarantine_moves_the_file_keeps_the_record_and_stops_training(app, client, post, cli, capsys):
    alice = register(client, "alice")
    client.post(f"/api/images/{post['id']}/ratings", json={"rating": 5}, headers=auth(alice))
    ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
    report(client, post["id"], reason="child_safety")

    assert cli(["quarantine", post["id"]]) == 0

    out = capsys.readouterr().out
    assert "bob" in out and "file sha256" in out and "no file on disk" not in out
    assert files_in(app.config["UPLOAD_DIR"]) == []
    assert files_in(app.config["QUARANTINE_DIR"]) == [f"{post['id']}.png"]
    assert client.get(f"/api/images/{post['id']}/file").status_code == 404
    assert feed_ids(client) == []
    assert training_ratings(app) == []
    ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
    assert training_ratings(app) == [], "ingest doesn't bring the votes back"
    assert cli(["reports"]) == 0
    assert "no open reports" in capsys.readouterr().out
    assert cli(["review-link", post["id"]]) == 1, "no way to open it"


def test_quarantined_post_cant_be_deleted_or_purged_by_accident(app, client, post, cli, capsys):
    assert cli(["quarantine", post["id"]]) == 0

    assert cli(["delete-post", post["id"]]) == 1
    assert cli(["ban", "bob", "--purge"]) == 1
    assert files_in(app.config["QUARANTINE_DIR"]) == [f"{post['id']}.png"]
    assert client.post(
        "/api/auth/login", json={"username": "bob", "password": "password123"}
    ).status_code == 200, "the refused purge didn't ban them either"

    assert cli(["ban", "bob"]) == 0
    assert cli(["unban", "bob"]) == 0
    assert feed_ids(client) == [], "unbanning doesn't bring a quarantined post back"

    assert cli(["delete-post", post["id"], "--quarantined-too"]) == 0
    assert files_in(app.config["QUARANTINE_DIR"]) == []


def test_a_mistaken_quarantine_can_be_undone_explicitly(client, post, cli):
    assert cli(["quarantine", post["id"]]) == 0
    assert cli(["restore", post["id"]]) == 1
    assert feed_ids(client) == []
    assert cli(["restore", post["id"], "--from-quarantine"]) == 0
    assert feed_ids(client) == [post["id"]]
    assert client.get(f"/api/images/{post['id']}/file").status_code == 200


def test_quarantine_a_post_with_no_file_still_records_it(app, client, fixed_score, cli, capsys):
    token = register(client, "alice")
    fixed_score["value"] = 0.0
    rejected = upload_png(client, token, solid_png(HORSE_COLORS[0]))["id"]
    assert cli(["quarantine", rejected]) == 0
    assert "no file on disk" in capsys.readouterr().out


# --- bans ------------------------------------------------------------------


def test_ban_hides_their_posts_and_unban_brings_back_only_those(client, post, cli):
    other = upload_png(client, post["bob"], solid_png(HORSE_COLORS[1]))["id"]
    assert cli(["hide", other]) == 0
    assert feed_ids(client) == [post["id"]]

    assert cli(["ban", "bob"]) == 0
    assert feed_ids(client) == []
    assert client.get(f"/api/images/{post['id']}/file").status_code == 404

    assert cli(["unban", "bob"]) == 0
    assert feed_ids(client) == [post["id"]], "the one hidden by hand stays hidden"


# --- limits and pauses -----------------------------------------------------


def test_upload_limit_per_user_counts_rejected_uploads(app, client, fixed_score):
    token = register(client, "alice")
    app.config["UPLOADS_PER_USER_DAY"] = 2
    fixed_score["value"] = 0.0
    upload_png(client, token, solid_png(HORSE_COLORS[0]))
    upload_png(client, token, solid_png(HORSE_COLORS[1]))

    resp = client.post(
        "/api/images",
        data={"image": (io.BytesIO(solid_png(HORSE_COLORS[2])), "x.png")},
        headers=auth(token),
        content_type="multipart/form-data",
    )
    assert resp.status_code == 429
    assert "tomorrow" in resp.get_json()["error"]

    other = register(client, "someone")
    upload_png(client, other, solid_png(HORSE_COLORS[0]))


def test_upload_limit_per_ip_spans_accounts(app, client, fixed_score):
    app.config["UPLOADS_PER_IP_DAY"] = 1
    fixed_score["value"] = 0.0

    def send(token, ip):
        return client.post(
            "/api/images",
            data={"image": (io.BytesIO(solid_png(HORSE_COLORS[0])), "x.png")},
            headers={**auth(token), "X-Forwarded-For": ip},
            content_type="multipart/form-data",
        ).status_code

    alice, bob = register(client, "alice"), register(client, "bob")
    assert send(alice, "5.5.5.5") == 201
    assert send(bob, "5.5.5.5") == 429
    assert send(bob, "6.6.6.6") == 201


def test_signup_limit_per_ip(app, client):
    app.config["SIGNUPS_PER_IP_HOUR"] = 1

    def sign_up(name, ip):
        return client.post(
            "/api/auth/register",
            json={"username": name, "password": "password123"},
            headers={"X-Forwarded-For": ip},
        ).status_code

    assert sign_up("first", "7.7.7.7") == 201
    assert sign_up("second", "7.7.7.7") == 429
    assert sign_up("second", "8.8.8.8") == 201


def test_report_limit(app, client, post):
    app.config["REPORTS_PER_REPORTER_DAY"] = 1
    assert report(client, details="one", reason="other").status_code == 201
    assert report(client, details="two", reason="other").status_code == 429
    assert report(client, details="two", reason="other", ip="9.9.9.9").status_code == 201


def test_old_rate_records_stop_counting_and_get_dropped(app):
    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO rate_events (kind, key, created_at) VALUES ('x', 'k', ?)",
            (time.time() - limits.KEEP_SECONDS - 1,),
        )
        db.commit()
        assert limits.allow(db, "x", "k", 1, limits.KEEP_SECONDS * 2) is True
        assert db.execute("SELECT COUNT(*) FROM rate_events").fetchone()[0] == 1


def test_stored_reporter_key_doesnt_contain_the_ip(app, client, post):
    report(client, post["id"], ip="123.45.67.89")
    with app.app_context():
        dump = "\n".join(get_db().iterdump())
    assert "123.45.67.89" not in dump


def test_pause_switches(client, post, cli, capsys):
    assert cli(["pause", "all"]) == 0
    capsys.readouterr()
    assert cli(["status"]) == 0
    assert capsys.readouterr().out.count("PAUSED") == 2

    upload = client.post(
        "/api/images",
        data={"image": (io.BytesIO(solid_png(HORSE_COLORS[0])), "x.png")},
        headers=auth(post["bob"]),
        content_type="multipart/form-data",
    )
    signup = client.post(
        "/api/auth/register", json={"username": "newbie", "password": "password123"}
    )
    assert upload.status_code == 503 and signup.status_code == 503
    assert feed_ids(client) == [post["id"]], "the feed stays up"

    assert cli(["unpause", "uploads"]) == 0
    upload_png(client, post["bob"], solid_png(HORSE_COLORS[1]))
    assert client.post(
        "/api/auth/register", json={"username": "newbie", "password": "password123"}
    ).status_code == 503, "signups are still paused"


# --- the content check -----------------------------------------------------


class FakeOpenAI:
    """Stands in for urllib.request.urlopen."""

    def __init__(self, categories=None, error=None):
        self.categories = categories or {}
        self.error = error
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        body = json.dumps({"results": [{"flagged": True, "categories": self.categories}]})
        return io.BytesIO(body.encode())


@pytest.fixture
def openai(app, monkeypatch):
    app.config["MODERATION_PROVIDER"] = "openai"
    app.config["OPENAI_API_KEY"] = "test-key"

    def install(**kwargs):
        fake = FakeOpenAI(**kwargs)
        monkeypatch.setattr(safety.urllib.request, "urlopen", fake)
        return fake

    return install


def send_upload(client, token, caption=None):
    data = {"image": (io.BytesIO(solid_png(HORSE_COLORS[0])), "x.png")}
    if caption:
        data["caption"] = caption
    return client.post(
        "/api/images", data=data, headers=auth(token), content_type="multipart/form-data"
    )


def test_a_clean_upload_passes_and_the_request_is_well_formed(client, fixed_score, openai):
    fake = openai(categories={"sexual": False, "violence": False})
    token = register(client, "alice")
    fixed_score["value"] = 4.0

    assert send_upload(client, token, caption="my horse").status_code == 201

    request = fake.requests[0]
    assert request.full_url == "https://api.openai.com/v1/moderations"
    assert request.get_header("Authorization") == "Bearer test-key"
    body = json.loads(request.data)
    assert body["model"] == "omni-moderation-latest"
    assert body["input"][0] == {"type": "text", "text": "my horse"}
    assert body["input"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_a_rejected_upload_is_never_sent_to_the_check(client, fixed_score, openai):
    fake = openai()
    token = register(client, "alice")
    fixed_score["value"] = 0.0
    assert send_upload(client, token).status_code == 201
    assert fake.requests == []


def test_a_blocked_upload_stores_nothing_but_a_record(app, client, fixed_score, openai, cli, capsys):
    openai(categories={"sexual": True, "violence": False})
    token = register(client, "alice")
    fixed_score["value"] = 4.0

    resp = send_upload(client, token)

    assert resp.status_code == 422
    assert files_in(app.config["UPLOAD_DIR"]) == []
    assert feed_ids(client) == []
    assert client.get("/api/images/mine", headers=auth(token)).get_json()["items"] == []
    assert cli(["blocks"]) == 0
    out = capsys.readouterr().out
    assert "alice" in out and "sexual" in out


@pytest.mark.parametrize("category", ["violence", "harassment", "illicit"])
def test_milder_categories_dont_block(client, fixed_score, openai, category):
    openai(categories={category: True})
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 201


def test_repeated_blocks_stop_a_user_uploading(app, client, fixed_score, openai):
    openai(categories={"violence/graphic": True})
    app.config["UPLOAD_BLOCKS_PER_USER_DAY"] = 2
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 422
    assert send_upload(client, token).status_code == 422
    assert send_upload(client, token).status_code == 429


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("down"), TimeoutError("slow"), ValueError("not json")],
)
def test_the_upload_is_refused_if_the_check_fails(app, client, fixed_score, openai, error):
    openai(error=error)
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 503
    assert files_in(app.config["UPLOAD_DIR"]) == []


def test_the_upload_is_refused_if_openai_sends_nonsense(app, client, fixed_score, monkeypatch, openai):
    openai()
    monkeypatch.setattr(safety.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'{"results": [{}]}'))
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 503


def test_no_key_means_refused_not_unchecked(app, client, fixed_score, cli, capsys):
    app.config["MODERATION_PROVIDER"] = "openai"
    app.config["OPENAI_API_KEY"] = ""
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 503
    assert cli(["status"]) == 0
    assert "NO KEY" in capsys.readouterr().out


def test_an_unknown_provider_is_refused_not_unchecked(app, client, fixed_score):
    app.config["MODERATION_PROVIDER"] = "typo"
    token = register(client, "alice")
    fixed_score["value"] = 4.0
    assert send_upload(client, token).status_code == 503


# --- alert emails ----------------------------------------------------------


class FakeSMTP:
    sent: list = []

    def __init__(self, host, port, timeout=None):
        self.address = (host, port)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, message):
        FakeSMTP.sent.append(message)


@pytest.fixture
def mail(app, monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    app.config.update(ALERT_EMAIL="me@example.com", SMTP_HOST="smtp.example.com")
    threads = []
    real_notify = alerts.notify

    def tracked(*args, **kwargs):
        thread = real_notify(*args, **kwargs)
        if thread:
            threads.append(thread)
        return thread

    monkeypatch.setattr(alerts, "notify", tracked)

    def sent():
        for thread in threads:
            thread.join(5)
        return FakeSMTP.sent

    return sent


def test_a_report_emails_the_operator_without_quoting_it(client, post, mail):
    report(client, post["id"], reason="child_safety", details="SECRET-WORDS", contact="x@y.z")
    message = mail()[0]
    assert message["To"] == "me@example.com"
    assert "child_safety" in message["Subject"]
    body = message.get_content()
    assert post["id"] in body and "hidden" in body
    assert "SECRET-WORDS" not in body and "x@y.z" not in body


def test_alerts_are_rate_limited(client, post, mail):
    for n in range(alerts.ALERTS_PER_HOUR + 3):
        report(client, details=f"complaint {n}", reason="other", ip=f"10.0.0.{n}")
    assert len(mail()) == alerts.ALERTS_PER_HOUR


def test_urgent_alerts_get_through_when_others_are_being_flooded(client, post, mail):
    for n in range(alerts.ALERTS_PER_HOUR + 1):
        report(client, details=f"complaint {n}", reason="other", ip=f"10.0.0.{n}")
    report(client, post["id"], reason="child_safety")
    assert any("child_safety" in m["Subject"] for m in mail())


def test_a_failed_email_doesnt_break_the_report(client, post, mail, monkeypatch):
    def refuse(*args, **kwargs):
        raise smtplib.SMTPException("no")

    monkeypatch.setattr(alerts.smtplib, "SMTP", refuse)
    assert report(client, post["id"]).status_code == 201
    mail()


def test_no_alert_settings_means_no_email(client, post, app, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not try to send")

    monkeypatch.setattr(alerts.smtplib, "SMTP", explode)
    assert app.config["ALERT_EMAIL"] == ""
    assert report(client, post["id"]).status_code == 201
