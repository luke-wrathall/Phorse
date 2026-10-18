"""Moderation CLI: delete posts, ban/unban, set passwords."""

from __future__ import annotations

import os
import time

import jwt
import pytest

from conftest import HORSE_COLORS, auth, register, solid_png, upload_png
from horse.storage import connect as connect_training_db
from horse.ingest import ingest
from server.admin import main as admin_cli
from server.db import get_db


@pytest.fixture
def cli(app, monkeypatch):
    # The CLI builds its own app from env config; hand it the test app instead.
    monkeypatch.setattr("server.admin.create_app", lambda: app)
    return admin_cli


@pytest.fixture
def posted(app, client):
    """bob posts an image, alice and bob rate it, alice posts one too;
    ratings synced into the training store."""
    bob = register(client, "bob")
    alice = register(client, "alice")
    bobs = upload_png(client, bob, solid_png(HORSE_COLORS[0]))["id"]
    alices = upload_png(client, alice, solid_png(HORSE_COLORS[1]))["id"]
    for token, image, value in [(alice, bobs, 4), (bob, bobs, 5), (bob, alices, 1)]:
        resp = client.post(f"/api/images/{image}/ratings", json={"rating": value}, headers=auth(token))
        assert resp.status_code in (200, 201), resp.get_json()
    ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
    return {"bob": bob, "alice": alice, "bobs": bobs, "alices": alices}


def training_ratings(app) -> list[tuple[str, str]]:
    conn = connect_training_db(app.config["TRAINING_DB"])
    try:
        return sorted(conn.execute("SELECT image_id, user_id FROM ratings").fetchall())
    finally:
        conn.close()


def count(app, sql, *params) -> int:
    with app.app_context():
        return get_db().execute(sql, params).fetchone()[0]


def login(client, username, password="password123"):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def test_delete_post_by_url_removes_row_ratings_file_and_training_rows(app, client, cli, posted):
    image = posted["bobs"]
    path = os.path.join(app.config["UPLOAD_DIR"], f"{image}.png")
    assert os.path.exists(path)

    assert cli(["delete-post", f"https://phorse.example/api/images/{image}/file"]) == 0

    assert count(app, "SELECT COUNT(*) FROM images WHERE id = ?", image) == 0
    assert count(app, "SELECT COUNT(*) FROM ratings WHERE image_id = ?", image) == 0
    assert not os.path.exists(path)
    assert all(img != image for img, _ in training_ratings(app))
    assert client.get(f"/api/images/{image}/file").status_code == 404
    assert image not in [i["id"] for i in client.get("/api/feed").get_json()["items"]]
    # A re-sync doesn't bring it back.
    ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
    assert all(img != image for img, _ in training_ratings(app))


def test_delete_unknown_post_fails(cli, posted):
    assert cli(["delete-post", "0" * 32]) == 1
    assert cli(["delete-post", "not-an-id"]) == 1


def test_ban_blocks_login_and_live_tokens_then_unban_restores(client, cli, posted):
    bob = posted["bob"]
    assert client.get("/api/auth/me", headers=auth(bob)).status_code == 200

    assert cli(["ban", "bob"]) == 0
    assert client.get("/api/auth/me", headers=auth(bob)).status_code == 403
    resp = client.post(f"/api/images/{posted['alices']}/ratings", json={"rating": 3}, headers=auth(bob))
    assert resp.status_code == 403
    assert login(client, "bob").status_code == 403
    # A wrong password still just says wrong password.
    assert login(client, "bob", "nope-nope-nope").status_code == 401

    assert cli(["unban", "bob"]) == 0
    assert login(client, "bob").status_code == 200
    assert client.get("/api/auth/me", headers=auth(bob)).status_code == 200


def test_ban_without_purge_keeps_their_posts_and_ratings(app, cli, posted):
    assert cli(["ban", "bob"]) == 0
    assert count(app, "SELECT COUNT(*) FROM images WHERE id = ?", posted["bobs"]) == 1
    assert len(training_ratings(app)) == 3


def test_ban_purge_removes_their_posts_and_votes_only(app, client, cli, posted):
    assert cli(["ban", "bob", "--purge"]) == 0
    bob_id = count(app, "SELECT id FROM users WHERE username = 'bob'")

    assert count(app, "SELECT COUNT(*) FROM images WHERE owner_id = ?", bob_id) == 0
    assert count(app, "SELECT COUNT(*) FROM ratings WHERE user_id = ?", bob_id) == 0
    # Alice's post survives; bob's vote on it is gone, from the training store too.
    assert count(app, "SELECT COUNT(*) FROM images WHERE id = ?", posted["alices"]) == 1
    assert training_ratings(app) == []
    ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
    assert training_ratings(app) == []


def test_set_password_changes_it_and_logs_out_old_tokens(app, client, cli, posted, monkeypatch):
    # A login from a minute ago (tokens are compared by whole second).
    alice_id = count(app, "SELECT id FROM users WHERE username = 'alice'")
    now = int(time.time())
    old = jwt.encode(
        {"sub": str(alice_id), "iat": now - 60, "exp": now + 3600},
        app.config["SECRET_KEY"],
        algorithm="HS256",
    )
    assert client.get("/api/auth/me", headers=auth(old)).status_code == 200
    answers = iter(["new-password-1", "new-password-1"])
    monkeypatch.setattr("server.admin.getpass.getpass", lambda prompt: next(answers))

    assert cli(["set-password", "alice"]) == 0
    assert client.get("/api/auth/me", headers=auth(old)).status_code == 401
    assert login(client, "alice").status_code == 401
    resp = login(client, "alice", "new-password-1")
    assert resp.status_code == 200
    assert client.get("/api/auth/me", headers=auth(resp.get_json()["token"])).status_code == 200


@pytest.mark.parametrize("answers", [["short", "short"], ["new-password-1", "different-1"]])
def test_set_password_rejects_short_or_mismatched(client, cli, posted, monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("server.admin.getpass.getpass", lambda prompt: next(it))
    assert cli(["set-password", "alice"]) == 1
    assert login(client, "alice").status_code == 200


def test_unknown_user_fails(cli):
    for cmd in (["ban", "ghost"], ["unban", "ghost"], ["set-password", "ghost"], ["posts", "ghost"]):
        assert cli(cmd) == 1


def test_listings(cli, posted, capsys):
    assert cli(["users"]) == 0
    out = capsys.readouterr().out
    assert "alice" in out and "bob" in out and "1 posts" in out

    assert cli(["posts", "bob"]) == 0
    out = capsys.readouterr().out
    assert posted["bobs"] in out and posted["alices"] not in out
