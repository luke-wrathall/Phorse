"""Invite-only signup: codes, use limits, revocation, and the CLI."""

from __future__ import annotations

import pytest

from server.db import get_db
from server.invites import main as invites_cli
from server.services import invites


@pytest.fixture
def invite_app(app):
    app.config["REQUIRE_INVITE"] = True
    return app


@pytest.fixture
def client(invite_app):
    return invite_app.test_client()


def make_invite(app, max_uses=None, code=None) -> str:
    with app.app_context():
        return invites.create_invite(get_db(), max_uses, code=code)


def signup(client, username, invite=None):
    body = {"username": username, "password": "password123"}
    if invite is not None:
        body["invite"] = invite
    return client.post("/api/auth/register", json=body)


def uses(app, code) -> int:
    with app.app_context():
        return get_db().execute("SELECT uses FROM invites WHERE code = ?", (code,)).fetchone()[0]


def test_signup_without_invite_is_refused(client):
    assert signup(client, "alice").status_code == 403
    assert signup(client, "alice", "   ").status_code == 403


def test_unknown_invite_is_refused(invite_app, client):
    make_invite(invite_app)
    assert signup(client, "alice", "nope").status_code == 403


def test_invite_lets_you_sign_up_and_is_recorded(invite_app, client):
    code = make_invite(invite_app)
    # Codes are case- and whitespace-insensitive (typed off a phone).
    resp = signup(client, "alice", f"  {code.upper()} ")
    assert resp.status_code == 201, resp.get_json()
    with invite_app.app_context():
        row = get_db().execute("SELECT invite_code FROM users WHERE username = 'alice'").fetchone()
    assert row["invite_code"] == code
    assert uses(invite_app, code) == 1
    # Logging in afterwards needs no code.
    resp = client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert resp.status_code == 200


def test_invite_stops_at_max_uses(invite_app, client):
    code = make_invite(invite_app, max_uses=2)
    assert signup(client, "alice", code).status_code == 201
    assert signup(client, "bob", code).status_code == 201
    assert signup(client, "carol", code).status_code == 403
    assert uses(invite_app, code) == 2


def test_taken_username_does_not_use_up_the_invite(invite_app, client):
    code = make_invite(invite_app, max_uses=2)
    assert signup(client, "alice", code).status_code == 201
    assert signup(client, "alice", code).status_code == 409
    assert uses(invite_app, code) == 1
    assert signup(client, "bob", code).status_code == 201


def test_revoked_invite_is_refused_but_existing_accounts_keep_working(invite_app, client):
    code = make_invite(invite_app)
    assert signup(client, "alice", code).status_code == 201
    with invite_app.app_context():
        assert invites.revoke_invite(get_db(), code)
    assert signup(client, "bob", code).status_code == 403
    resp = client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert resp.status_code == 200


def test_invite_ignored_when_not_required(app):
    client = app.test_client()  # conftest default: REQUIRE_INVITE off
    assert signup(client, "alice").status_code == 201
    assert signup(client, "bob", "whatever").status_code == 201


@pytest.mark.parametrize("bad", ["has space", "x" * 65, "semi;colon"])
def test_create_rejects_bad_custom_codes(invite_app, bad):
    with invite_app.app_context(), pytest.raises(ValueError):
        invites.create_invite(get_db(), None, code=bad)


def test_create_rejects_duplicate_codes(invite_app):
    make_invite(invite_app, code="horses")
    with invite_app.app_context(), pytest.raises(ValueError):
        invites.create_invite(get_db(), None, code="HORSES")


def test_cli_create_list_revoke(invite_app, monkeypatch, capsys):
    # The CLI builds its own app from env config; hand it the test app instead.
    monkeypatch.setattr("server.invites.create_app", lambda: invite_app)
    invite_app.config["SITE_URL"] = "https://phorse.example"
    link = "https://phorse.example/join/insta-story"

    assert invites_cli(["create", "--code", "Insta-Story", "--uses", "300", "--note", "story"]) == 0
    assert capsys.readouterr().out.strip() == link

    assert invites_cli(["list"]) == 0
    out = capsys.readouterr().out
    assert "insta-story" in out and "0/300" in out and "story" in out and link in out

    assert invites_cli(["revoke", "insta-story"]) == 0
    assert invites_cli(["list"]) == 0
    out = capsys.readouterr().out
    assert "REVOKED" in out and link not in out

    assert invites_cli(["revoke", "missing"]) == 1
    assert invites_cli(["create", "--uses", "0"]) == 1
