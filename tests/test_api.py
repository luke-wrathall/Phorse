"""HTTP API contract: auth, ratings, uploads, own-images, file serving, feed
parameters. The drift loop itself is covered in test_core_loop.py."""

from __future__ import annotations

import io

import jwt
import pytest
from PIL import Image

from conftest import HORSE_COLORS, auth, register, solid_png, upload_png


@pytest.fixture
def fixed_score(monkeypatch):
    """Stub the head so every upload scores `fixed_score["value"]`."""
    from server.services import model

    state = {"value": 4.0}
    monkeypatch.setattr(model, "score_embedding", lambda _e: state["value"])
    return state


def _upload(client, token, png=None, name="img.png", caption=None):
    data = {"image": (io.BytesIO(png or solid_png(HORSE_COLORS[0])), name)}
    if caption is not None:
        data["caption"] = caption
    return client.post(
        "/api/images", data=data, headers=auth(token), content_type="multipart/form-data"
    )


def _rate(client, token, image_id, body):
    return client.post(f"/api/images/{image_id}/ratings", json=body, headers=auth(token))


# --- auth ------------------------------------------------------------------


def test_register_login_and_me(client):
    token = register(client, "  alice  ")
    me = client.get("/api/auth/me", headers=auth(token)).get_json()
    assert me["user"]["username"] == "alice", "username is stripped"

    resp = client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert resp.status_code == 200
    assert client.get("/api/auth/me", headers=auth(resp.get_json()["token"])).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"username": "ab", "password": "password123"},
        {"username": "alice", "password": "short"},
        {"username": "   ", "password": "password123"},
        {},
    ],
)
def test_register_rejects_bad_credentials(client, body):
    resp = client.post("/api/auth/register", json=body)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_register_duplicate_username_conflicts(client):
    register(client, "alice")
    resp = client.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "username, password", [("alice", "wrongpassword"), ("nobody", "password123")]
)
def test_login_failures_are_indistinguishable(client, username, password):
    register(client, "alice")
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "incorrect username or password"


def _forged_token(app, secret: str, exp_offset: int) -> str:
    import time

    now = int(time.time())
    return jwt.encode({"sub": "1", "iat": now, "exp": now + exp_offset}, secret, algorithm="HS256")


@pytest.mark.parametrize("case", ["missing", "garbage", "wrong_secret", "expired", "no_bearer"])
def test_me_rejects_bad_tokens(app, client, case):
    register(client, "alice")
    headers = {
        "missing": {},
        "garbage": auth("not.a.jwt"),
        "wrong_secret": auth(_forged_token(app, "some-other-secret-" + "x" * 32, 3600)),
        "expired": auth(_forged_token(app, app.config["SECRET_KEY"], -10)),
        "no_bearer": {"Authorization": "Token abc"},
    }[case]
    assert client.get("/api/auth/me", headers=headers).status_code == 401


# --- ratings ---------------------------------------------------------------


@pytest.fixture
def rated_setup(client, fixed_score):
    owner = register(client, "owner")
    image_id = upload_png(client, owner, solid_png(HORSE_COLORS[0]))["id"]
    return owner, image_id


def test_rating_requires_auth(client, rated_setup):
    _, image_id = rated_setup
    assert client.post(f"/api/images/{image_id}/ratings", json={"rating": 3}).status_code == 401


@pytest.mark.parametrize(
    "body",
    [{}, {"rating": None}, {"rating": "lots"}, {"rating": -0.1}, {"rating": 5.1},
     {"rating": "nan"}, {"rating": "inf"}],
)
def test_rating_rejects_bad_values(client, rated_setup, body):
    owner, image_id = rated_setup
    assert _rate(client, owner, image_id, body).status_code == 400


@pytest.mark.parametrize("value", [0, 5, 2.5])
def test_rating_accepts_scale_bounds(client, rated_setup, value):
    owner, image_id = rated_setup
    resp = _rate(client, owner, image_id, {"rating": value})
    assert resp.status_code == 200
    assert resp.get_json()["rating_mean"] == pytest.approx(value)


def test_rating_unknown_image_404s(client, rated_setup):
    owner, _ = rated_setup
    assert _rate(client, owner, "doesnotexist", {"rating": 3}).status_code == 404


def test_rerating_replaces_previous_vote(client, rated_setup):
    owner, image_id = rated_setup
    other = register(client, "other")
    _rate(client, owner, image_id, {"rating": 1})
    resp = _rate(client, owner, image_id, {"rating": 5}).get_json()
    assert (resp["rating_count"], resp["rating_mean"]) == (1, 5.0)

    resp = _rate(client, other, image_id, {"rating": 3}).get_json()
    assert (resp["rating_count"], resp["rating_mean"]) == (2, 4.0)


# --- uploads ---------------------------------------------------------------


def test_upload_requires_auth(client):
    resp = client.post(
        "/api/images",
        data={"image": (io.BytesIO(solid_png(HORSE_COLORS[0])), "a.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_upload_without_file_is_rejected(client):
    token = register(client, "alice")
    resp = client.post("/api/images", data={}, headers=auth(token), content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_of_non_image_is_rejected(client):
    token = register(client, "alice")
    resp = _upload(client, token, png=b"definitely not an image", name="x.png")
    assert resp.status_code == 400
    assert resp.get_json()["error"].startswith("could not read image")


def _noise_png(size: tuple[int, int]) -> bytes:
    import numpy as np

    pixels = np.random.default_rng(0).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_over_byte_limit_is_rejected_as_json(app, client):
    token = register(client, "alice")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    png = _noise_png((700, 700))  # noise doesn't compress: ~1.4 MB
    assert len(png) > 1024 * 1024

    resp = _upload(client, token, png=png)

    assert resp.status_code == 413
    assert resp.get_json()["error"] == "file too big -- the limit is 1 MB"


def test_upload_over_pixel_limit_is_rejected(app, client, fixed_score):
    token = register(client, "alice")
    app.config["MAX_UPLOAD_PIXELS"] = 100

    small = _upload(client, token, png=solid_png(HORSE_COLORS[0], size=(10, 10)))
    big = _upload(client, token, png=solid_png(HORSE_COLORS[0], size=(10, 11)))

    assert small.status_code == 201, "exactly at the limit is allowed"
    assert big.status_code == 413
    assert "megapixel limit" in big.get_json()["error"]
    mine = client.get("/api/images/mine", headers=auth(token)).get_json()["items"]
    assert len(mine) == 1, "nothing stored for the rejected upload"


def test_upload_limits_are_published_for_the_client(app, client):
    app.config.update(MAX_CONTENT_LENGTH=123, MAX_UPLOAD_PIXELS=456)
    resp = client.get("/api/images/limits")
    assert resp.status_code == 200, "no login needed"
    assert resp.get_json() == {"max_bytes": 123, "max_pixels": 456}


def test_upload_reports_untrained_head_until_a_retrain(app, client):
    from server.services.retrain import run_retrain

    token = register(client, "alice")
    assert upload_png(client, token, solid_png(HORSE_COLORS[0]))["head_loaded"] is False
    with app.app_context():
        run_retrain()
    assert upload_png(client, token, solid_png(HORSE_COLORS[1]))["head_loaded"] is True


def test_upload_verdict_uses_the_gate_threshold(app, client, fixed_score):
    token = register(client, "alice")
    threshold = app.config["GATE_THRESHOLD"]

    fixed_score["value"] = threshold
    at = upload_png(client, token, solid_png(HORSE_COLORS[0]))
    fixed_score["value"] = threshold - 0.01
    below = upload_png(client, token, solid_png(HORSE_COLORS[1]))

    assert at["passed"] is True, "score == threshold passes"
    assert below["passed"] is False
    assert at["threshold"] == threshold


def test_rejected_upload_is_in_mine_but_not_the_feed(client, fixed_score):
    token = register(client, "alice")
    fixed_score["value"] = 0.0
    image_id = upload_png(client, token, solid_png(HORSE_COLORS[0]))["id"]

    mine = client.get("/api/images/mine", headers=auth(token)).get_json()["items"]
    assert [(i["id"], i["passed"]) for i in mine] == [(image_id, False)]
    feed = client.get("/api/feed").get_json()["items"]
    assert image_id not in [i["id"] for i in feed]


@pytest.mark.parametrize(
    "caption, stored",
    [("  a horse  ", "a horse"), ("   ", None), ("", None), ("h" * 300, "h" * 280)],
)
def test_caption_is_cleaned(client, fixed_score, caption, stored):
    token = register(client, "alice")
    assert _upload(client, token, caption=caption).status_code == 201
    (item,) = client.get("/api/images/mine", headers=auth(token)).get_json()["items"]
    assert item["caption"] == stored


def test_mine_lists_only_own_uploads_newest_first_with_aggregates(client, fixed_score):
    alice = register(client, "alice")
    bob = register(client, "bob")
    first = upload_png(client, alice, solid_png(HORSE_COLORS[0]))["id"]
    second = upload_png(client, alice, solid_png(HORSE_COLORS[1]))["id"]
    upload_png(client, bob, solid_png(HORSE_COLORS[2]))
    _rate(client, bob, first, {"rating": 4})
    _rate(client, alice, first, {"rating": 2})

    items = client.get("/api/images/mine", headers=auth(alice)).get_json()["items"]

    assert [i["id"] for i in items] == [second, first]
    by_id = {i["id"]: i for i in items}
    assert (by_id[first]["rating_count"], by_id[first]["rating_mean"]) == (2, 3.0)
    assert (by_id[second]["rating_count"], by_id[second]["rating_mean"]) == (0, None)
    assert by_id[first]["image_url"] == f"/api/images/{first}/file"


def test_mine_requires_auth(client):
    assert client.get("/api/images/mine").status_code == 401


def test_file_is_served_with_lowercased_extension(app, client, fixed_score):
    token = register(client, "alice")
    image_id = _upload(client, token, name="PHOTO.PNG").get_json()["id"]

    resp = client.get(f"/api/images/{image_id}/file")

    assert resp.status_code == 200
    assert Image.open(io.BytesIO(resp.data)).size == (8, 8)
    resp.close()


def test_unknown_file_404s_as_json(client):
    resp = client.get("/api/images/nope/file")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# --- feed endpoint ---------------------------------------------------------

# Everything an item may carry. Adding a field should be a deliberate choice:
# the UI must not surface the drift mechanism (no variance / pool / weight).
FEED_ITEM_KEYS = {
    "id", "owner", "score", "caption", "created_at", "rating_count", "rating_mean", "image_url",
}


@pytest.fixture
def three_horses(client, fixed_score):
    token = register(client, "alice")
    return [upload_png(client, token, solid_png(c))["id"] for c in HORSE_COLORS]


@pytest.mark.parametrize("strategy", ["blend", "contested", "consensus", "top"])
def test_feed_items_expose_no_drift_fields(app, client, three_horses, strategy):
    app.config["FEED_STRATEGY"] = strategy
    voter = register(client, "voter")
    _rate(client, voter, three_horses[0], {"rating": 1})
    items = client.get("/api/feed", headers=auth(voter)).get_json()["items"]
    assert items and all(set(i) == FEED_ITEM_KEYS for i in items)


def test_feed_pages_cover_catalogue_exactly_once(client, three_horses):
    seen, offset = [], 0
    while offset is not None:
        page = client.get(f"/api/feed?limit=2&offset={offset}").get_json()
        seen += [i["id"] for i in page["items"]]
        offset = page["next_offset"]
    assert sorted(seen) == sorted(three_horses)


@pytest.mark.parametrize(
    "query, n_items", [("limit=0", 1), ("limit=-3", 1), ("limit=999", 3), ("offset=-5&limit=2", 2)]
)
def test_feed_clamps_limit_and_offset(client, three_horses, query, n_items):
    resp = client.get(f"/api/feed?{query}")
    assert resp.status_code == 200
    assert len(resp.get_json()["items"]) == n_items


def test_feed_limit_is_capped(app, client, fixed_score):
    from server.services.feed import MAX_LIMIT

    token = register(client, "alice")
    for i in range(MAX_LIMIT + 2):
        upload_png(client, token, solid_png((i, 100, i)))
    page = client.get("/api/feed?limit=500").get_json()
    assert len(page["items"]) == MAX_LIMIT
    assert page["next_offset"] == MAX_LIMIT


@pytest.mark.parametrize("query", ["limit=abc", "offset=1.5", "session=" + "s" * 65])
def test_feed_rejects_bad_params(client, query):
    assert client.get(f"/api/feed?{query}").status_code == 400


def test_feed_with_bad_token_falls_back_to_logged_out(client, three_horses):
    resp = client.get("/api/feed", headers=auth("garbage"))
    assert resp.status_code == 200
    assert len(resp.get_json()["items"]) == 3


def test_health_and_unknown_route(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}
    resp = client.get("/api/nope")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "not found"}
