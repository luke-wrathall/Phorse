"""Feed strategies (config FEED_STRATEGY): blend / contested / consensus / top,
and feed sessions (loaded posts never move; new pages ranked fresh)."""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from conftest import auth, register

# image -> (model score, ratings from rater0..rater3; None = skipped).
# Scores deliberately disagree with user ratings: "highest scored" must
# follow the model, not the raters.
CATALOGUE = {
    "A": (4.5, [5, 5, 5, 5]),
    "B": (3.5, [0, 5, 0, 5]),  # contested
    "C": (2.6, [5, 5, 5, 5]),  # raters love it, model doesn't
    "D": (2.7, [5, None, None, None]),
    "E": (3.0, [None, None, None, None]),  # unrated, newest
}
BY_MODEL_SCORE = ["A", "B", "E", "D", "C"]


@pytest.fixture
def catalogue(app, client):
    """Seed CATALOGUE directly (A oldest ... E newest). Returns rater tokens."""
    from server.db import get_db

    tokens = [register(client, f"rater{i}") for i in range(4)]
    now = time.time()
    with app.app_context():
        db = get_db()
        user_ids = [
            db.execute("SELECT id FROM users WHERE username = ?", (f"rater{i}",)).fetchone()["id"]
            for i in range(4)
        ]
        for age, (image_id, (score, ratings)) in enumerate(reversed(list(CATALOGUE.items()))):
            created = now - 60 * (age + 1)
            db.execute(
                "INSERT INTO images (id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
                "VALUES (?, ?, ?, ?, NULL, ?, ?, 1, ?)",
                (image_id, user_ids[0], f"{image_id}.png", ".png",
                 np.zeros(512, np.float32).tobytes(), score, created),
            )
            for user_id, rating in zip(user_ids, ratings):
                if rating is not None:
                    db.execute(
                        "INSERT INTO ratings (image_id, user_id, rating, created_at) VALUES (?, ?, ?, ?)",
                        (image_id, user_id, float(rating), created),
                    )
        db.commit()
    return tokens


def _get(client, token=None, session=None, limit=50, offset=0) -> dict:
    url = f"/api/feed?limit={limit}&offset={offset}" + (f"&session={session}" if session else "")
    return client.get(url, headers=auth(token) if token else {}).get_json()


def _feed_ids(app, client, strategy: str, token: str | None = None) -> list[str]:
    app.config["FEED_STRATEGY"] = strategy
    ids = [item["id"] for item in _get(client, token)["items"]]
    assert sorted(ids) == sorted(CATALOGUE), f"every passed image exactly once: {ids}"
    return ids


def test_top_follows_model_score_not_ratings(app, client, catalogue):
    assert _feed_ids(app, client, "top") == BY_MODEL_SCORE


def test_consensus_interleaves_newest(app, client, catalogue):
    # Newest (E, D, C, ...) and most horse (A, B, E, ...) take turns, no repeats.
    assert _feed_ids(app, client, "consensus") == ["E", "A", "D", "B", "C"]


def test_contested_interleaves_newest(app, client, catalogue):
    ids = _feed_ids(app, client, "contested")
    assert ids[:2] == ["E", "B"]


def test_new_feed_only_shows_horses_the_viewer_hasnt_rated(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "top"
    # rater0 rated A-D; rater1 rated A-C (skipped D and E).
    assert [i["id"] for i in _get(client, catalogue[0])["items"]] == ["E"]
    assert [i["id"] for i in _get(client, catalogue[1])["items"]] == ["E", "D"]


def test_new_feed_logged_out_or_fresh_viewer_sees_everything(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "top"
    newbie = register(client, "newbie")
    assert [i["id"] for i in _get(client)["items"]] == BY_MODEL_SCORE
    assert [i["id"] for i in _get(client, newbie)["items"]] == BY_MODEL_SCORE


def test_new_feed_empty_once_youve_rated_everything(app, client, catalogue):
    token = catalogue[0]
    assert client.post("/api/images/E/ratings", json={"rating": 2}, headers=auth(token)).status_code == 200
    page = _get(client, token)
    assert page["items"] == [] and page["next_offset"] is None


def test_unknown_strategy_fails_loudly(app, client, catalogue):
    from server.services.feed import get_feed

    app.config["FEED_STRATEGY"] = "vibes"
    with app.app_context(), pytest.raises(ValueError, match="FEED_STRATEGY"):
        get_feed()


def test_rating_mid_scroll_keeps_session_order(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "top"
    token = register(client, "newbie")
    session = uuid.uuid4().hex

    first = _get(client, token, session, limit=2)
    assert [i["id"] for i in first["items"]] == ["A", "B"]

    for image in ("A", "D"):  # one loaded, one not yet
        resp = client.post(f"/api/images/{image}/ratings", json={"rating": 4.0}, headers=auth(token))
        assert resp.status_code == 200

    # the rating refetch (offset 0) keeps loaded posts in place, A included...
    refetch = _get(client, token, session, limit=2)
    assert [i["id"] for i in refetch["items"]] == ["A", "B"]
    assert refetch["items"][0]["rating_count"] == 5, "item details stay live"
    # ...the next page skips the now-rated D...
    rest = _get(client, token, session, limit=50, offset=first["next_offset"])
    assert [i["id"] for i in rest["items"]] == ["E", "C"]

    # ...and a new page load drops both rated horses.
    reloaded = [i["id"] for i in _get(client, token, uuid.uuid4().hex)["items"]]
    assert reloaded == ["B", "E", "C"]


def test_retrain_mid_scroll_only_affects_unloaded_pages(app, client, catalogue):
    from server.db import get_db

    app.config["FEED_STRATEGY"] = "top"
    session = uuid.uuid4().hex
    first = _get(client, session=session, limit=2)
    assert [i["id"] for i in first["items"]] == ["A", "B"]

    with app.app_context():
        db = get_db()
        db.execute("UPDATE images SET score = 5.0 WHERE id = 'C'")  # a retrain re-scores
        db.commit()

    # loaded posts stay exactly where they were...
    assert [i["id"] for i in _get(client, session=session, limit=2)["items"]] == ["A", "B"]
    # ...and the next page reflects the new scores.
    rest = _get(client, session=session, offset=first["next_offset"])
    assert [i["id"] for i in rest["items"]] == ["C", "E", "D"]


def test_new_upload_mid_scroll_appears_in_next_page(app, client, catalogue):
    from server.db import get_db

    app.config["FEED_STRATEGY"] = "top"
    session = uuid.uuid4().hex
    first = _get(client, session=session, limit=2)

    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO images (id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
            "VALUES ('F', 1, 'F.png', '.png', NULL, ?, 4.9, 1, ?)",
            (np.zeros(512, np.float32).tobytes(), time.time()),
        )
        db.commit()

    assert [i["id"] for i in _get(client, session=session, limit=2)["items"]] == ["A", "B"]
    rest = _get(client, session=session, offset=first["next_offset"])
    assert [i["id"] for i in rest["items"]] == ["F", "E", "D", "C"]


def test_refetching_every_loaded_page_keeps_next_offsets(app, client, catalogue):
    """TanStack refetches each loaded page in turn, chaining next_offset."""
    app.config["FEED_STRATEGY"] = "top"
    session = uuid.uuid4().hex
    pages, offset = [], 0
    while offset is not None:
        page = _get(client, session=session, limit=2, offset=offset)
        pages.append(page)
        offset = page["next_offset"]
    replay, offset = [], 0
    while offset is not None:
        page = _get(client, session=session, limit=2, offset=offset)
        replay.append(page)
        offset = page["next_offset"]
    assert [[i["id"] for i in p["items"]] for p in replay] == [
        [i["id"] for i in p["items"]] for p in pages
    ]
    assert [i["id"] for p in pages for i in p["items"]] == BY_MODEL_SCORE


def test_sessions_are_per_viewer(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "consensus"
    session = uuid.uuid4().hex
    rater0 = [i["id"] for i in _get(client, catalogue[0], session)["items"]]
    rater1 = [i["id"] for i in _get(client, catalogue[1], session)["items"]]
    assert rater0 != rater1, "same session string, different viewers -> separate orders"


def test_interleave_share_and_dedupe():
    from server.services.feed import _interleave

    def entries(*ids):
        return [{"item": {"id": i}} for i in ids]

    merged = _interleave(entries("p1", "p2", "u2", "p3"), entries("u1", "u2", "u3"), share=0.5)
    assert [e["item"]["id"] for e in merged] == ["u1", "p1", "u2", "p2", "u3", "p3"]


# --- the two feeds: new (unseen, FEED_STRATEGY) and ranked (all, most horse first)


def _ids(client, query: str, token=None) -> list[str]:
    resp = client.get(f"/api/feed?limit=50&{query}", headers=auth(token) if token else {})
    assert resp.status_code == 200, resp.get_json()
    return [item["id"] for item in resp.get_json()["items"]]


@pytest.mark.parametrize("strategy", ["blend", "contested", "consensus"])
def test_ranked_feed_is_most_horse_first_whatever_the_strategy(app, client, catalogue, strategy):
    app.config["FEED_STRATEGY"] = strategy
    assert _ids(client, "feed=ranked", catalogue[0]) == BY_MODEL_SCORE


def test_new_feed_is_the_default_and_uses_feed_strategy(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "consensus"
    assert _ids(client, "feed=new") == ["E", "A", "D", "B", "C"]
    assert _ids(client, "") == ["E", "A", "D", "B", "C"]


def test_ranked_feed_includes_horses_youve_rated(app, client, catalogue):
    assert _ids(client, "feed=ranked", catalogue[0]) == BY_MODEL_SCORE
    assert _ids(client, "feed=ranked", catalogue[0]) == _ids(client, "feed=ranked")


def test_feed_sessions_are_per_feed(app, client, catalogue):
    """One session id used on both feeds doesn't mix up their orders."""
    app.config["FEED_STRATEGY"] = "consensus"
    session = uuid.uuid4().hex
    token = catalogue[1]
    new = _ids(client, f"feed=new&session={session}", token)
    ranked = _ids(client, f"feed=ranked&session={session}", token)
    assert new == ["E", "D"]
    assert ranked == BY_MODEL_SCORE


def test_unknown_feed_is_rejected(client, catalogue):
    assert client.get("/api/feed?feed=vibes").status_code == 400
