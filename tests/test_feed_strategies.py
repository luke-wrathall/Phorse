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


def test_top_is_the_same_for_everyone(app, client, catalogue):
    assert _feed_ids(app, client, "top", catalogue[0]) == _feed_ids(app, client, "top")


def test_consensus_interleaves_viewers_unrated(app, client, catalogue):
    # rater0 rated A-D, so only E is "unseen" for them: it leads, then by model score.
    assert _feed_ids(app, client, "consensus", catalogue[0]) == ["E", "A", "B", "D", "C"]


def test_contested_interleaves_viewers_unrated(app, client, catalogue):
    ids = _feed_ids(app, client, "contested", catalogue[0])
    assert ids[:2] == ["E", "B"]


def test_unrated_depends_on_viewer(app, client, catalogue):
    # rater1 skipped D, so D (low model score) is pulled up into their unseen slots.
    assert _feed_ids(app, client, "consensus", catalogue[1])[:3] == ["E", "A", "D"]


def test_logged_out_counts_nothing_as_rated(app, client, catalogue):
    ids = _feed_ids(app, client, "consensus")
    assert ids[:2] == ["E", "A"]


def test_unknown_strategy_fails_loudly(app, client, catalogue):
    from server.services.feed import get_feed

    app.config["FEED_STRATEGY"] = "vibes"
    with app.app_context(), pytest.raises(ValueError, match="FEED_STRATEGY"):
        get_feed()


def test_rating_mid_scroll_keeps_session_order(app, client, catalogue):
    app.config["FEED_STRATEGY"] = "consensus"
    token = catalogue[0]
    session = uuid.uuid4().hex

    first = _get(client, token, session, limit=2)
    assert [i["id"] for i in first["items"]] == ["E", "A"]

    resp = client.post("/api/images/E/ratings", json={"rating": 4.0}, headers=auth(token))
    assert resp.status_code == 200

    # the rating refetch (offset 0) keeps loaded posts in place...
    refetch = _get(client, token, session, limit=2)
    assert [i["id"] for i in refetch["items"]] == ["E", "A"]
    assert refetch["items"][0]["rating_count"] == 1, "item details stay live"
    rest = _get(client, token, session, limit=50, offset=first["next_offset"])
    assert [i["id"] for i in rest["items"]] == ["B", "D", "C"]

    # ...and a new page load ranks E by model score like any rated horse.
    reloaded = [i["id"] for i in _get(client, token, uuid.uuid4().hex)["items"]]
    assert reloaded == BY_MODEL_SCORE


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
