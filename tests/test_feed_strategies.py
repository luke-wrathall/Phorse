"""Feed strategies (config FEED_STRATEGY): blend / contested / consensus / top."""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import auth, register

# image -> ratings from rater0..rater3 (None = that rater skipped it)
CATALOGUE = {
    "A": [5, 5, 5, 5],  # consensus horse
    "B": [0, 5, 0, 5],  # contested
    "C": [1, 1, 1, 1],  # consensus not-horse
    "D": [5, None, None, None],  # one 5-star vote
    "E": [None, None, None, None],  # unrated, newest
}


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
        for age, (image_id, ratings) in enumerate(reversed(list(CATALOGUE.items()))):
            created = now - 60 * (age + 1)
            db.execute(
                "INSERT INTO images (id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
                "VALUES (?, ?, ?, ?, NULL, ?, 3.0, 1, ?)",
                (image_id, user_ids[0], f"{image_id}.png", ".png",
                 np.zeros(512, np.float32).tobytes(), created),
            )
            for user_id, rating in zip(user_ids, ratings):
                if rating is not None:
                    db.execute(
                        "INSERT INTO ratings (image_id, user_id, rating, created_at) VALUES (?, ?, ?, ?)",
                        (image_id, user_id, float(rating), created),
                    )
        db.commit()
    return tokens


def _feed_ids(app, client, strategy: str, token: str | None = None) -> list[str]:
    app.config["FEED_STRATEGY"] = strategy
    headers = auth(token) if token else {}
    items = client.get("/api/feed?limit=50", headers=headers).get_json()["items"]
    ids = [item["id"] for item in items]
    assert sorted(ids) == sorted(CATALOGUE), f"every passed image exactly once: {ids}"
    return ids


def test_top_is_shrunk_mean(app, client, catalogue):
    ids = _feed_ids(app, client, "top")
    assert ids[0] == "A"
    assert ids.index("A") < ids.index("D"), "one 5-star vote shouldn't beat four"
    assert ids[-1] == "C"


def test_top_is_the_same_for_everyone(app, client, catalogue):
    assert _feed_ids(app, client, "top", catalogue[0]) == _feed_ids(app, client, "top")


def test_consensus_interleaves_viewers_unrated(app, client, catalogue):
    # rater0 rated A-D, so only E is "unseen" for them: it leads, then top-rated.
    ids = _feed_ids(app, client, "consensus", catalogue[0])
    assert ids[:2] == ["E", "A"]


def test_contested_interleaves_viewers_unrated(app, client, catalogue):
    ids = _feed_ids(app, client, "contested", catalogue[0])
    assert ids[:2] == ["E", "B"]


def test_unrated_depends_on_viewer(app, client, catalogue):
    # rater1 skipped D, so D (newer than A-C) is unseen for them too.
    ids = _feed_ids(app, client, "consensus", catalogue[1])
    assert ids[:3] == ["E", "A", "D"]


def test_logged_out_counts_nothing_as_rated(app, client, catalogue):
    ids = _feed_ids(app, client, "consensus")
    assert ids[0] == "E" and ids[1] == "A"


def test_unknown_strategy_fails_loudly(app, client, catalogue):
    from server.services.feed import get_feed

    app.config["FEED_STRATEGY"] = "vibes"
    with app.app_context(), pytest.raises(ValueError, match="FEED_STRATEGY"):
        get_feed()


def test_interleave_share_and_dedupe():
    from server.services.feed import _interleave

    def entries(*ids):
        return [{"item": {"id": i}} for i in ids]

    merged = _interleave(entries("p1", "p2", "u2", "p3"), entries("u1", "u2", "u3"), share=0.5)
    assert [e["item"]["id"] for e in merged] == ["u1", "p1", "u2", "p2", "u3", "p3"]
