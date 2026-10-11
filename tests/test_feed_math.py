"""Per-image feed math (server/services/feed.py) and feed-session bookkeeping,
tested directly rather than through the HTTP feed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.services import feed

DAY = 86400.0
NOW = 1_000_000_000.0


def test_pool_terms_without_ratings():
    assert feed._pool_terms([], 0.01, NOW) == (0.0, 0.0, 0.0, 0)


def test_pool_terms_same_age_is_plain_mean_and_variance():
    mean, var, weight, count = feed._pool_terms([(0.0, NOW), (4.0, NOW)], 0.01, NOW)
    assert (mean, var, weight, count) == pytest.approx((2.0, 4.0, 2.0, 2))


def test_pool_terms_weight_old_votes_less():
    half_life = 0.6931471805599453 / 0.01
    mean, _var, weight, count = feed._pool_terms(
        [(0.0, NOW - half_life * DAY), (5.0, NOW)], 0.01, NOW
    )
    assert weight == pytest.approx(1.5)
    assert mean == pytest.approx(5.0 / 1.5), "fresh vote counts double the half-life-old one"
    assert count == 2, "count is raw, not weighted"


def test_settled_controversy_leaves_the_contested_pool():
    """Old split votes + fresh agreement -> lower contested score than a live split."""
    live = [(0.0, NOW), (5.0, NOW), (0.0, NOW), (5.0, NOW)]
    settled = [(0.0, NOW - 400 * DAY), (5.0, NOW - 400 * DAY), (5.0, NOW), (5.0, NOW)]

    def contested(ratings):
        _mean, var, weight, _n = feed._pool_terms(ratings, 0.01, NOW)
        return feed._contested_score(var, weight)

    assert contested(settled) < 0.5 * contested(live)


def test_blend_exploration_favours_unrated():
    mix = (0.0, 0.0, 1.0)
    assert feed._blended_score(0, 0, 0.0, mix) > feed._blended_score(5, 0, 10.0, mix)


def test_blend_mix_is_normalised():
    a = feed._blended_score(4.0, 1.0, 5.0, (0.5, 0.3, 0.2))
    b = feed._blended_score(4.0, 1.0, 5.0, (5.0, 3.0, 2.0))
    assert a == pytest.approx(b)


@pytest.fixture
def clean_sessions():
    saved = feed._sessions.copy()
    feed._sessions.clear()
    yield
    feed._sessions.clear()
    feed._sessions.update(saved)


def test_session_replays_served_positions(clean_sessions):
    key = (None, "s")
    first, more = feed._session_page(key, 0, 2, lambda: ["a", "b", "c"])
    replay, _ = feed._session_page(key, 0, 2, lambda: ["c", "b", "a"])
    rest, more_after = feed._session_page(key, 2, 2, lambda: ["c", "b", "a"])
    assert (first, more) == (["a", "b"], True)
    assert replay == ["a", "b"]
    assert (rest, more_after) == (["c"], False)


def test_sessions_expire_after_ttl(monkeypatch, clean_sessions):
    clock = {"t": NOW}
    monkeypatch.setattr(feed, "time", SimpleNamespace(time=lambda: clock["t"]))
    feed._session_page((None, "old"), 0, 1, lambda: ["a", "b"])

    clock["t"] += feed.SESSION_TTL_SECONDS + 1
    page, _ = feed._session_page((None, "new"), 0, 1, lambda: ["b", "a"])

    assert (None, "old") not in feed._sessions
    assert page == ["b"]


def test_session_count_is_bounded(monkeypatch, clean_sessions):
    monkeypatch.setattr(feed, "SESSION_MAX", 3)
    for i in range(5):
        feed._session_page((None, f"s{i}"), 0, 1, lambda: ["a"])
    assert list(feed._sessions) == [(None, "s2"), (None, "s3"), (None, "s4")]
