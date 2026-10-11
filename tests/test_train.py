"""Unit tests for the drift-head loss in horse/train.py.

These pin the properties CLAUDE.md "Training / drift mechanism" relies on,
directly on `train_head` rather than through the whole retrain loop:
recency weights, lambda_anchor as a constant ratio (not diluted by vote
volume), the single *pooled* floored denominator (not per image), and the
proximal pull.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from horse.embed import EMBEDDING_DIM
from horse.model import HorseHead
from horse.train import recency_weights, train_from_store, train_head

DAY = 86400.0


def _unit_vectors(n: int, seed: int) -> np.ndarray:
    vecs = np.random.default_rng(seed).standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def anchors():
    embeddings = _unit_vectors(6, seed=0)
    targets = np.array([5, 5, 5, 0, 0, 0], dtype=np.float32)
    return embeddings, targets


def _seeded_head(seed: int = 0) -> HorseHead:
    torch.manual_seed(seed)
    return HorseHead()


def _predict(head: HorseHead, embeddings: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(embeddings)).numpy()


# --- recency weights -------------------------------------------------------


def test_recency_weight_is_one_now_and_halves_each_half_life():
    lambda_decay = 0.01
    half_life_days = math.log(2) / lambda_decay
    now = 1_000_000_000.0
    timestamps = np.array([now, now - half_life_days * DAY, now - 2 * half_life_days * DAY])

    weights = recency_weights(timestamps, lambda_decay, now=now)

    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.25], rtol=1e-5)


def test_zero_decay_weights_every_rating_equally():
    now = 1_000_000_000.0
    weights = recency_weights(np.array([now, now - 1000 * DAY]), 0.0, now=now)
    np.testing.assert_allclose(weights, [1.0, 1.0])


# --- the loss itself -------------------------------------------------------


def _initial_community_loss(anchors, rating_x, rating_y, rating_w, floor: float) -> float:
    """community_loss as train_head reports it for epoch 0 (before any step)."""
    _, summary = train_head(
        *anchors, rating_x, rating_y, rating_w,
        community_weight_floor=floor, epochs=1, head=_seeded_head(),
    )
    return summary["community_loss"]


def test_community_term_is_pooled_weighted_mean_above_floor(anchors):
    """Two images with 2 effective votes each: the denominator is the pooled
    total (4), not a per-image floor applied to each image's 2."""
    rating_x = np.repeat(_unit_vectors(2, seed=1), 2, axis=0)
    rating_y = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    rating_w = np.ones(4, dtype=np.float32)
    sq_err = (_predict(_seeded_head(), rating_x) - rating_y) ** 2

    reported = _initial_community_loss(anchors, rating_x, rating_y, rating_w, floor=3.0)

    pooled = float(np.sum(rating_w * sq_err) / max(rating_w.sum(), 3.0))
    per_image = float(np.mean([sq_err[:2].sum() / 3.0, sq_err[2:].sum() / 3.0]))
    assert reported == pytest.approx(pooled, rel=1e-5)
    assert reported != pytest.approx(per_image, rel=1e-3), "denominator must not be per image"


def test_floor_bounds_the_denominator_when_total_weight_is_tiny(anchors):
    """One heavily decayed vote: sum(w) = 0.01, so the floor (3.0) is the
    denominator -- the community term is scaled down, not blown up."""
    rating_x = _unit_vectors(1, seed=2)
    rating_y = np.array([0.0], dtype=np.float32)
    rating_w = np.array([0.01], dtype=np.float32)
    sq_err = float(((_predict(_seeded_head(), rating_x) - rating_y) ** 2)[0])

    reported = _initial_community_loss(anchors, rating_x, rating_y, rating_w, floor=3.0)

    assert reported == pytest.approx(0.01 * sq_err / 3.0, rel=1e-5)


def test_where_votes_land_does_not_change_their_total_pull(anchors):
    """Spreading votes across images vs. stacking them on one contributes the
    same total weight to the shared denominator (CLAUDE.md "Deliberately not
    per-image"): with identical errors, the community loss is identical."""
    head = _seeded_head()
    one_image = np.repeat(_unit_vectors(1, seed=3), 6, axis=0)
    many_images = _unit_vectors(6, seed=4)
    # targets = the head's own predictions + 1, so every vote has sq_err == 1.
    stacked_y = _predict(head, one_image) + 1.0
    spread_y = _predict(head, many_images) + 1.0
    w = np.ones(6, dtype=np.float32)

    stacked = _initial_community_loss(anchors, one_image, stacked_y, w, floor=3.0)
    spread = _initial_community_loss(anchors, many_images, spread_y, w, floor=3.0)

    assert stacked == pytest.approx(spread, rel=1e-5) == pytest.approx(1.0, rel=1e-5)


def test_lambda_anchor_ratio_is_not_diluted_by_vote_volume(anchors):
    """Ten times the same votes (well above the floor) leaves the loss -- and
    so the trained head -- unchanged. The old summed form let volume swamp
    the anchor."""
    rating_x = _unit_vectors(5, seed=5)
    rating_y = np.zeros(5, dtype=np.float32)
    rating_w = np.ones(5, dtype=np.float32)

    few, _ = train_head(*anchors, rating_x, rating_y, rating_w, epochs=50, head=_seeded_head())
    many, _ = train_head(
        *anchors,
        np.tile(rating_x, (10, 1)), np.tile(rating_y, 10), np.tile(rating_w, 10),
        epochs=50, head=_seeded_head(),
    )

    probe = np.concatenate([anchors[0], rating_x])
    np.testing.assert_allclose(_predict(few, probe), _predict(many, probe), atol=1e-3)


def test_higher_lambda_anchor_holds_a_rated_anchor_closer_to_its_target(anchors):
    """Community votes say horse-anchor 0 is a 0; a bigger lambda_anchor keeps
    it nearer its anchor target of 5."""
    rating_x = np.repeat(anchors[0][:1], 5, axis=0)
    rating_y = np.zeros(5, dtype=np.float32)
    rating_w = np.ones(5, dtype=np.float32)

    def score_after(lambda_anchor: float) -> float:
        head, _ = train_head(
            *anchors, rating_x, rating_y, rating_w,
            lambda_anchor=lambda_anchor, epochs=300, head=_seeded_head(),
        )
        return float(_predict(head, anchors[0][:1])[0])

    assert score_after(5.0) > score_after(0.25) + 0.3


def test_trains_on_anchors_alone_when_there_are_no_ratings(anchors):
    empty_x = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)

    head, summary = train_head(*anchors, empty_x, empty, empty, epochs=300, head=_seeded_head())

    assert summary["community_loss"] == 0.0
    assert summary["final_loss"] == pytest.approx(0.25 * summary["anchor_loss"], rel=1e-5)
    scores = _predict(head, anchors[0])
    assert scores[:3].min() > scores[3:].max(), f"horse anchors should outscore non-horse: {scores}"


# --- proximal pull ---------------------------------------------------------


def _param_distance(a: HorseHead, b: HorseHead) -> float:
    with torch.no_grad():
        return float(sum(torch.sum((p - q) ** 2) for p, q in zip(a.parameters(), b.parameters())))


def test_prox_weight_keeps_a_warm_start_closer_to_the_previous_head(anchors):
    rating_x = _unit_vectors(3, seed=6)
    rating_y = np.zeros(3, dtype=np.float32)
    rating_w = np.ones(3, dtype=np.float32)
    start = _seeded_head()

    def trained(prox_weight: float) -> HorseHead:
        head, _ = train_head(
            *anchors, rating_x, rating_y, rating_w,
            epochs=100, head=copy.deepcopy(start), prox_weight=prox_weight,
        )
        return head

    assert _param_distance(trained(10.0), start) < 0.5 * _param_distance(trained(0.0), start)


def test_prox_weight_is_ignored_without_a_warm_start(anchors):
    empty_x = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)

    torch.manual_seed(0)
    with_prox, s1 = train_head(*anchors, empty_x, empty, empty, epochs=20, prox_weight=5.0)
    torch.manual_seed(0)
    without, s2 = train_head(*anchors, empty_x, empty, empty, epochs=20, prox_weight=0.0)

    assert s1["final_loss"] == pytest.approx(s2["final_loss"])
    assert _param_distance(with_prox, without) == pytest.approx(0.0)


# --- train_from_store ------------------------------------------------------


@pytest.fixture
def anchor_cache(tmp_path) -> Path:
    from conftest import _build_anchor_cache

    path = tmp_path / "anchors.npz"
    _build_anchor_cache(path)
    return path


def test_train_from_store_writes_head_atomically_and_reports_counts(tmp_path, anchor_cache):
    from horse.storage import add_rating, connect

    ratings_db = tmp_path / "ratings.db"
    conn = connect(ratings_db)
    embedding = _unit_vectors(1, seed=7)[0]
    add_rating(conn, "img", "alice", embedding, 1.0)
    add_rating(conn, "img", "bob", embedding, 2.0)
    conn.close()
    head_out = tmp_path / "head.pt"

    summary = train_from_store(str(ratings_db), str(anchor_cache), str(head_out), epochs=5)

    assert summary["n_ratings"] == 2
    assert summary["n_anchors"] == 6
    assert head_out.exists()
    assert not Path(f"{head_out}.tmp").exists(), "temp checkpoint should be renamed into place"
    HorseHead.load(head_out)  # loadable


def test_train_from_store_with_empty_store_trains_on_anchors(tmp_path, anchor_cache):
    head_out = tmp_path / "head.pt"
    summary = train_from_store(
        str(tmp_path / "ratings.db"), str(anchor_cache), str(head_out), epochs=5
    )
    assert summary["n_ratings"] == 0
    assert summary["community_loss"] == 0.0


def test_train_from_store_warm_starts_from_head_in(tmp_path, anchor_cache):
    head_in = tmp_path / "prev.pt"
    prev = _seeded_head(seed=42)
    prev.save(head_in)
    head_out = tmp_path / "head.pt"

    train_from_store(
        str(tmp_path / "ratings.db"), str(anchor_cache), str(head_out),
        head_in=str(head_in), epochs=1, lr=1e-6,
    )

    assert _param_distance(HorseHead.load(head_out), prev) < 1e-6


def test_train_from_store_ignores_missing_head_in(tmp_path, anchor_cache):
    head_out = tmp_path / "head.pt"
    train_from_store(
        str(tmp_path / "ratings.db"), str(anchor_cache), str(head_out),
        head_in=str(tmp_path / "nope.pt"), epochs=1,
    )
    assert head_out.exists()
