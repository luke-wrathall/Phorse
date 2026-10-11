"""Anchor set plumbing (horse/anchors.py): the operator-curated folder, the
embedding cache, and the guard against an empty anchor set. The real
datasets (CIFAR-10, PACS) are not exercised here -- see horse.check_anchors
for validating those against real CLIP."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from horse.anchors import (
    HORSE_TARGET,
    NOT_HORSE_TARGET,
    AnchorSet,
    AnchorSource,
    CuratedFolderAnchorSource,
    build_anchor_set,
    load_or_build_anchor_set,
)
from horse.embed import EMBEDDING_DIM, FakeEmbedder


def _save(path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_curated_folder_targets(tmp_path):
    root = tmp_path / "curated"
    _save(root / "horse" / "a.png", (0, 255, 0))
    _save(root / "not_horse" / "b.JPG", (255, 0, 0))
    _save(root / "3.5" / "c.png", (0, 0, 255))
    _save(root / "misc" / "d.png", (9, 9, 9))  # not a target name: ignored
    (root / "horse" / "notes.txt").write_text("not an image")

    examples = {example_id: target for example_id, _img, target in CuratedFolderAnchorSource(root)}

    assert examples == {
        "curated-3.5-c.png": 3.5,
        "curated-horse-a.png": HORSE_TARGET,
        "curated-not_horse-b.JPG": NOT_HORSE_TARGET,
    }


def test_missing_curated_folder_yields_nothing(tmp_path):
    assert list(CuratedFolderAnchorSource(tmp_path / "nope")) == []


def test_build_rejects_an_empty_anchor_set(tmp_path):
    with pytest.raises(ValueError, match="no anchor examples"):
        build_anchor_set(sources=[CuratedFolderAnchorSource(tmp_path / "nope")], embedder=FakeEmbedder())


class _ExplodingSource(AnchorSource):
    name = "exploding"

    def __iter__(self):
        raise AssertionError("cache should have been used instead of rebuilding")


def test_cache_is_reused_until_forced(tmp_path):
    root = tmp_path / "curated"
    _save(root / "horse" / "a.png", (0, 255, 0))
    cache = tmp_path / "cache.npz"
    source = CuratedFolderAnchorSource(root)

    built = load_or_build_anchor_set(cache, sources=[source], embedder=FakeEmbedder())
    cached = load_or_build_anchor_set(cache, sources=[_ExplodingSource()], embedder=FakeEmbedder())

    assert cached.ids == built.ids == ["curated-horse-a.png"]
    assert cached.embeddings.shape == (1, EMBEDDING_DIM)
    np.testing.assert_array_equal(cached.embeddings, built.embeddings)
    np.testing.assert_array_equal(cached.targets, [HORSE_TARGET])

    with pytest.raises(AssertionError, match="cache should have been used"):
        load_or_build_anchor_set(
            cache, sources=[_ExplodingSource()], embedder=FakeEmbedder(), force_rebuild=True
        )


def test_anchor_set_round_trips(tmp_path):
    original = AnchorSet(
        ids=["a", "b"],
        embeddings=np.eye(2, EMBEDDING_DIM, dtype=np.float32),
        targets=np.array([5.0, 0.0], dtype=np.float32),
    )
    original.save(tmp_path / "sub" / "set.npz")
    loaded = AnchorSet.load(tmp_path / "sub" / "set.npz")
    assert loaded.ids == ["a", "b"]
    np.testing.assert_array_equal(loaded.embeddings, original.embeddings)
    np.testing.assert_array_equal(loaded.targets, original.targets)
