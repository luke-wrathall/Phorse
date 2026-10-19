"""The head checkpoint, the fake embedder, the gate, and the API's model
singleton (server/services/model.py) -- in particular, that a half-written
checkpoint from the scheduler process never breaks scoring."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

from horse.embed import EMBEDDING_DIM, FakeEmbedder
from horse.gate import passes_gate
from horse.model import HorseHead


def test_head_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    head = HorseHead(hidden_dim=16)
    head.save(tmp_path / "h.pt")
    loaded = HorseHead.load(tmp_path / "h.pt")

    x = torch.randn(4, EMBEDDING_DIM)
    assert loaded.hidden_dim == 16
    assert not loaded.training, "loaded heads are in eval mode"
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), head(x))


def test_fake_embedder_is_deterministic_unit_norm_and_discriminating():
    embedder = FakeEmbedder()
    red = Image.new("RGB", (8, 8), (255, 0, 0))
    green = Image.new("RGB", (8, 8), (0, 255, 0))

    a, b, c = embedder.embed_image(red), embedder.embed_image(red.copy()), embedder.embed_image(green)

    assert a.shape == (EMBEDDING_DIM,) and a.dtype == np.float32
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(a, b)
    assert abs(float(a @ c)) < 0.3, "different images -> near-orthogonal embeddings"


@pytest.mark.parametrize("score, passed", [(2.5, True), (2.4999, False), (5.0, True), (-1.0, False)])
def test_gate_is_inclusive_at_threshold(score, passed):
    assert passes_gate(score, 2.5) is passed


# --- server model singleton -------------------------------------------------


def _probe() -> np.ndarray:
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _bump_mtime(path) -> None:
    mtime = os.path.getmtime(path) + 100
    os.utime(path, (mtime, mtime))


def test_no_checkpoint_means_untrained_head(app):
    from server.services import model

    with app.app_context():
        assert model.head_loaded() is False
        assert isinstance(model.score_embedding(_probe()), float)


def test_half_written_checkpoint_keeps_the_previous_head(app):
    from server.services import model

    head_path = app.config["HEAD_PATH"]
    torch.manual_seed(0)
    HorseHead().save(head_path)
    with app.app_context():
        before = model.score_embedding(_probe())

    with open(head_path, "wb") as f:
        f.write(b"\x80\x02truncated")
    _bump_mtime(head_path)

    with app.app_context():
        assert model.score_embedding(_probe()) == pytest.approx(before)
        assert model.head_loaded() is True

    # once a good checkpoint lands, it is picked up.
    torch.manual_seed(1)
    HorseHead().save(head_path)
    _bump_mtime(head_path)
    with app.app_context():
        assert model.score_embedding(_probe()) != pytest.approx(before)


def test_reload_head_forces_a_reload(app):
    from server.services import model

    with app.app_context():
        assert model.head_loaded() is False
        HorseHead().save(app.config["HEAD_PATH"])
        assert model.reload_head() is True


def test_save_replaces_the_checkpoint_atomically(tmp_path, monkeypatch):
    """A save that dies mid-write leaves the old checkpoint, not a corrupt one."""
    path = tmp_path / "head.pt"
    torch.manual_seed(0)
    old = HorseHead(hidden_dim=16)
    old.save(path)
    assert not (tmp_path / "head.pt.tmp").exists()

    def dies_mid_write(obj, f):
        with open(f, "wb") as out:
            out.write(b"half a checkp")
        raise KeyboardInterrupt  # e.g. the scheduler killed during a deploy

    monkeypatch.setattr("horse.model.torch.save", dies_mid_write)
    with pytest.raises(KeyboardInterrupt):
        HorseHead(hidden_dim=16).save(path)
    monkeypatch.undo()

    x = torch.randn(2, EMBEDDING_DIM)
    with torch.no_grad():
        torch.testing.assert_close(HorseHead.load(path)(x), old(x))


def test_linear_head_round_trips(tmp_path):
    """hidden_dim 0 is a plain linear head (the smallest drift head)."""
    torch.manual_seed(0)
    head = HorseHead(hidden_dim=0)
    assert sum(p.numel() for p in head.parameters()) == EMBEDDING_DIM + 1
    head.save(tmp_path / "head.pt")
    x = torch.randn(3, EMBEDDING_DIM)
    with torch.no_grad():
        torch.testing.assert_close(HorseHead.load(tmp_path / "head.pt")(x), head(x))
