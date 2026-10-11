"""Shared test fixtures.

Everything here runs with HORSE_FAKE_EMBEDDER on, so no CLIP / CIFAR
downloads: images map to deterministic embeddings (horse.embed.FakeEmbedder)
and the anchor cache is built from a handful of solid-colour images. The
test image the core-loop test uploads is byte-identical to one of the
"horse" anchors, so it starts life scoring high and we can watch community
ratings drag it down (and the anchor pull it back).
"""

from __future__ import annotations

import io
import os

os.environ["HORSE_FAKE_EMBEDDER"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

# Greenish = "horse" anchors, reddish = "not horse" anchors.
HORSE_COLORS = [(10, 200, 10), (20, 180, 40), (30, 160, 25)]
NOT_HORSE_COLORS = [(200, 10, 10), (180, 40, 20), (160, 25, 30)]


def solid_png(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _embed_png(embedder, png: bytes) -> np.ndarray:
    return embedder.embed_image(Image.open(io.BytesIO(png)).convert("RGB"))


def _build_anchor_cache(path) -> None:
    from horse.embed import get_embedder

    embedder = get_embedder()
    ids, embs, targets = [], [], []
    for i, color in enumerate(HORSE_COLORS):
        ids.append(f"anchor-horse-{i}")
        embs.append(_embed_png(embedder, solid_png(color)))
        targets.append(5.0)
    for i, color in enumerate(NOT_HORSE_COLORS):
        ids.append(f"anchor-not-{i}")
        embs.append(_embed_png(embedder, solid_png(color)))
        targets.append(0.0)
    np.savez(
        path,
        ids=np.array(ids),
        embeddings=np.stack(embs).astype(np.float32),
        targets=np.array(targets, dtype=np.float32),
    )


def _reset_model_singletons() -> None:
    from server.services import model

    model._state.update(
        {"embedder": None, "head": None, "head_loaded": False, "head_mtime": None}
    )


@pytest.fixture
def app(tmp_path):
    from server import create_app

    anchor_cache = tmp_path / "anchors.npz"
    _build_anchor_cache(anchor_cache)

    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-" + "0" * 32,
            "DB_PATH": str(tmp_path / "app.db"),
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "HEAD_PATH": str(tmp_path / "head.pt"),
            "TRAINING_DB": str(tmp_path / "ratings.db"),
            "ANCHOR_CACHE": str(anchor_cache),
            "GATE_THRESHOLD": 2.5,
            # anchor deliberately given real leverage for the test so the
            # "anchor stopped full collapse" assertion has margin.
            "LAMBDA_ANCHOR": 1.0,
            "LAMBDA_DECAY": 0.01,
            "COMMUNITY_WEIGHT_FLOOR": 3.0,
            "RETRAIN_INTERVAL_SECONDS": 0,
        }
    )
    _reset_model_singletons()
    yield application
    _reset_model_singletons()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, username: str) -> str:
    """Register a user, return their bearer token."""
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": "password123"}
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def upload_png(client, token: str, png: bytes, name: str = "img.png") -> dict:
    resp = client.post(
        "/api/images",
        data={"image": (io.BytesIO(png), name)},
        headers=auth(token),
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()


def image_row(app, image_id: str):
    from server.db import get_db

    with app.app_context():
        return get_db().execute(
            "SELECT score, passed FROM images WHERE id = ?", (image_id,)
        ).fetchone()
