"""Uploads are scored at several rotations and stored the best way up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import register, solid_png, upload_png


@pytest.fixture
def tall_scores(monkeypatch):
    """Stub the model so score depends only on orientation: a wide image
    scores `wide`, a tall one `tall`. Returns a setter for the two scores."""
    from server.services import model

    scores = {"wide": 1.0, "tall": 1.0}

    def embed_image(image):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0 if image.height > image.width else 0.0
        return embedding

    monkeypatch.setattr(model, "embed_image", embed_image)
    monkeypatch.setattr(
        model, "score_embedding", lambda e: scores["tall"] if e[0] else scores["wide"]
    )
    return scores


def _stored_size(app, image_id: str) -> tuple[int, int]:
    upload_dir = Path(app.config["UPLOAD_DIR"])
    (path,) = upload_dir.glob(f"{image_id}.*")
    with Image.open(path) as image:
        return image.size


def test_sideways_upload_is_rotated_and_saved_upright(app, client, tall_scores):
    tall_scores.update(wide=1.0, tall=4.0)
    token = register(client, "drawer")

    result = upload_png(client, token, solid_png((10, 10, 10), size=(40, 20)))

    assert result["passed"] is True
    assert result["score"] == pytest.approx(4.0)
    assert _stored_size(app, result["id"]) == (20, 40), "stored file should be the rotated one"


@pytest.mark.parametrize(
    "wide, tall, why",
    [
        (3.0, 4.5, "upright already passes -- rotation not needed"),
        (0.5, 2.0, "no rotation passes -- store as sent"),
        (2.2, 2.6, "rotation passes but beats upright by less than the 0.5 margin"),
    ],
)
def test_rotation_only_used_when_it_rescues(app, client, tall_scores, wide, tall, why):
    tall_scores.update(wide=wide, tall=tall)
    token = register(client, "photographer")

    result = upload_png(client, token, solid_png((10, 10, 10), size=(40, 20)))

    assert result["score"] == pytest.approx(wide), why
    assert _stored_size(app, result["id"]) == (40, 20), why
