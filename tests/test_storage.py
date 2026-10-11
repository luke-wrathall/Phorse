"""Training store (horse/storage.py), the app -> training bridge
(horse/ingest.py), the app-db column migration, and re-scoring."""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from conftest import HORSE_COLORS, auth, register, solid_png, upload_png
from horse.embed import EMBEDDING_DIM
from horse.storage import add_rating, all_ratings, connect


def _vec(seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# --- storage ---------------------------------------------------------------


def test_add_rating_rejects_wrong_embedding_shape(tmp_path):
    conn = connect(tmp_path / "r.db")
    with pytest.raises(ValueError, match="embedding shape"):
        add_rating(conn, "img", "u", np.zeros(10, np.float32), 3.0)
    conn.close()


def test_one_live_rating_per_user_per_image(tmp_path):
    conn = connect(tmp_path / "r.db")
    add_rating(conn, "img", "alice", _vec(), 1.0, timestamp=100.0)
    add_rating(conn, "img", "alice", _vec(), 4.0, timestamp=200.0)
    add_rating(conn, "img", "bob", _vec(), 2.0, timestamp=150.0)

    rows = sorted((r[2], r[3]) for r in all_ratings(conn))
    conn.close()

    assert rows == [(2.0, 150.0), (4.0, 200.0)], "re-rating replaces the vote and its timestamp"


def test_all_ratings_round_trips_the_embedding(tmp_path):
    conn = connect(tmp_path / "r.db")
    add_rating(conn, "img", "alice", _vec(3), 2.0)
    ((image_id, embedding, rating, _ts),) = all_ratings(conn)
    conn.close()
    assert image_id == "img" and rating == 2.0
    np.testing.assert_array_equal(embedding, _vec(3))


def _legacy_db(path, rows: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE ratings (image_id TEXT PRIMARY KEY, embedding BLOB, rating REAL, timestamp REAL)"
    )
    for i in range(rows):
        conn.execute("INSERT INTO ratings VALUES (?, ?, 1.0, 0.0)", (f"img{i}", b""))
    conn.commit()
    conn.close()


def test_empty_legacy_store_is_migrated(tmp_path):
    path = tmp_path / "r.db"
    _legacy_db(path, rows=0)
    conn = connect(path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ratings)")}
    conn.close()
    assert "user_id" in cols


def test_non_empty_legacy_store_refuses_to_open(tmp_path):
    path = tmp_path / "r.db"
    _legacy_db(path, rows=2)
    with pytest.raises(RuntimeError, match="old single-rating-per-image schema"):
        connect(path)


# --- ingest ----------------------------------------------------------------


def test_ingest_without_app_db_is_a_no_op(tmp_path):
    from horse.ingest import ingest

    assert ingest(tmp_path / "missing.db", tmp_path / "r.db") == {"ratings_synced": 0, "images": 0}


def test_ingest_reflects_app_rerating(app, client):
    from horse.ingest import ingest
    from server.db import get_db

    alice = register(client, "alice")
    image_id = upload_png(client, alice, solid_png(HORSE_COLORS[0]))["id"]
    client.post(f"/api/images/{image_id}/ratings", json={"rating": 1.0}, headers=auth(alice))
    with app.app_context():
        ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
        client.post(f"/api/images/{image_id}/ratings", json={"rating": 4.0}, headers=auth(alice))
        app_ts = get_db().execute("SELECT created_at FROM ratings").fetchone()[0]
        ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])

    conn = connect(app.config["TRAINING_DB"])
    ((_id, embedding, rating, ts),) = all_ratings(conn)
    conn.close()
    assert (rating, ts) == (4.0, app_ts)
    assert embedding.shape == (EMBEDDING_DIM,)


def _insert_image(app, image_id: str, embedding_blob: bytes) -> None:
    from server.db import get_db

    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO images (id, owner_id, filename, ext, caption, embedding, score, passed, created_at) "
            "VALUES (?, 1, 'x.png', '.png', NULL, ?, 3.0, 1, ?)",
            (image_id, embedding_blob, time.time()),
        )
        db.commit()


def test_ingest_skips_malformed_embeddings_and_does_not_count_them(app, client):
    from horse.ingest import ingest

    alice = register(client, "alice")
    good = upload_png(client, alice, solid_png(HORSE_COLORS[0]))["id"]
    _insert_image(app, "legacy", np.zeros(3, np.float32).tobytes())
    for image_id in (good, "legacy"):
        resp = client.post(f"/api/images/{image_id}/ratings", json={"rating": 2.0}, headers=auth(alice))
        assert resp.status_code == 200

    with app.app_context():
        summary = ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])

    assert summary == {"ratings_synced": 1, "images": 1}


# --- re-scoring + app db ---------------------------------------------------


def test_retrain_rescore_skips_malformed_embeddings(app, client):
    from server.services.retrain import run_retrain

    alice = register(client, "alice")
    upload_png(client, alice, solid_png(HORSE_COLORS[0]))
    _insert_image(app, "legacy", np.zeros(3, np.float32).tobytes())

    with app.app_context():
        summary = run_retrain()

    assert summary["images_rescored"] == 1


def test_app_db_gains_caption_column_on_old_schema(tmp_path):
    from server import create_app

    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE images (id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, filename TEXT, "
        "ext TEXT NOT NULL, embedding BLOB NOT NULL, score REAL NOT NULL, passed INTEGER NOT NULL, "
        "created_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    create_app({"TESTING": True, "DB_PATH": str(db_path)})
    create_app({"TESTING": True, "DB_PATH": str(db_path)})  # second start: column already there

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    conn.close()
    assert "caption" in cols
