"""End-to-end proof that the Stage 2 drift loop is closed.

register -> upload -> feed -> rate (via the real HTTP API) -> run_retrain
(ingest + train + head reload + re-score) -> the classifier has measurably
moved, and the anchor set kept it from collapsing to the raw rating. Then:
time passes with no new votes -> the stale ratings fade and the anchor
pulls the score back.
"""

from __future__ import annotations

import time

from conftest import HORSE_COLORS, auth, image_row, register, solid_png, upload_png


def _fresh_app(tmp_path, **config_overrides):
    """Build a standalone app instance (own db/head/anchor-cache) -- the
    same way conftest's `app` fixture does, but callable more than once so
    a test can run two independent platform scenarios (e.g. with vs.
    without a second busy image) without one contaminating the other's
    trained head or model singleton state.
    """
    from conftest import _build_anchor_cache, _reset_model_singletons
    from server import create_app

    tmp_path.mkdir(parents=True, exist_ok=True)
    anchor_cache = tmp_path / "anchors.npz"
    _build_anchor_cache(anchor_cache)
    config = {
        "TESTING": True,
        "SECRET_KEY": "test-secret-" + "0" * 32,
        "DB_PATH": str(tmp_path / "app.db"),
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "HEAD_PATH": str(tmp_path / "head.pt"),
        "TRAINING_DB": str(tmp_path / "ratings.db"),
        "ANCHOR_CACHE": str(anchor_cache),
        "GATE_THRESHOLD": 2.5,
        "LAMBDA_ANCHOR": 1.0,
        "LAMBDA_DECAY": 0.01,
        "COMMUNITY_WEIGHT_FLOOR": 3.0,
        "RETRAIN_INTERVAL_SECONDS": 0,
        "REQUIRE_INVITE": False,
    }
    config.update(config_overrides)
    application = create_app(config)
    _reset_model_singletons()
    return application


def test_stale_image_fades_more_with_other_platform_activity(tmp_path):
    """community_weight_floor floors a single, platform-wide denominator,
    not a per-image one (see CLAUDE.md "Deliberately not per-image") -- a
    stale image's fade-back is *platform-relative*: it depends on how much
    OTHER, fresher weighted activity exists on the platform, not on a fixed
    per-image vote-count threshold.

    Proof: run the same "rate an image down, then let it go stale" story
    twice -- once alone, once alongside a second image that keeps getting
    fresh votes throughout. The lone image's stale weight only has to clear
    the bare floor (3.0) to keep some pull; the same stale weight sitting
    next to a heavily-and-freshly-rated second image is diluted by that
    image's large contribution to the SHARED denominator too, so it should
    recover a bigger fraction of its original drop.
    """
    from server.db import get_db
    from server.services.retrain import run_retrain

    def rate_all(client, image, tokens, rating=0.0):
        for token in tokens:
            resp = client.post(
                f"/api/images/{image['id']}/ratings", json={"rating": rating}, headers=auth(token)
            )
            assert resp.status_code == 200, resp.get_json()

    def run_scenario(app_dir, with_busy_second_image: bool) -> tuple[float, float, float]:
        """Returns (s0, s1, s2) for image A: pre-rating, post-rating, post-stale."""
        # This tests the loss's fixed point (pooled-floor weighting), so each
        # cycle should reach it: prox off. With prox on, one cycle is only a
        # partial step toward the same fixed point, and the busier scenario's
        # step budget is shared with image B, which confounds the comparison.
        app = _fresh_app(app_dir, RETRAIN_PROX_WEIGHT=0.0)
        client = app.test_client()
        with app.app_context():
            run_retrain()

        uploader = register(client, "uploader")
        raters_a = [register(client, f"rater_a{i}") for i in range(5)]
        image_a = upload_png(client, uploader, solid_png(HORSE_COLORS[0]), name="a.png")
        s0 = image_a["score"]
        rate_all(client, image_a, raters_a)

        if with_busy_second_image:
            raters_b = [register(client, f"rater_b{i}") for i in range(20)]
            image_b = upload_png(client, uploader, solid_png(HORSE_COLORS[1]), name="b.png")
            rate_all(client, image_b, raters_b)

        with app.app_context():
            run_retrain()
        s1 = image_row(app, image_a["id"])["score"]

        # A's votes go stale.
        with app.app_context():
            db = get_db()
            db.execute(
                "UPDATE ratings SET created_at = ? WHERE image_id = ?",
                (time.time() - 400 * 86400, image_a["id"]),
            )
            db.commit()

        if with_busy_second_image:
            rate_all(client, image_b, raters_b)  # re-rate "now" -- fresh weight

        with app.app_context():
            run_retrain()
        s2 = image_row(app, image_a["id"])["score"]

        return s0, s1, s2

    s0_lone, s1_lone, s2_lone = run_scenario(tmp_path / "lone", with_busy_second_image=False)
    s0_busy, s1_busy, s2_busy = run_scenario(tmp_path / "busy", with_busy_second_image=True)

    assert s1_lone < s0_lone - 0.3, "sanity: A's rating should have dragged its score down (lone)"
    assert s1_busy < s0_busy - 0.3, "sanity: A's rating should have dragged its score down (busy)"

    # Fraction of A's original drop recovered once its votes are stale --
    # normalized per-scenario so this doesn't depend on the two scenarios'
    # (independently random-initialized) heads landing on identical scores.
    recovered_lone = (s2_lone - s1_lone) / (s0_lone - s1_lone)
    recovered_busy = (s2_busy - s1_busy) / (s0_busy - s1_busy)

    assert recovered_busy > recovered_lone + 0.1, (
        "a stale image sitting next to a heavily, freshly-rated second image should recover "
        f"more of its drop than the same stale image alone: lone={recovered_lone:.2f} "
        f"busy={recovered_busy:.2f}"
    )


def test_retrain_cycle_nudges_rather_than_refits(tmp_path):
    """RETRAIN_PROX_WEIGHT makes each cycle a partial step: one cycle moves a
    freshly rated image less than an unregularized refit would, and repeated
    cycles keep moving it further toward the rating.
    """
    from server.services.retrain import run_retrain

    def scores_over_cycles(app_dir, prox_weight: float, cycles: int) -> list[float]:
        app = _fresh_app(app_dir, RETRAIN_PROX_WEIGHT=prox_weight)
        client = app.test_client()
        with app.app_context():
            run_retrain()
        uploader = register(client, "uploader")
        image = upload_png(client, uploader, solid_png(HORSE_COLORS[0]))
        scores = [image["score"]]
        for i in range(5):
            token = register(client, f"rater{i}")
            resp = client.post(
                f"/api/images/{image['id']}/ratings", json={"rating": 0.0}, headers=auth(token)
            )
            assert resp.status_code == 200, resp.get_json()
        for _ in range(cycles):
            with app.app_context():
                run_retrain()
            scores.append(image_row(app, image["id"])["score"])
        return scores

    refit = scores_over_cycles(tmp_path / "refit", prox_weight=0.0, cycles=1)
    nudged = scores_over_cycles(tmp_path / "nudged", prox_weight=1.0, cycles=4)

    assert refit[0] - refit[1] > 0.3, f"sanity: an unregularized refit should move the score: {refit}"
    assert nudged[1] < nudged[0], f"one prox cycle should still move the score: {nudged}"
    assert nudged[1] > refit[1] + 0.5, (
        f"one prox cycle should move less than a full refit: nudged={nudged} refit={refit}"
    )
    assert nudged[-1] < nudged[1] - 0.05, f"later cycles should keep drifting: {nudged}"


def test_core_loop(app, client):
    from server.db import get_db
    from server.services.retrain import run_retrain

    # 0. train the head on the anchor set alone so scores mean something.
    with app.app_context():
        run_retrain()

    uploader = register(client, "uploader")
    raters = [register(client, f"rater{i}") for i in range(5)]

    # 1. upload an image byte-identical to a "horse" anchor -> scores high, passes.
    horse_png = solid_png(HORSE_COLORS[0])
    result = upload_png(client, uploader, horse_png)
    image_id = result["id"]
    s0 = result["score"]
    assert result["passed"] is True
    assert s0 > 3.0, f"anchor-matching image should read as a horse, got {s0}"

    # 2. it's in the feed.
    feed = client.get("/api/feed").get_json()
    assert any(item["id"] == image_id for item in feed["items"])

    # 3. everyone rates it down hard.
    for token in raters:
        resp = client.post(
            f"/api/images/{image_id}/ratings", json={"rating": 0.5}, headers=auth(token)
        )
        assert resp.status_code == 200, resp.get_json()

    # 4. retrain -> the loop carries those ratings into the model.
    with app.app_context():
        summary = run_retrain()
    assert summary["ratings_synced"] == 5
    assert summary["images_rescored"] >= 1

    row = image_row(app, image_id)
    s1 = row["score"]
    assert s1 < s0 - 0.3, f"ratings should have dragged the score down: {s0} -> {s1}"
    assert s1 > 0.5, f"anchor should keep it above the raw rating: {s1}"
    assert row["passed"] == 1, "admission is point-in-time, not re-evaluated"

    # 5. time passes, nobody re-rates -> stale ratings fade, anchor pulls back.
    with app.app_context():
        db = get_db()
        db.execute("UPDATE ratings SET created_at = ?", (time.time() - 400 * 86400,))
        db.commit()
    with app.app_context():
        run_retrain()

    s2 = image_row(app, image_id)["score"]
    assert s2 > s1 + 0.5, f"decayed ratings should let the score drift back up: {s1} -> {s2}"
    assert s2 <= s0 + 0.5


def test_feed_still_serves_drifted_image(app, client):
    """An image whose score fell below the gate threshold stays in the feed."""
    from server.services.retrain import run_retrain

    with app.app_context():
        run_retrain()
    uploader = register(client, "owner")
    raters = [register(client, f"rater{i}") for i in range(5)]

    image_id = upload_png(client, uploader, solid_png(HORSE_COLORS[0]))["id"]
    for token in raters:
        client.post(f"/api/images/{image_id}/ratings", json={"rating": 0.0}, headers=auth(token))
    with app.app_context():
        run_retrain()

    assert image_row(app, image_id)["score"] < app.config["GATE_THRESHOLD"]
    feed = client.get("/api/feed?limit=50").get_json()
    assert any(item["id"] == image_id for item in feed["items"])


def test_ingest_idempotent(app, client):
    from horse.ingest import ingest
    from horse.storage import all_ratings, connect

    alice = register(client, "alice")
    bob = register(client, "bob")
    image_id = upload_png(client, alice, solid_png(HORSE_COLORS[0]))["id"]
    client.post(f"/api/images/{image_id}/ratings", json={"rating": 3.0}, headers=auth(alice))
    client.post(f"/api/images/{image_id}/ratings", json={"rating": 4.0}, headers=auth(bob))

    with app.app_context():
        first = ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])
        second = ingest(app.config["DB_PATH"], app.config["TRAINING_DB"])

    assert first == second == {"ratings_synced": 2, "images": 1}
    conn = connect(app.config["TRAINING_DB"])
    try:
        assert len(all_ratings(conn)) == 2
    finally:
        conn.close()


def test_api_hot_reloads_head_written_out_of_process(app, client):
    """The API process picks up a head.pt written by the scheduler process
    (a separate process) via the mtime check -- no reload_head() call, no restart.
    """
    import os

    import numpy as np

    from horse.train import train_from_store
    from server.services import model
    from server.services.retrain import run_retrain

    with app.app_context():
        run_retrain()

    probe = np.ones(512, dtype=np.float32)
    probe /= np.linalg.norm(probe)
    with app.app_context():
        before = model.score_embedding(probe)

    # simulate the scheduler process retraining with a very different signal:
    # write head.pt directly, WITHOUT touching this process's model singleton.
    head_path = app.config["HEAD_PATH"]
    with app.app_context():
        train_from_store(
            app.config["TRAINING_DB"],
            app.config["ANCHOR_CACHE"],
            head_path,
            lambda_anchor=50.0,  # yank the whole function toward the anchor targets
            epochs=300,
        )
    os.utime(head_path, (os.path.getmtime(head_path) + 100,) * 2)

    with app.app_context():
        after = model.score_embedding(probe)
    assert abs(after - before) > 1e-3, "score_embedding should have hot-reloaded the new head"


def test_feed_blend_orders_pools():
    from server.services.feed import _blended_score

    mix = (0.5, 0.3, 0.2)
    consensus = _blended_score(mean=4.5, var=0.1, weight=10.0, mix=mix)
    contested = _blended_score(mean=2.5, var=4.0, weight=10.0, mix=mix)
    fresh = _blended_score(mean=0.0, var=0.0, weight=0.0, mix=mix)
    boring = _blended_score(mean=1.0, var=0.1, weight=10.0, mix=mix)

    assert consensus > contested > fresh > boring
