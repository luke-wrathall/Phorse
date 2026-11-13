"""horse.tune: the offline tuning report runs end to end on a real app.db
and leaves it untouched."""

from __future__ import annotations

import re
import sys

from horse import tune

from conftest import HORSE_COLORS, NOT_HORSE_COLORS, auth, register, solid_png, upload_png


def test_tune_reports_on_a_copy_without_touching_it(app, client, tmp_path, monkeypatch, capsys, fixed_score):
    alice, bob = register(client, "alice"), register(client, "bob")
    horses = [upload_png(client, alice, solid_png(c)) for c in HORSE_COLORS]
    fixed_score["value"] = 0.0  # one the gate turned away
    upload_png(client, alice, solid_png(NOT_HORSE_COLORS[0]))
    for token, value in ((alice, 5), (bob, 0)):
        for image in horses[:2]:
            resp = client.post(f"/api/images/{image['id']}/ratings", json={"rating": value},
                               headers=auth(token))
            assert resp.status_code in (200, 201), resp.get_json()

    db_path = tmp_path / "app.db"
    before = db_path.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "tune", "--app-db", str(db_path), "--anchor-cache", app.config["ANCHOR_CACHE"],
        "--lambda-anchor", "0.25,1", "--prox", "1", "--hidden", "64,0", "--epochs", "3",
        "--base-epochs", "5", "--cycles", "0,2", "--folds", "2", "--augment-weight", "0.25,0",
        "--workers", "1",
    ])
    tune.main()
    out = capsys.readouterr().out

    assert "4 votes on 2 images from 2 people" in out
    assert "votes by value:  0: 2   5: 2" in out
    assert "== knob sweep" in out
    # the two rated images' three other rotations each, applied to every vote on them
    assert "12 augmentation rows (9 stored rotations)" in out
    # baseline + one other lambda_anchor + one other head size + augmentations off,
    # two checkpoints each
    sweep_rows = [line for line in out.splitlines()
                  if re.match(r"\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+ \|", line)]
    assert len(sweep_rows) == 8
    assert db_path.read_bytes() == before
