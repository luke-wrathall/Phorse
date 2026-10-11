"""Feed ranking -- the three-pool blend (CLAUDE.md "Recommendation / feed
algorithm"), community-global (one feed for everyone, no per-user
personalization).

Each gate-passing image is scored on three pools and the blend decides its
rank:

- consensus   -- high recency-weighted rating mean, low variance, enough votes
- contested   -- high recency-weighted rating variance (mirrors divisive
                 engagement-boosting on real platforms)
- exploration -- few/no ratings, so new candidates surface and get rated
                 rather than the same debated set recirculating

Rating mean AND variance use the same recency weighting as training
(w = exp(-lambda_decay * age_days)), so an image that was contested but has
since settled drops out of the contested pool over time.

The blend weights (config FEED_MIX) and the per-image pool math are
deterministic and independent of the rest of the catalogue, so pagination
is stable. At MVP scale this loads the whole passed catalogue + all
ratings per request -- fine for now, revisit if it gets big (nubbins.md).

The response deliberately exposes no variance / pool / "contested" fields
-- the UI must not surface the drift mechanism (CLAUDE.md).
"""

from __future__ import annotations

import math
import time

from flask import current_app

from ..db import get_db

DEFAULT_LIMIT = 20
MAX_LIMIT = 50

# Vote-count saturation: sat(weight) = weight / (weight + SAT_K). ~SAT_K
# effective (recency-weighted) votes gets an image halfway to "fully rated".
SAT_K = 3.0


def _pool_terms(ratings: list[tuple[float, float]], lambda_decay: float, now: float):
    """(recency-weighted mean, variance, total weight, raw count) for one image.

    `ratings` is a list of (rating, created_at).
    """
    if not ratings:
        return 0.0, 0.0, 0.0, 0
    wsum = 0.0
    weighted_val = 0.0
    for rating, created_at in ratings:
        w = math.exp(-lambda_decay * (now - created_at) / 86400.0)
        wsum += w
        weighted_val += w * rating
    mean = weighted_val / wsum
    var = sum(
        math.exp(-lambda_decay * (now - created_at) / 86400.0) * (rating - mean) ** 2
        for rating, created_at in ratings
    ) / wsum
    return mean, var, wsum, len(ratings)


def _blended_score(mean: float, var: float, weight: float, mix: tuple[float, float, float]) -> float:
    sat = weight / (weight + SAT_K)
    consensus = (mean / 5.0) * (1.0 / (1.0 + var)) * sat
    contested = (var / (1.0 + var)) * sat
    exploration = 1.0 / (1.0 + weight)
    total = mix[0] + mix[1] + mix[2]
    return (mix[0] * consensus + mix[1] * contested + mix[2] * exploration) / total


def get_feed(limit: int = DEFAULT_LIMIT, offset: int = 0) -> dict:
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    cfg = current_app.config
    lambda_decay = cfg["LAMBDA_DECAY"]
    mix = cfg["FEED_MIX"]
    now = time.time()

    db = get_db()
    images = db.execute(
        """
        SELECT i.id, i.score, i.caption, i.created_at, u.username AS owner
        FROM images i
        JOIN users u ON u.id = i.owner_id
        WHERE i.passed = 1
        """
    ).fetchall()

    ratings_by_image: dict[str, list[tuple[float, float]]] = {}
    for row in db.execute("SELECT image_id, rating, created_at FROM ratings").fetchall():
        ratings_by_image.setdefault(row["image_id"], []).append((row["rating"], row["created_at"]))

    ranked = []
    for img in images:
        mean, var, weight, count = _pool_terms(
            ratings_by_image.get(img["id"], []), lambda_decay, now
        )
        ranked.append(
            {
                "sort_key": (
                    _blended_score(mean, var, weight, mix),
                    img["created_at"],
                ),
                "item": {
                    "id": img["id"],
                    "owner": img["owner"],
                    "score": img["score"],
                    "caption": img["caption"],
                    "created_at": img["created_at"],
                    "rating_count": count,
                    "rating_mean": mean if count else None,
                    "image_url": f"/api/images/{img['id']}/file",
                },
            }
        )

    ranked.sort(key=lambda r: r["sort_key"], reverse=True)
    page = [r["item"] for r in ranked[offset : offset + limit]]
    next_offset = offset + limit if offset + limit < len(ranked) else None
    return {"items": page, "next_offset": next_offset}
