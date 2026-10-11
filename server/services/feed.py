"""Feed ranking. Config FEED_STRATEGY picks one of:

- blend      -- the three-pool blend below (CLAUDE.md "Recommendation / feed
                algorithm"), community-global.
- contested  -- most divisive horses first, interleaved with horses the
                viewer hasn't rated yet (newest first).
- consensus  -- highest-rated horses first, interleaved the same way.
- top        -- straight highest-rated, same for everyone.

"Highest rated" is a shrunk mean: the recency-weighted mean pulled toward
the platform-wide mean by SAT_K effective votes, so one 5-star vote can't
top the chart. "Unrated by the viewer" uses the ratings table -- an image
scrolled past without rating stays "unseen". FEED_UNSEEN_SHARE sets the
fraction of slots it gets. Logged out, nothing counts as rated.

The blend scores each gate-passing image on three pools:

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
is stable. (The interleaved strategies shift when the viewer rates
something mid-scroll, since that image leaves their "unrated" list.) At MVP scale this loads the whole passed catalogue + all
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

STRATEGIES = ("blend", "contested", "consensus", "top")
# Platform-wide prior for the shrunk mean when nothing is rated yet.
NEUTRAL_RATING = 2.5


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


def _shrunk_mean(mean: float, weight: float, prior: float) -> float:
    """Recency-weighted mean pulled toward `prior` by SAT_K effective votes."""
    return (weight * mean + SAT_K * prior) / (weight + SAT_K)


def _contested_score(var: float, weight: float) -> float:
    return (var / (1.0 + var)) * (weight / (weight + SAT_K))


def _interleave(primary: list[dict], unseen: list[dict], share: float) -> list[dict]:
    """Merge two ranked entry lists so ~`share` of slots come from `unseen`,
    dropping duplicates (an unrated image can be in both)."""
    out: list[dict] = []
    used: set[str] = set()
    p = u = taken_unseen = 0
    while p < len(primary) or u < len(unseen):
        want_unseen = taken_unseen < share * (len(out) + 1)
        if (want_unseen and u < len(unseen)) or p >= len(primary):
            item, from_unseen = unseen[u], True
            u += 1
        else:
            item, from_unseen = primary[p], False
            p += 1
        image_id = item["item"]["id"]
        if image_id in used:
            continue
        used.add(image_id)
        out.append(item)
        taken_unseen += from_unseen
    return out


def get_feed(limit: int = DEFAULT_LIMIT, offset: int = 0, user_id: int | None = None) -> dict:
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    cfg = current_app.config
    lambda_decay = cfg["LAMBDA_DECAY"]
    mix = cfg["FEED_MIX"]
    strategy = cfg["FEED_STRATEGY"]
    if strategy not in STRATEGIES:
        raise ValueError(f"FEED_STRATEGY must be one of {STRATEGIES}, got {strategy!r}")
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
    rated_by_viewer: set[str] = set()
    for row in db.execute("SELECT image_id, user_id, rating, created_at FROM ratings").fetchall():
        ratings_by_image.setdefault(row["image_id"], []).append((row["rating"], row["created_at"]))
        if user_id is not None and row["user_id"] == user_id:
            rated_by_viewer.add(row["image_id"])

    entries = []
    for img in images:
        mean, var, weight, count = _pool_terms(
            ratings_by_image.get(img["id"], []), lambda_decay, now
        )
        entries.append(
            {
                "mean": mean,
                "var": var,
                "weight": weight,
                "created_at": img["created_at"],
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

    ranked = [
        e["item"]
        for e in _rank(entries, strategy, mix, cfg["FEED_UNSEEN_SHARE"], rated_by_viewer)
    ]
    page = ranked[offset : offset + limit]
    next_offset = offset + limit if offset + limit < len(ranked) else None
    return {"items": page, "next_offset": next_offset}


def _rank(
    entries: list[dict],
    strategy: str,
    mix: tuple[float, float, float],
    unseen_share: float,
    rated_by_viewer: set[str],
) -> list[dict]:
    """Order per-image entries (mean/var/weight/created_at/item) by strategy.
    Ties break newest first."""

    def by(key) -> list[dict]:
        return sorted(entries, key=lambda e: (key(e), e["created_at"]), reverse=True)

    if strategy == "blend":
        return by(lambda e: _blended_score(e["mean"], e["var"], e["weight"], mix))

    total_weight = sum(e["weight"] for e in entries)
    prior = (
        sum(e["mean"] * e["weight"] for e in entries) / total_weight
        if total_weight
        else NEUTRAL_RATING
    )
    top_rated = by(lambda e: _shrunk_mean(e["mean"], e["weight"], prior))
    if strategy == "top":
        return top_rated

    primary = top_rated if strategy == "consensus" else by(
        lambda e: _contested_score(e["var"], e["weight"])
    )
    unseen = [e for e in by(lambda e: 0.0) if e["item"]["id"] not in rated_by_viewer]
    return _interleave(primary, unseen, unseen_share)
