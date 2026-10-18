"""Feed ranking. There are two feeds (`kind`):

- new     -- horses the viewer hasn't rated yet, ranked by config
             FEED_STRATEGY (below). The site's front page.
- ranked  -- every horse, the model's most horse to least horse ("top"),
             same for everyone. The /ranked page.

"Hasn't rated" uses the ratings table -- a horse scrolled past without
rating stays unseen (no impressions tracking). Logged out, nothing counts
as rated, so the new feed is every horse.

Config FEED_STRATEGY picks the new feed's ranking, one of:

- contested  -- most divisive horses first, interleaved with the newest.
- consensus  -- highest-scored horses first, interleaved with the newest.
- blend      -- the three-pool blend below (CLAUDE.md "Recommendation / feed
                algorithm").
- top        -- straight highest-scored.

"Highest scored" is the model's own horseness score (images.score,
re-scored every retrain), not user ratings. FEED_NEWEST_SHARE sets the
fraction of slots the newest horses get, so fresh uploads -- which have no
ratings yet to be contested -- still surface.

Feed sessions: the client sends a `session` id it generates once per page
load, and the server remembers which images it has already served to that
(viewer, session), in order. Already-served positions never change -- a
refetch of loaded pages (e.g. after rating) returns the same images in the
same slots, with live details -- so nothing on screen jumps or vanishes.
Each *new* page is ranked fresh and skips everything already served, so
new uploads, retrain re-scores and the viewer's own ratings shape what
loads next. Without a session every request is ranked fresh.

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
is stable for already-loaded items within a feed session (see above). At MVP scale this loads the whole passed catalogue + all
ratings per request -- fine for now, revisit if it gets big (nubbins.md).

The response deliberately exposes no variance / pool / "contested" fields
-- the UI must not surface the drift mechanism (CLAUDE.md).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict

from flask import current_app

from ..db import get_db

DEFAULT_LIMIT = 20
MAX_LIMIT = 50

# Vote-count saturation: sat(weight) = weight / (weight + SAT_K). ~SAT_K
# effective (recency-weighted) votes gets an image halfway to "fully rated".
SAT_K = 3.0

STRATEGIES = ("blend", "contested", "consensus", "top")
FEEDS = ("new", "ranked")

# Images already served per feed session, keyed by (viewer user_id or None,
# feed kind, client session id), oldest session first. In-process memory: fine for the
# single dev server; multiple workers would each keep their own (nubbins.md).
SESSION_TTL_SECONDS = 3600
SESSION_MAX = 1000
SESSION_ID_MAX_LEN = 64
_sessions: OrderedDict[tuple[int | None, str, str], tuple[float, list[str]]] = OrderedDict()
_sessions_lock = threading.Lock()


def _session_page(
    key: tuple[int | None, str, str], offset: int, limit: int, compute_order
) -> tuple[list[str], bool]:
    """(image ids for this page, whether more exist) within a feed session.

    Positions already served are replayed as-is; past them, the page is
    filled from a fresh ranking minus everything already served.
    """
    now = time.time()
    end = offset + limit
    with _sessions_lock:
        while _sessions and (
            len(_sessions) >= SESSION_MAX
            or now - next(iter(_sessions.values()))[0] > SESSION_TTL_SECONDS
        ):
            _sessions.popitem(last=False)
        served = _sessions.setdefault(key, (now, []))[1]
        remaining: list[str] = []
        if len(served) <= end:
            already = set(served)
            remaining = [i for i in compute_order() if i not in already]
            take = max(0, end - len(served))
            served.extend(remaining[:take])
            remaining = remaining[take:]
        return served[offset:end], len(served) > end or bool(remaining)


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


def _contested_score(var: float, weight: float) -> float:
    return (var / (1.0 + var)) * (weight / (weight + SAT_K))


def _interleave(primary: list[dict], newest: list[dict], share: float) -> list[dict]:
    """Merge two ranked entry lists so ~`share` of slots come from `newest`
    (the newest-first list), dropping duplicates (an image is in both)."""
    out: list[dict] = []
    used: set[str] = set()
    p = u = taken_newest = 0
    while p < len(primary) or u < len(newest):
        want_newest = taken_newest < share * (len(out) + 1)
        if (want_newest and u < len(newest)) or p >= len(primary):
            item, from_newest = newest[u], True
            u += 1
        else:
            item, from_newest = primary[p], False
            p += 1
        image_id = item["item"]["id"]
        if image_id in used:
            continue
        used.add(image_id)
        out.append(item)
        taken_newest += from_newest
    return out


def get_feed(
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    user_id: int | None = None,
    session: str | None = None,
    kind: str = "new",
) -> dict:
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    cfg = current_app.config
    lambda_decay = cfg["LAMBDA_DECAY"]
    mix = cfg["FEED_MIX"]
    if kind not in FEEDS:
        raise ValueError(f"feed must be one of {FEEDS}, got {kind!r}")
    strategy = cfg["FEED_STRATEGY"] if kind == "new" else "top"
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

    # The new feed ranks only horses the viewer hasn't rated. Filtered here,
    # not out of `entries`: a horse rated mid-scroll keeps its loaded slot
    # (and live details) for the rest of the feed session.
    candidates = (
        [e for e in entries if e["item"]["id"] not in rated_by_viewer]
        if kind == "new"
        else entries
    )

    def compute_order() -> list[str]:
        ranked = _rank(candidates, strategy, mix, cfg["FEED_NEWEST_SHARE"])
        return [e["item"]["id"] for e in ranked]

    if session:
        page_ids, has_more = _session_page((user_id, kind, session), offset, limit, compute_order)
    else:
        order = compute_order()
        page_ids, has_more = order[offset : offset + limit], offset + limit < len(order)
    items_by_id = {e["item"]["id"]: e["item"] for e in entries}
    page = [items_by_id[i] for i in page_ids if i in items_by_id]
    return {"items": page, "next_offset": offset + limit if has_more else None}


def _rank(
    entries: list[dict],
    strategy: str,
    mix: tuple[float, float, float],
    newest_share: float,
) -> list[dict]:
    """Order per-image entries (mean/var/weight/created_at/item) by strategy.
    Ties break newest first."""

    def by(key) -> list[dict]:
        return sorted(entries, key=lambda e: (key(e), e["created_at"]), reverse=True)

    if strategy == "blend":
        return by(lambda e: _blended_score(e["mean"], e["var"], e["weight"], mix))

    top_scored = by(lambda e: e["item"]["score"])
    if strategy == "top":
        return top_scored

    primary = top_scored if strategy == "consensus" else by(
        lambda e: _contested_score(e["var"], e["weight"])
    )
    newest = by(lambda e: 0.0)
    return _interleave(primary, newest, newest_share)
