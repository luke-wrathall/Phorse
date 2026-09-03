"""Feed queries.

SCAFFOLD ONLY: this returns gate-passing images newest-first with basic
rating aggregates. The real design (CLAUDE.md "Recommendation / feed
algorithm") blends three pools -- consensus / contested / exploration --
with recency-weighted mean *and variance*. That is a tracked nubbin; this
stub exists so the frontend has a real endpoint to render against.

Pagination is keyset-by-timestamp (`before`), which survives new uploads
mid-scroll better than OFFSET -- infinite scroll is a stated priority.
"""

from __future__ import annotations

from ..db import get_db

DEFAULT_LIMIT = 20
MAX_LIMIT = 50


def get_feed(limit: int = DEFAULT_LIMIT, before: float | None = None) -> dict:
    limit = max(1, min(limit, MAX_LIMIT))
    clauses = ["i.passed = 1"]
    params: list = []
    if before is not None:
        clauses.append("i.created_at < ?")
        params.append(before)
    where = " AND ".join(clauses)

    rows = get_db().execute(
        f"""
        SELECT i.id,
               i.score,
               i.caption,
               i.created_at,
               u.username                AS owner,
               COUNT(r.rating)           AS rating_count,
               AVG(r.rating)             AS rating_mean
        FROM images i
        JOIN users u        ON u.id = i.owner_id
        LEFT JOIN ratings r ON r.image_id = i.id
        WHERE {where}
        GROUP BY i.id
        ORDER BY i.created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()

    items = [
        {
            "id": row["id"],
            "owner": row["owner"],
            "score": row["score"],
            "caption": row["caption"],
            "created_at": row["created_at"],
            "rating_count": row["rating_count"],
            "rating_mean": row["rating_mean"],  # None until first rating
            "image_url": f"/api/images/{row['id']}/file",
        }
        for row in rows
    ]
    next_before = items[-1]["created_at"] if len(items) == limit else None
    return {"items": items, "next_before": next_before}
