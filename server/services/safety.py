"""The automated content check on anything about to go public.

Only uploads that pass the horse gate reach this: a rejected upload is
never stored or shown, so it's never sent to a third party either. If the
check can't run, the upload is refused -- nothing goes up unchecked.

This catches porn, gore and self-harm in images, and the same plus hate and
threats in captions. It does NOT catch child sexual abuse images: OpenAI's
`sexual/minors` category only reads text. That takes hash matching against
known images (not built yet, see nubbins.md).
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from flask import current_app
from PIL import Image

from ..errors import ApiError

MODERATION_URL = "https://api.openai.com/v1/moderations"
MODEL = "omni-moderation-latest"
TIMEOUT_SECONDS = 15
# Plenty for the classifier, and far less to send than a phone photo.
MAX_SIDE = 1024

# What gets an upload refused. Plain `violence`, `harassment` and `illicit`
# are left out on purpose: a knight on a horse or "this horse is an idiot"
# shouldn't need a human to sort out. Reports catch the real cases.
BLOCKED_CATEGORIES = (
    "sexual",
    "sexual/minors",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "violence/graphic",
    "hate",
    "hate/threatening",
    "harassment/threatening",
    "illicit/violent",
)


class CheckUnavailable(Exception):
    """The check couldn't run, so we don't know if this is safe."""


def _image_data_url(image: Image.Image) -> str:
    small = image.copy()
    small.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _ask_openai(image: Image.Image, caption: str | None) -> list[str]:
    key = current_app.config["OPENAI_API_KEY"]
    if not key:
        raise CheckUnavailable("HORSE_OPENAI_API_KEY is not set")
    inputs: list[dict] = [{"type": "image_url", "image_url": {"url": _image_data_url(image)}}]
    if caption:
        inputs.insert(0, {"type": "text", "text": caption})
    request = urllib.request.Request(
        MODERATION_URL,
        data=json.dumps({"model": MODEL, "input": inputs}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            results = json.load(response)["results"]
        return sorted(
            {
                category
                for result in results
                for category, hit in result["categories"].items()
                if hit and category in BLOCKED_CATEGORIES
            }
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CheckUnavailable(f"{type(exc).__name__}: {exc}") from exc


def blocked_categories(image: Image.Image, caption: str | None) -> list[str]:
    """Why this mustn't go public ([] if it's fine). Raises a 503 ApiError if
    the check can't run. HORSE_MODERATION=off turns it off, for local dev."""
    provider = current_app.config["MODERATION_PROVIDER"]
    if provider == "off":
        return []
    try:
        if provider != "openai":
            raise CheckUnavailable(f"unknown HORSE_MODERATION {provider!r}")
        return _ask_openai(image, caption)
    except CheckUnavailable as exc:
        current_app.logger.error("content check unavailable: %s", exc)
        raise ApiError("we can't check pictures right now, try again in a bit", 503)
