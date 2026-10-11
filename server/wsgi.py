"""Dev entrypoint:  python -m server.wsgi   (or point gunicorn at `server.wsgi:app`).

    HORSE_SECRET_KEY=... python -m server.wsgi

`python -m server.wsgi` is the localhost-only debug server and falls back to
a dev JWT secret. Imported by a real WSGI server (gunicorn), it refuses to
start without a proper HORSE_SECRET_KEY -- checked before the app is
created, so a misconfigured deploy fails with no side effects.
"""

from __future__ import annotations

import os

from . import create_app
from .config import Config
from .security import check_secret_key
from .services import model

if __name__ != "__main__":
    check_secret_key(Config.SECRET_KEY)

app = create_app()

# Preload CLIP so the first upload isn't slow. Skip the debug reloader's
# watcher process -- only its child (WERKZEUG_RUN_MAIN) serves requests.
if __name__ != "__main__" or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    model.warm_up(app)

if __name__ == "__main__":
    # threaded=True so a slow CLIP embed on one request doesn't block others.
    # tests/ excluded from the reloader: editing a test shouldn't restart the
    # API and throw away the loaded model.
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        threaded=True,
        exclude_patterns=["*/tests/*", "*\\tests\\*"],
    )
