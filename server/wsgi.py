"""Dev entrypoint:  python -m server.wsgi   (or point gunicorn at `server.wsgi:app`).

    HORSE_SECRET_KEY=... python -m server.wsgi
"""

from __future__ import annotations

from . import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True so a slow CLIP embed on one request doesn't block others.
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)
