"""Application factory for the Horse web API.

    from server import create_app
    app = create_app()

JSON-only API (no templates -- CLAUDE.md "Web app architecture"). All
routes are under /api. The SvelteKit frontend in web/ is a separate
process that talks to this over CORS.
"""

from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS

from . import db
from .config import Config
from .errors import register_error_handlers


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    db.init_app(app)
    register_error_handlers(app)

    from .blueprints import auth, feed, images, ratings

    for module in (auth, images, feed, ratings):
        app.register_blueprint(module.bp)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    return app
