"""One exception type for expected client-facing failures + its JSON handler.

Anywhere in a request we can `raise ApiError("message", 404)` and the
client gets `{"error": "message"}` with that status. Keeps route code
free of repetitive try/return-tuple noise.
"""

from __future__ import annotations

from flask import Flask, jsonify


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        return jsonify(error=exc.message), exc.status

    @app.errorhandler(404)
    def _handle_404(_exc):
        return jsonify(error="not found"), 404
