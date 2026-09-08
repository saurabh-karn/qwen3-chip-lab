"""Standard-library HTTP/JSON serving hooks for the reference model."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


JsonFunction = Callable[[dict[str, object]], object]


def make_handler(routes: dict[str, JsonFunction]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "qwen_ref/1"

        def _json(self, status: int, value: object) -> None:
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            route = routes.get(self.path)
            if route is None:
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1_048_576:
                    raise ValueError("request body is too large")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                self._json(200, route(payload))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(
    routes: dict[str, JsonFunction], host: str = "127.0.0.1", port: int = 8000
) -> None:
    ThreadingHTTPServer((host, port), make_handler(routes)).serve_forever()
