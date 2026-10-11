"""Serve the dashboard page and one run directory's status and event files (standard library only)."""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAGE_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js"}
_RUN_FILES = {"/status.json": "async_status.json", "/events.jsonl": "async_events.jsonl"}
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
}


def _handler(run_dir: str) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in _PAGE_FILES:
                self._send(os.path.join(_HERE, _PAGE_FILES[path]))
            elif path in _RUN_FILES:
                self._send(os.path.join(run_dir, _RUN_FILES[path]))
            else:
                self.send_error(404)

        def _send(self, file: str) -> None:
            try:
                with open(file, "rb") as fh:
                    body = fh.read()
            except OSError:
                self.send_error(404, f"{os.path.basename(file)} not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", _TYPES.get(os.path.splitext(file)[1], "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # quiet
            return

    return Handler


def serve(run_dir: str, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve ``run_dir``'s dashboard until interrupted."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"not a directory: {run_dir}")
    server = ThreadingHTTPServer((host, port), _handler(run_dir))
    print(f"dashboard for {run_dir} at http://{host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the async engine dashboard for one run directory.")
    ap.add_argument("run_dir", help="directory holding async_status.json and async_events.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    serve(a.run_dir, host=a.host, port=a.port)
