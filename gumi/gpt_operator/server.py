"""Supervisory HTTP dashboard for :class:`GPTWebOperator`."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .operator import GPTWebOperator

STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_handler(operator: GPTWebOperator):
    class GPTDashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A003 - keep the robot console quiet
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._send_json({"ok": False, "message": "invalid JSON body"}, 400)
                return None
            if not isinstance(body, dict):
                self._send_json({"ok": False, "message": "body must be a JSON object"}, 400)
                return None
            return body

        def do_GET(self):  # noqa: N802 - http.server API
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                try:
                    body = (STATIC_DIR / "index.html").read_bytes()
                except OSError:
                    self._send_json({"ok": False, "message": "dashboard asset missing"}, 500)
                    return
                self._send_bytes(body, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(operator.snapshot())
            elif path == "/api/prompt":
                self._send_json({"prompt": operator.prompt})
            elif path.startswith("/api/frame/"):
                name = path.split("/api/frame/", 1)[1]
                frame = operator.frame(name)
                if frame is None:
                    self._send_json(
                        {"ok": False, "message": f"no GPT observation for {name!r} yet"},
                        404,
                    )
                else:
                    self._send_bytes(frame, "image/jpeg")
            elif path.startswith("/api/live/"):
                name = path.split("/api/live/", 1)[1]
                try:
                    frame = operator.target.snapshot_bytes(name)
                except Exception as exc:  # noqa: BLE001 - show target connectivity
                    self._send_json({"ok": False, "message": str(exc)}, 502)
                else:
                    self._send_bytes(frame, "image/jpeg")
            elif path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", 204)
            else:
                self._send_json({"ok": False, "message": "not found"}, 404)

        def do_POST(self):  # noqa: N802 - http.server API
            path = urlparse(self.path).path
            body = self._read_json()
            if body is None:
                return
            try:
                if path == "/api/run":
                    operator.resume()
                    result = {"ok": True, "message": "GPT operator running"}
                elif path == "/api/pause":
                    operator.pause()
                    result = {"ok": True, "message": "GPT operator paused"}
                elif path == "/api/step":
                    operator.step_once()
                    result = {"ok": True, "message": "one GPT step queued"}
                elif path == "/api/record/start":
                    result = operator.target_command("/api/start")
                elif path == "/api/record/stop":
                    result = operator.target_command("/api/stop")
                elif path == "/api/cancel":
                    result = operator.target_command("/api/cancel")
                elif path == "/api/task":
                    task = str(body.get("task", "")).strip()
                    if not task:
                        self._send_json({"ok": False, "message": "task is required"}, 400)
                        return
                    result = operator.target_command("/api/task", {"task": task})
                else:
                    self._send_json({"ok": False, "message": "not found"}, 404)
                    return
            except Exception as exc:  # noqa: BLE001 - surface control errors in the UI
                self._send_json({"ok": False, "message": str(exc)}, 500)
                return
            status = 200 if result.get("ok") else int(result.get("http_status", 400))
            result["operator"] = operator.snapshot()
            self._send_json(result, status)

    return GPTDashboardHandler


def serve(
    operator: GPTWebOperator, host: str = "0.0.0.0", port: int = 8630
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), build_handler(operator))
