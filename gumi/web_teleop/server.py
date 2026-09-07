"""Stdlib HTTP server for the web teleop: static UI + MJPEG streams + JSON API.

No third-party web framework on purpose -- the robot hosts run a pinned ROS-era
Python environment, so the server sticks to http.server (threading) and the JPEG
frames the backend already encodes.

Endpoints
---------
GET  /                     the UI (static/index.html)
GET  /api/state            backend state (recording, task progress, gripper, ...)
GET  /snapshot/<view>      latest JPEG frame (one image; intended for VLM clients)
GET  /stream/agentview     MJPEG live stream (multipart/x-mixed-replace)
GET  /stream/wrist         MJPEG live stream
POST /api/start            start a rollout recording
POST /api/stop             stop + save (REFUSED with 409 until the task is done)
POST /api/cancel           discard the in-progress rollout, home the arm
POST /api/gripper          toggle GRASP <-> RELEASE
POST /api/move             body {"token": "MV_FWD"} -- one atomic step
POST /api/task             body {"task": "..."} -- retarget the task string
                           (REFUSED with 409 while a recording is open)
"""
from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAM_BOUNDARY = b"mjpeg-frame"

#: /api/step upper bounds. The per-token timeout is generous because a batch runs
#: serially on the worker thread and a gripper move settles for ~2 s on hardware.
MAX_BATCH_TOKENS = 64
STEP_TIMEOUT_PER_TOKEN_S = 20.0


from .backend import parse_step_request


def build_handler(backend):
    class TeleopHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # -- helpers -------------------------------------------------------
        def log_message(self, fmt, *args):  # noqa: A003 - silence per-request spam
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, ctype: str) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self._send_json({"ok": False, "message": "not found"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # -- routes --------------------------------------------------------
        def do_GET(self):  # noqa: N802 - http.server API
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send_json(backend.state())
            elif path.startswith("/snapshot/"):
                self._snapshot(path.split("/snapshot/", 1)[1])
            elif path.startswith("/stream/"):
                self._stream(path.split("/stream/", 1)[1])
            elif path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send_json({"ok": False, "message": "not found"}, 404)

        def _snapshot(self, name: str) -> None:
            """Return one current frame without opening a long-lived MJPEG stream.

            The VLM operator consumes this endpoint.  Use the backend's frame map as
            the whitelist so the same base handler works for both the two-view and
            three-view teleop servers.
            """
            with backend.frame_cond:
                frame = backend.frames.get(name)
                available = sorted(backend.frames)
            if frame is None:
                self._send_json(
                    {
                        "ok": False,
                        "message": (
                            f"snapshot {name!r} unavailable; "
                            f"available: {', '.join(available) or 'none'}"
                        ),
                    },
                    404 if available else 503,
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if not path.startswith("/api/"):
                self._send_json({"ok": False, "message": "not found"}, 404)
                return
            cmd = path[len("/api/"):]
            length = int(self.headers.get("Content-Length") or 0)
            body = {}
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except (ValueError, UnicodeDecodeError):
                    self._send_json({"ok": False, "message": "invalid JSON body"}, 400)
                    return
            if cmd == "step":
                self._do_step(body)
                return
            if cmd not in ("start", "stop", "cancel", "gripper", "move", "task"):
                self._send_json({"ok": False, "message": f"unknown command {cmd}"}, 404)
                return
            # "move" carries a token, "task" carries free text; everything else no arg.
            arg = body.get("token") if cmd != "task" else body.get("task")
            result = backend.submit(cmd, arg)
            status = int(result.pop("status", 200)) if not result.get("ok") else 200
            result["state"] = backend.state()
            self._send_json(result, status)

        # -- /api/step: one or many tokens, executed atomically on the worker ---
        def _do_step(self, body: dict) -> None:
            tokens, err = parse_step_request(body, backend)
            if err is not None:
                status, message = err
                self._send_json({"ok": False, "message": message, "state": backend.state()}, status)
                return
            if len(tokens) > MAX_BATCH_TOKENS:
                self._send_json(
                    {
                        "ok": False,
                        "message": f"too many tokens ({len(tokens)}); max {MAX_BATCH_TOKENS} per call",
                        "state": backend.state(),
                    },
                    400,
                )
                return
            timeout = max(60.0, STEP_TIMEOUT_PER_TOKEN_S * len(tokens))
            result = backend.submit("step", tokens, timeout=timeout)
            status = int(result.pop("status", 200)) if not result.get("ok") else 200
            result["state"] = backend.state()
            self._send_json(result, status)

        # -- MJPEG ----------------------------------------------------------
        def _stream(self, name: str) -> None:
            if name not in ("agentview", "wrist"):
                self._send_json({"ok": False, "message": "unknown stream"}, 404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY.decode()}",
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last_seq = -1
            try:
                while True:
                    with backend.frame_cond:
                        backend.frame_cond.wait_for(
                            lambda: backend.frame_seq != last_seq, timeout=2.0
                        )
                        frame = backend.frames.get(name)
                        last_seq = backend.frame_seq
                    if frame is None:
                        continue
                    self.wfile.write(
                        b"--" + STREAM_BOUNDARY + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                return  # viewer went away; nothing to clean up

    return TeleopHandler


class TeleopServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(backend, host: str = "0.0.0.0", port: int = 8600) -> TeleopServer:
    server = TeleopServer((host, port), build_handler(backend))
    return server
