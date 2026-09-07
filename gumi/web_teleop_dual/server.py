"""HTTP server for the DUAL-ARM teleop UI: three streams + the paired /api/step.

Reuses ``gumi.web_teleop.server.build_handler`` (JSON helpers,
/api/state, start/stop/cancel/task routing -- those submit to the dual backend
unchanged) and overrides what the dual rig genuinely changes: which index.html is
served, the THREE stream names, and the paired batch endpoint. The base file is
not modified.

Endpoints
---------
GET  /                      the dual-arm UI (this package's static/)
GET  /api/state             dual backend state (per-arm grippers, gap_m, history)
GET  /stream/agentview      MJPEG live stream -- front camera, BOTH arms visible
GET  /stream/wrist_left     MJPEG live stream -- left wrist
GET  /stream/wrist_right    MJPEG live stream -- right wrist
POST /api/start             start a rollout recording        (inherited routing)
POST /api/stop              stop + save, gated on task completion (inherited)
POST /api/cancel            discard the in-progress rollout, home BOTH arms
POST /api/task              {"task": "..."} -- retarget the task
POST /api/step              one or more SYNCHRONIZED (a_L, a_R) pairs:
                              {"left": "MV_FWD", "right": "STILL"}
                              {"left": "MV_FWD*3", "right": "MV_UP"}   <- STILL-padded
                              {"command": "L:w*3 R:u g"}               <- command box
                            -> {"ok", "results": [per-pair], "executed", "skipped",
                                "state"}
POST /api/move, /api/gripper  answered 400 by the backend with a pointer to
                            /api/step (single-arm calls have no place on a dual rig)
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.parse import urlparse

from gumi.web_teleop.server import (
    STREAM_BOUNDARY,
    TeleopServer,
    build_handler as build_base_handler,
)

from .dual_backend import parse_dual_step_request

STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAMS = ("agentview", "wrist_left", "wrist_right")
STEP_TIMEOUT_PER_PAIR_S = 25.0  # worst real pair ~2 s (gripper settle); wide margin


def build_handler(backend):
    BaseHandler = build_base_handler(backend)

    class DualTeleopHandler(BaseHandler):
        # -- routes --------------------------------------------------------
        def do_GET(self):  # noqa: N802 - http.server API
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            super().do_GET()

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/step":
                super().do_POST()
                return
            body = self._read_json_body()
            if body is None:
                return  # error already sent
            pairs, err = parse_dual_step_request(body, backend)
            if err is not None:
                status, message = err
                self._send_json({"ok": False, "message": message, "state": backend.state()}, status)
                return
            timeout = max(60.0, STEP_TIMEOUT_PER_PAIR_S * len(pairs))
            result = backend.submit("steps", pairs, timeout=timeout)
            status = int(result.pop("status", 200)) if not result.get("ok") else 200
            result["state"] = backend.state()
            self._send_json(result, status)

        # -- MJPEG -----------------------------------------------------------
        # Same loop as the base handler's _stream; only the whitelist differs
        # (three named views instead of two -- the tuple is baked into the base).
        def _stream(self, name: str) -> None:
            if name not in STREAMS:
                self._send_json(
                    {"ok": False, "message": f"unknown stream (have: {', '.join(STREAMS)})"}, 404
                )
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

        # -- helpers -------------------------------------------------------
        def _read_json_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._send_json({"ok": False, "message": "invalid JSON body"}, 400)
                return None

    return DualTeleopHandler


def serve(backend, host: str = "0.0.0.0", port: int = 8620) -> TeleopServer:
    return TeleopServer((host, port), build_handler(backend))
