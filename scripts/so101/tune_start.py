"""Match the real start pose to the sim by eye and save the per-joint zero offsets.

ExactScale fixes the angle SCALE; what remains is where each joint's zero is, which depends on
how symmetric the calibration sweep was. This moves the arm to the Workshop start pose and
shows the live front / wrist images half-blended with sim reference images of the same pose
(``--ref_dir`` with front.png and wrist.png, 640x480). Nudge joints until they overlap, then
Enter saves joint_offset.json (sim angle = encoder angle + offset_deg).

Keys: a/d shoulder_pan  w/s shoulder_lift  e/r elbow_flex  c/v wrist_flex  z/x wrist_roll
      (1 deg; Shift = 5 deg)   Enter = save and quit   q = quit without saving
An existing joint_offset.json is loaded first, so tuning can continue where it stopped.

Caveat (measured): a wrist camera mounted differently from the sim's makes the wrist overlay
pull wrist_flex the wrong way; trust the front overlay for the arm itself.

    python scripts/so101/tune_start.py --port COM3 --front 2 --wrist 1 --ref_dir path/to/refs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.so101.cameras import open_cameras  # noqa: E402
from core.so101.rig import DEFAULT_CONFIG, open_rig  # noqa: E402
from core.so101.units import JOINTS  # noqa: E402

KEYS = {"a": (0, -1), "d": (0, 1), "w": (1, 1), "s": (1, -1), "e": (2, -1), "r": (2, 1),
        "c": (3, -1), "v": (3, 1), "z": (4, -1), "x": (4, 1)}


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--port", default=None)
    ap.add_argument("--front", default="0")
    ap.add_argument("--wrist", default="2")
    ap.add_argument("--ref_dir", required=True, help="sim front.png / wrist.png of the start pose")
    ap.add_argument("--out", default="joint_offset.json")
    args = ap.parse_args()

    ref = {k: cv2.resize(cv2.imread(str(Path(args.ref_dir) / f"{k}.png")), (640, 480)) for k in ("front", "wrist")}
    o0 = np.zeros(6)
    if Path(args.out).exists():
        o0 = np.asarray(json.loads(Path(args.out).read_text(encoding="utf-8"))["offset_deg"], dtype=np.float64)
        print(f"[tune_start] starting from {args.out}: {np.round(o0, 1)}", flush=True)
    tmp = Path(args.out).with_suffix(".tmp.json")
    tmp.write_text(json.dumps({"offset_deg": o0.tolist()}))
    cams = open_cameras(args.front, args.wrist)
    rig = open_rig(args.config, port=args.port, joint_offset=str(tmp), cameras=cams)
    tmp.unlink()
    d = np.zeros(6)  # degrees moved this session
    saved = False
    try:
        rig.arm.move_to(rig.start_q, duration_s=3.0)
        while True:
            rows = []
            for k, cam in (("front", "external"), ("wrist", "wrist")):
                real = cv2.cvtColor(cams[cam].read(), cv2.COLOR_RGB2BGR)
                rows.append(np.hstack([ref[k], cv2.addWeighted(real, 0.5, ref[k], 0.5, 0)]))
            show = cv2.resize(np.vstack(rows), (960, 720))
            lines = ["left: sim / right: real + sim",
                     "a/d pan  w/s lift  e/r elbow  c/v wflex  z/x wroll  (Shift = 5 deg)",
                     "moved: " + " ".join(f"{j[:6]}={v:+.0f}" for j, v in zip(JOINTS[:5], d[:5])),
                     "Enter = save   q = quit"]
            for i, t in enumerate(lines):
                cv2.putText(show, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(show, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("tune_start", show)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key in (10, 13):
                o1 = o0 - d  # the pose that now LOOKS like the sim start is the sim start
                Path(args.out).write_text(json.dumps({"offset_deg": o1.round(2).tolist(), "joints": list(JOINTS),
                                                      "time": time.strftime("%F %T")}, indent=1))
                print(f"[tune_start] saved {args.out}: {np.round(o1, 1)}", flush=True)
                saved = True
                break
            ch = chr(key) if key < 128 else ""
            if ch.lower() in KEYS:
                j, sign = KEYS[ch.lower()]
                d[j] += sign * (5.0 if ch.isupper() else 1.0)
                rig.arm.move_to(rig.start_q + np.radians(d), duration_s=0.5)
    finally:
        cv2.destroyAllWindows()
        input("[tune_start] the arm goes limp on exit: support it and press Enter")
        rig.close()
    if not saved:
        print("[tune_start] not saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
