"""``lerobot-calibrate`` for a lite install (no torch / accelerate / datasets).

Same arguments as lerobot-calibrate; core.so101.lerobot_io.lerobot_lite() stands in for the
packages that lerobot 0.4.1 imports on the motor-bus path but the arm never uses.

    python scripts/so101/calibrate.py --robot.type=so101_follower --robot.port=COM3 --robot.id=follower_arm_1

When a calibration file for that id exists, the first prompt asks whether to reuse it: type
``c`` then Enter to calibrate again (Enter alone only writes the old file back to the motors).
Put every joint at the MIDDLE of its range before the first Enter: a joint that crosses the
encoder wrap during the sweep records 0..4095 (seen on shoulder_lift and elbow_flex).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.so101.lerobot_io import lerobot_lite  # noqa: E402

lerobot_lite()

from lerobot.scripts.lerobot_calibrate import main  # noqa: E402

if __name__ == "__main__":
    main()
