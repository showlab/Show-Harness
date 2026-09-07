"""Browser-based teleoperation + rollout recorder (web counterpart of core/teleop/single.py).

Modules:
    sim.py      synthetic tabletop pick-and-place world (robot + rendered cameras),
                so the full UI -> controller -> recorder pipeline runs on a machine
                without CAN/ROS/cameras.
    backend.py  TeleopBackend: the single robot-owning worker thread, recording and
                the task-completion rule that gates Stop.
    server.py   stdlib HTTP server: static UI, MJPEG camera streams, JSON API.
    static/     the browser UI (see .ai/agent_todo/uidesign.png).

Launcher: gumi/collect_rollouts_web.py.
"""
