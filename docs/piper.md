# AgileX Piper runbook

Dual-arm AgileX Piper rig driven by the Show-Harness loop: zero-shot single arm
(`scripts/run_real.py --arm`), zero-shot dual arm (`scripts/run_real_dual.py`), or fine-tuned
(`scripts/run_real_mvtoken.py` / `scripts/run_real_dual_mvtoken.py`). All robot and camera I/O
goes over ROS topics.

## 1. Hardware prerequisites

- Two Piper arms, each on its own USB-CAN adapter (`can_left` / `can_right`,
  1 Mbaud). Enter your adapters' serials at the top of
  `scripts/piper/run_can.sh` (discover them with `run_can.sh --list`).
- ROS Noetic plus the AgileX `cobot_magic` workspaces, built under
  `$COBOT_MAGIC_DIR` (default `~/cobot_magic`):
  `Piper_ros_private-ros-noetic` (arm nodes) and `camera_ws` (`astra_camera`).
- Three Orbbec DaBai DC cameras: front (`/camera_f/color/image_raw`) plus one
  wrist camera per arm (`/camera_l`, `/camera_r`).
- A conda env named `aloha` providing `piper_sdk` (only
  `scripts/piper/run_arm.sh` uses it; everything else runs the repo venv).
- A display for the live view and keyboard override (`--no-show` disables both).

## 2. One-time setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/requirements-real.txt

cp configs/site/piper_arms.yaml.example configs/site/piper_arms.yaml
cp configs/secrets.env.example configs/secrets.env   # set GEMINI_API_KEY (default backend)
```

`configs/site/piper_arms.yaml` holds the per-arm calibration (`arms.left` /
`arms.right`: wrist camera topic, Z floor, poses). The configs refuse to run
without it. `configs/secrets.local.env` optionally overlays per-machine values.

## 3. Bring-up (each session)

```bash
scripts/piper/run_can.sh        # once per boot; needs sudo
scripts/piper/run_cameras.sh    # terminal 1: waits until all 3 streams publish
scripts/piper/run_arm.sh        # terminal 2: BOTH arm nodes, software-control mode
```

`run_arm.sh` closes both grippers to width 0 when it enables the arms — clear
all fingers first. Each launcher stops cleanly on Ctrl+C; the shared detached
roscore stays up until you `pkill -f roscore`.

Resets: `scripts/piper/go_begin.sh` sends both arms to the start pose,
`go_rest.sh` parks them; `go_begin.sh --list` shows the named poses.

## 4. Calibration (required before autonomous runs)

The shipped values are examples from the reference rig and must be recaptured.
Every autonomous run enforces each arm's `z_floor_m` as a hard limit and
refuses to descend below it.

Z floor, one arm at a time — rest that gripper on the tabletop, then:

```bash
scripts/piper/capture_z_floor.sh --arm left --write --robot-config configs/site/piper_arms.yaml
scripts/piper/capture_z_floor.sh --arm right --write --robot-config configs/site/piper_arms.yaml
```

Begin pose — position one arm, then capture it (this does not move the arm).
`configs/robot_piper.yaml` selects `begin_pose: natural`, so capture under
that name (or point `begin_pose:` at a name you chose):

```bash
scripts/piper/go_begin.sh --arm left --pose natural --capture --write \
    --robot-config configs/site/piper_arms.yaml
```

The capture also writes the mirrored pose for the other arm (the arms face
each other); verify it with `go_begin.sh --arm right --pose natural` and
recapture that arm directly if your mounts are not symmetric (`--no-mirror`
opts out). `go_rest.sh --arm left --capture --write --robot-config ...` saves
the park pose the same way. Pass `--robot-config configs/site/piper_arms.yaml`
on every capture as shown: the writes anchor on the `arms:` block, which lives
in the site file.

## 5. Preflight

With bring-up running, one command verifies the environment, the per-arm
calibration, the VLM backend (live call), and the ROS session (read-only):

```bash
.venv/bin/python scripts/check_setup.py --robot-config configs/robot_piper.yaml
```

## 6. Autonomous rollouts

Zero-shot, one arm (the VLM commands a single arm on the dual rig):

```bash
scripts/piper/run_rollout.sh --arm left --task "Pick up the banana and place it on the plate"
```

Zero-shot, both arms:

```bash
scripts/piper/run_rollout_dual.sh --mode B
```

`--mode B` (default) is one unified model commanding both arms, with `STILL`
for a waiting arm; `--mode A` runs two independent single-arm stacks and takes
`--task-left` / `--task-right`. Both wrappers source ROS and forward all flags
to `scripts/run_real.py` / `scripts/run_real_dual.py` with `configs/robot_piper.yaml`.

Fine-tuned — serve your fine-tuned checkpoint
(`MODEL=<base-model> FAMILY=<family> LORA=<adapter-name>=<path> bash scripts/serve_vlm.sh`, which
mounts the adapter's training chat template; the `finetuned_local` profile in
`configs/robot_piper_ft.yaml` points at `http://localhost:8000/v1`), source ROS in
the shell, then:

```bash
source /opt/ros/noetic/setup.bash
source "${COBOT_MAGIC_DIR:-$HOME/cobot_magic}/Piper_ros_private-ros-noetic/devel/setup.bash"

# single arm (reads robot.arm and the top-level z_floor_m of the config)
.venv/bin/python scripts/run_real_mvtoken.py \
  --robot-config configs/robot_piper_ft.yaml \
  --vlm-backend finetuned_local --model <adapter-name> \
  --version v4 --piper --task "..."

# dual arm (flattens the site arms calibration per side)
.venv/bin/python scripts/run_real_dual_mvtoken.py \
  --robot-config configs/robot_piper_ft.yaml \
  --model <adapter-name> --version v4 --once --task "..."
```

For the single-arm runner, pass `--vlm-backend finetuned_local` explicitly
(the flag's built-in default names a profile this config does not define) and
set the config's top-level `z_floor_m` to your captured floor (or pass
`--z-floor-m`). `--version` must match the checkpoint's training prompt under
`prompts/<version>/` (`v4 --piper` for the split egocentric prompt); the dual
runner's scheme flag (`--once` / `--twice` / `--chain`) must likewise match
how the adapter was trained. `docs/finetuned.md` documents these contracts.

During any rollout with `plugins.dagger: true` (the default), the teleop keys
override the model for that arm in real time — click the live-view window
first; its banner shows whether keys are armed.

## 7. Collecting demonstrations

- Keyboard (pygame, dual-arm): `scripts/piper/run_teleop.sh data/my_task` —
  records synchronized both-arm steps by default (`--mode A` for independent
  per-arm datasets); `P` starts/stops a recording.
- Browser (GUMI): `.venv/bin/python gumi/collect_rollouts_web_dual.py data/rollouts_dual`
  serves the dual compose-and-commit UI on port 8620; `gumi/README.md` covers
  the single-arm and agent-driven variants, and `--sim` runs without hardware.

## 8. Where results land

Each run writes `rollouts/real/<variant>/<MMDD>/task_0/<HH-MM-SS>/` (variant
examples: `Gemini-CoT`, `MVTOKEN`; override the root with `--log-dir`):
`steps.jsonl` (full reasoning in `steps.json`), `metadata.json`,
`summary.json`, `calibration.json`, `images/agentview/` + `images/wrist/`, and
`rollout_success.mp4` / `rollout_failure.mp4` (`rollout_live.mp4` while the
run is going).
