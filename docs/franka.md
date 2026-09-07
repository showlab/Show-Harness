# Franka runbook

Single-arm Franka driven by the Show-Harness loop: zero-shot (`scripts/run_real.py`, a
frontier VLM plus the plugin harness) or fine-tuned (`scripts/run_real_mvtoken.py`,
a fine-tuned VLM emitting one action token per step).

## 1. Hardware prerequisites

- A Franka arm controlled by a NUC running the Polymetis-based `franka_server`
  (ZeroRPC, port 4242). The workstation running this repo talks to it over the
  network; the RPC surface the client expects is what
  `core/franka/franka_interface.py` calls (`get_ee_pose`,
  `update_desired_ee_pose`, `start_cartesian_impedance`, `control_gripper`, ...).
- Two Intel RealSense cameras plugged into the workstation: a third-person
  D435 and a wrist-mounted D405. Find their serials with
  `rs-enumerate-devices | grep Serial`.
- A display on the workstation for the live view and keyboard override
  (`--no-show` runs headless, but disables DAGGER).

## 2. One-time setup

Create the venv at `.venv` (the helper scripts under `scripts/franka/` require
this exact path):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/requirements-real.txt
```

Declare your rig. The config layering requires a site file and refuses to run
without it:

```bash
cp configs/site/franka.yaml.example configs/site/franka.yaml
# fill in: robot.nuc_ip, robot.external_camera_serial, robot.wrist_camera_serial
```

Add credentials for the hosted VLM (the zero-shot default is the `gemini`
profile in `configs/robot_franka.yaml`):

```bash
cp configs/secrets.env.example configs/secrets.env
# set GEMINI_API_KEY (default backend), or CHATGPT_API_KEY for --vlm-backend chatgpt
```

A gitignored `configs/secrets.local.env` overlays per-machine values; variables
already exported in your shell win over both files.

To run without hosted APIs, serve an open VLM locally and select the `local`
profile:

```bash
bash scripts/setup.sh serve            # creates .venv-vllm
bash scripts/serve_vlm.sh              # serves Qwen/Qwen3.5-2B on :8000
.venv/bin/python scripts/run_real.py --robot-config configs/robot_franka.yaml --vlm-backend local
```

## 3. Calibration (required before autonomous runs)

The shipped `z_floors:` and `poses:` values in the robot configs are examples
from the reference rig. Recapture them on your hardware: every autonomous run
enforces the active Z floor as a hard limit and refuses to descend below it,
so a floor captured on someone else's table is either unsafe or unreachable on
yours.

Z floor — rest the closed gripper on the work surface, then:

```bash
bash scripts/franka/capture_z_floor.sh --name default --write --activate
```

This is a read-only robot query (the arm never moves) that writes
`z_floors.default` into `configs/robot_franka.yaml`; `--activate` sets
`z_floor_name` so the floor is the active one. Keep one named floor per table
setup and switch per run with `--z-floor-name <name>`. The fine-tuned config
keeps its own block: repeat with
`--robot-config configs/robot_franka_ft.yaml`.

Begin pose — hand-guide the arm to the wanted start configuration, then:

```bash
bash scripts/franka/capture_pose.sh --name my_task --write --activate
```

This writes `poses.my_task` and, with `--activate`, sets `begin_pose` and
`move_to_begin_on_init: true` so every rollout homes there first. The name
`default` is reserved for the arm home and refused. Select per run with
`--begin-pose <name>`.

## 4. Preflight

One command verifies everything the first rollout depends on — environment,
site config and calibration, a live call to the selected VLM backend, and that
the NUC and both cameras answer (read-only, nothing moves):

```bash
.venv/bin/python scripts/check_setup.py --robot-config configs/robot_franka.yaml
```

Fix anything it flags (each failure prints the command that fixes it), then run.

## 5. Zero-shot rollout

```bash
.venv/bin/python scripts/run_real.py --robot-config configs/robot_franka.yaml
```

The task string comes from `task:` in the config; override with `--task "..."`.
Useful flags (`--help` for the full list): `--vlm-backend`, `--max-steps`,
`--z-floor-name`, `--begin-pose`, and `--mock-robot --mock-cameras` for a
hardware-free smoke test. With `plugins.dagger: true` (the default) you can
override the model mid-rollout from the keyboard: click the live-view window
first; the banner shows when keys are armed.

## 6. Fine-tuned rollout

Serve your fine-tuned action checkpoint:

```bash
MODEL=<base-model> FAMILY=<family> LORA=<adapter-name>=<path> bash scripts/serve_vlm.sh
```

The script mounts a jinja that reproduces the adapter's training-time rendering; serving
the base model's own template silently shifts the prompt off the training distribution.

The `finetuned_local` profile in `configs/robot_franka_ft.yaml`
points at `http://localhost:8000/v1`; select the adapter with `--model`. Then:

```bash
.venv/bin/python scripts/run_real_mvtoken.py \
  --robot-config configs/robot_franka_ft.yaml \
  --vlm-backend finetuned_local --model <adapter-name> \
  --version v3 --task "Pick up the banana and place it on the plate"
```

Pass `--vlm-backend finetuned_local` explicitly: the flag's built-in default
names a profile this config does not define. `--version` selects the prompt
contract under `prompts/<version>/` and must match what the checkpoint was
trained on (`v0`-`v3` are unified; `v4` has split prompts and needs
`--franka`). A mismatched prompt degrades silently, not loudly;
`docs/finetuned.md` documents these training contracts.

## 7. Collecting demonstrations

- Keyboard (pygame window):
  `.venv/bin/python scripts/trajectory/collect_rollouts.py data/rollouts` —
  `P` starts/stops a recording, movement keys are printed at startup.
- Browser (GUMI): `.venv/bin/python gumi/collect_rollouts_web.py data/rollouts_web --robot franka`
  serves a teleop UI on port 8600; `gumi/README.md` covers the dual-arm and
  agent-driven variants, and `--sim` runs all of them without hardware.

## 8. Where results land

Each run writes `rollouts/real/<variant>/<MMDD>/task_0/<HH-MM-SS>/` (variant
examples: `Gemini-CoT`, `MVTOKEN`; override the root with `--log-dir`):

- `steps.jsonl` — one compact record per step (full reasoning in `steps.json`)
- `metadata.json`, `summary.json`, `calibration.json`
- `images/agentview/`, `images/wrist/` — the frames the VLM saw
- `rollout_success.mp4` / `rollout_failure.mp4` — the analysis video
  (`rollout_live.mp4` while the run is still going)
