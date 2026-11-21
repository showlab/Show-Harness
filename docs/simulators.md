# Simulators

Show-Harness integrates two simulators, RoboLab and LIBERO, plus a second embodiment, the
LeRobot SO-101 on an Isaac Lab scene ([last section](#so-101-isaac-lab-workshop)). Start with
RoboLab: the released `qwen3_5_2b_sim` adapter runs there as is. They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

Each integration keeps the deployment contracts: the same nine-token action
vocabulary, the same image transforms (`core/record/images.py`), and measured
per-config step calibration so one token means ~2 cm of physical travel.

The released `sim` split also contains ManiSkill demonstrations, used as training data for
the paper's sim-to-real experiment. They are data only: this repository does not include a
ManiSkill runner or scenes.

## RoboLab (Isaac Lab)

NVIDIA's [RoboLab](https://github.com/NVLabs/RoboLab) benchmark: 120 authored
Isaac Sim manipulation tasks with automated success predicates and photoreal
rendering.

- `scripts/run_robolab_mvtoken.py` — entry point; config `configs/robot_robolab.yaml`.
- `scripts/robolab/eval_batch.sh <model> <n_episodes> [task ...]` — batch eval,
  one process per task (Isaac Sim's cold start dominates otherwise).

The RoboLab checkout location comes from the `ROBOLAB_ROOT` environment
variable (see `robolab_root()` in `core/sim/robolab_task.py`). Isaac Sim pins
Python 3.11 with its own large dependency set, so run this repo's scripts with
the RoboLab venv's interpreter — it also satisfies everything the runner needs.
First launch requires accepting Isaac Sim's EULA (`OMNI_KIT_ACCEPT_EULA=YES`),
and `libGLU.so.1` must be loadable or Isaac Sim segfaults during stage creation
with a misleading backtrace; `launch_isaac` in `core/sim/robolab_task.py`
checks for it up front and prints the fix.

```bash
python scripts/run_robolab_mvtoken.py --list-tasks                                     # no Isaac Sim needed
python scripts/run_robolab_mvtoken.py --task RubiksCubeTask --dump-views --probe-axes --no-rollout   # calibration only, no VLM
python scripts/run_robolab_mvtoken.py --version v3 --task RubiksCubeTask --episodes 5
```

`--task` takes the task class name; `--episodes N` reuses one Isaac Sim app and
env across episodes; `--gui` shows the viewport (default headless).

Embodiment. The sim Franka wears the real rig's short yellow fingertips, not
the stock black fingers — for a policy that reads pixels, the stock finger is a
distribution shift in the middle of every frame.
`assets/robolab_franka/panda_short_finger.usda` replaces both finger visuals
and collision meshes (rebuild with
`scripts/trajectory/real2sim/robolab/make_short_finger_asset.py`; set
`ROBOLAB_PANDA_USD` to compare against the stock robot). The camera geometry
also mirrors the real rig rather than RoboLab's DROID default: Panda hand, and
the wrist camera centered between the fingers looking down the grasp axis
(`core/sim/robolab_franka.py`).

Calibration facts (load-bearing; measured tables in the yaml comments):

- RoboLab's relative differential-IK achieves a constant ~28% of any commanded
  delta, so `step_m: 0.072` commanded yields ~20.1 mm measured per decision —
  see `configs/robot_robolab.yaml`, including why settle steps do not help.
- `wrist_rotation_degrees: 270` / `wrist_flip: none` are measured for the Panda
  hand (fingertips at the top of the frame, no mirror). The camera contract
  lives only in this yaml; the runner and the data generators both read it via
  `core.config.camera_contract()` — never restate it elsewhere.

Training-data generation lives in `scripts/trajectory/real2sim/`: a
simulator-agnostic core (`atomic_tokenizer.py` — token vocabulary, closed-loop
2 cm execution, Manhattan/RDP/chase planners, teleop-format writer) plus one
backend per simulator (`backends/robolab.py`, `backends/libero.py`). It
produces rollouts in exactly the real-teleop format, so sim and real data mix
without special cases. The RoboLab oracle parses
each task's own subtask declaration, so any single-object pick-and-place task
among the 120 generates without code changes (others raise `UnsupportedTask`).
Generate per task with `real2sim/robolab/record_demos.py` then
`follow_tokenize.py` (commands in
[real2sim/README.md](../scripts/trajectory/real2sim/README.md)), looping over the
task names `--list-tasks` prints to build a set. Quality gate before training:
`python scripts/robolab/check_dataset.py <dir>` (nonzero exit = do not train on
it). Prefer rotation-insensitive objects — the vocabulary has no wrist-rotation
token, so elongated objects are ungraspable.

## LIBERO

The standard manipulation benchmark, plus the two robustness suites built on it.
All three are separate checkouts that install as the same `libero` package, so
they share one runner and one camera contract and differ only in which
interpreter you point at them:

| checkout | what it adds | protocol |
| --- | --- | --- |
| LIBERO | the original suites | `--suite <name> --episodes N` |
| LIBERO-plus | seven perturbation dimensions over one large suite | `--libero-plus`, every task once |
| LIBERO-PRO | perturbation suites beside the base one (`*_object`, `*_swap`, `*_lan`, `*_task`, `*_temp`, and the `*_env_*` / `*_with_*` / numbered families) | `--suite <perturbed> --tasks <base task names>` |

- `scripts/run_libero_mvtoken.py` — entry point; config `configs/robot_libero.yaml`.
- `scripts/libero/eval_batch.sh <model> <n_episodes> [suite ...]` — batch eval,
  one process per suite. `PY=` selects the checkout, `RUN_ARGS=` the protocol;
  everything else is identical across the three.
- `bash scripts/setup.sh libero <checkout>` builds that checkout's venv
  (Python 3.10, torch 2.6.0+cu124, this repo's requirements installed alongside).

```bash
PY=../LIBERO-PRO/.venv/bin/python bash scripts/libero/eval_batch.sh <adapter> 10 \
    libero_spatial libero_spatial_object libero_spatial_swap libero_spatial_lan libero_spatial_task
PY=../LIBERO-plus/.venv/bin/python RUN_ARGS=--libero-plus \
    bash scripts/libero/eval_batch.sh <adapter> 1 libero_spatial
```

Add `RUN_ARGS="--ignore-done"` when comparing against published numbers: the
other VLAs have no stop action, so their episodes end only on the environment's
success check or the step limit.

Facts that do not fail loudly (the reason each is handled in code):

- **The task sentence must come from the bddl**, never `task.language`:
  LIBERO-plus appends the variant name to it, and LIBERO-PRO derives it from the
  file stem, so in exactly the two suites that perturb the instruction
  (`*_lan`, `*_task`) it reports the *un*perturbed sentence. `bddl_language()`
  in the runner reads the bddl, the way the data generator did.
- **agentview is rendered upside down.** Upstream un-flips it with
  `obs[cam][::-1]`, a vertical flip; this repo rotates 180°. Both stand the
  scene upright but differ by a horizontal mirror, and they disagree about which
  way is right — under `[::-1]` world +y moves left on screen, under the
  rotation it moves right, and `MV_RIGHT` has to look like right. Our frames are
  therefore mirrored with respect to upstream figures. Generation and inference
  both read `configs/robot_libero.yaml`, so changing it invalidates every frame
  the policy was trained on.
- **Fixtures are not in the state vector.** Cabinets and stoves have no joints,
  so LIBERO samples their placement into `model.body_pos` on every `reset()`, a
  few mm apart each time — measured at 2.7% of agentview pixels off by >20/255.
  A single-episode re-run is therefore *not* the same scene as that episode
  inside a full run. Store `fixture_poses()` with the rollout and
  `set_fixture_poses()` before re-rendering; `--replay` already does.

## SO-101 (Isaac Lab Workshop)

The [SO-101](https://github.com/TheRobotStudio/SO-ARM100) is LeRobot's low-cost 5-DoF arm.
It runs here on NVIDIA's
[Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
(tag `v1.0`, Isaac Sim 5.1 / Isaac Lab), on its "vials → rack" task
`Lerobot-So101-Teleop-Vials-To-Rack-DR`. The real arm (a LeRobot follower behind the same
controller) is in [so101.md](so101.md).

| Path | What it is |
| --- | --- |
| `interpreters/so101_kinematics.py` | numpy FK/IK on the USD's own joint chain (`configs/so101_chain.json`) |
| `interpreters/so101_atomic_controller.py` | `So101ArmController` (motion core, I/O-agnostic) + `So101AtomicExec` (token executor) |
| `configs/primitives_so101.yaml` | MV_* vectors, rotation units, measured execution numbers |
| `configs/robot_so101_workshop.yaml` | task, scene variants, **the camera contract** |
| `scripts/trajectory/real2sim/backends/so101_workshop.py` | `AtomicSimEnv` backend: joints, cameras, contact sensor, success |
| `scripts/trajectory/real2sim/so101_workshop/` | `run_in_docker.sh`, `check_views.py`, `verify_motion.py`, USD extraction scripts, privileged `scene.py` |

Setup. The Workshop brings its own Isaac Sim / Isaac Lab stack in a Docker image, so nothing
from it is installed into this repo's environments. As with RoboLab, its dependencies stay
isolated from the base environment. Build the image once from a checkout at tag `v1.0`, then
run any script of this repo inside it through the wrapper. The wrapper mounts the repo
read-only, puts it on `PYTHONPATH`, and writes outputs to `rollouts/so101_workshop/`.

```bash
git clone -b v1.0 https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop ~/sim2real/Sim-to-Real-SO-101-Workshop
(cd ~/sim2real/Sim-to-Real-SO-101-Workshop && docker build -t teleop-docker -f docker/sim/Dockerfile .)
export WORKSHOP_ROOT=~/sim2real/Sim-to-Real-SO-101-Workshop
W=scripts/trajectory/real2sim/so101_workshop
bash $W/run_in_docker.sh $W/check_views.py   --out /workspace/out/check_views     # camera contract, exit 1 on mismatch
bash $W/run_in_docker.sh $W/verify_motion.py --out /workspace/out/verify_motion   # execution numbers, exit 1 on FAIL
```

Rotation units on 5 DoF. The shared vocabulary is unchanged, and the interpreter states what
the arm can realise. Pitch, Elbow and Wrist_Pitch are parallel, so the tool axis can only tilt
within the arm plane:

| Unit | On SO-101 |
| --- | --- |
| `RT_PITCH_FWD` / `RT_PITCH_BACK` | tool pitch in the arm plane, TCP held; about the arm-plane normal, which equals the shared world axis only while the arm points along +X |
| `RT_YAW_CW` / `RT_YAW_CCW` | hand turned through Wrist_Roll, TCP held; about the tool axis, which equals the shared world vertical only while the tool points straight down |
| `RT_ROLL_LEFT` / `RT_ROLL_RIGHT` | **not realisable**: always refused with reason `unsupported_on_5dof`; the arm does not move |

So neither realisable rotation is about a fixed world axis in every pose. For yaw the gap grows
with the tilt of the hand. In the default roll mode, at tool pitch *p* (90° = straight down) one
token gives 10° × sin *p* about the vertical, and the rest tips the jaw's closing axis out of
horizontal. At the 47° used for the measurements below, that is 7.3° of yaw per token and about
7° / 14° / 20° of tilt after one / two / three tokens (kinematics, reproduced on the unit-test
arm). That reaches the 13–22° of tilt at which the fixed finger dips and grasps fail (module
docstring of `interpreters/so101_atomic_controller.py`).

Refusals are whole-token and deterministic. Every token is checked before it runs (end-point
and mid-point IK feasibility, joint-limit margin, the whole hand's collision hull above the
table, workspace box). `So101AtomicExec.check()` answers without moving, so a generator never
records a frame for a token that would be refused. The backend does not claim
`supports_rotation`, because that contract means an arbitrary world-axis rotation. Execute
tokens with `So101AtomicExec`, not the generic `AtomicExec`.

Calibration facts (load-bearing; measured tables in the yaml comments):

- The SO-101's joint gains are low. The executor servos to a fixed target, capped at
  5 mm per 30 Hz control step, learns gravity sag at rest, and returns only once the arm is
  at rest. Measured by `verify_motion.py` (pitch 47°): one MV token moves 19.8–20.1 mm (96 tokens), at
  most 0.8 mm off-axis and 0.7 mm of overshoot. 80 random tokens and their reverse return
  within 0.1 mm. RT tokens turn 9.9–10.1° with at most 1.5 mm of TCP drift.
- Camera contract (`configs/robot_so101_workshop.yaml`): agentview `external_D455` rotated
  90°, wrist `ego` rotated 180°, both letterboxed to 256. Raw, neither view matches the
  shared prompt geometry. `check_views.py` measures it: agentview by TCP projection, wrist on
  the rendered frames (scene shift per executed token, and the fingers found as the pixels
  that ride with the camera). The rotated agentview is portrait, so its letterbox bars are
  left/right.
- Success is the Workshop's own `vial_placed` term, and it is lenient. It fires at the frame
  the fingers open if a just-held vial is within 45° of vertical and inside the rack's
  footprint, and it stays set. It does not check the hole, the vial staying upright, or which
  vial it was, so review rollouts by eye before trusting a success rate.
