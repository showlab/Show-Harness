# Simulators

Show-Harness integrates two simulators. They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

Each integration keeps the deployment contracts: the same nine-token action
vocabulary, the same image transforms (`core/record/images.py`), and measured
per-config step calibration so one token means ~2 cm of physical travel.

## ManiSkill

Fine-tuned policy only, in ManiSkill 3's translation-only `pd_ee_delta_pos`
control mode (rotation locked — the assumption the atomic-token policy makes).

- `scripts/run_maniskill_mvtoken.py` — entry point; config `configs/robot_maniskill.yaml`.
  One config covers both protocols: it defaults to the fixed layout preset, and
  `--traj-id random --layout wide` reproduces the randomized object layout the
  training data was generated with. `--env-id` switches scenes.
- `scripts/maniskill/eval_batch.sh <config> <model> <n_episodes> [max_steps] [tag]`
  — batch eval over consecutive seeds; prints the closed-loop success rate.

ManiSkill needs its own Python environment; the
runner's remaining dependencies (numpy, requests, pyyaml, PIL, imageio) are
standard. Scenes are declared as one `SceneSpec` row each in
`core/sim/maniskill_scenes.py`. The default `BlockPAP-v1` is a real2sim replica
of the real Franka rig (table, pedestal, block + coaster, calibrated front
camera) and needs an RLinf checkout (`RLINF_ROOT`); `BlockStack-v1` is the same
rig with a stacking task. Stock tasks (`PickCube-v1`, `StackCube-v1`) run too
but with a much larger domain gap.

```bash
python scripts/run_maniskill_mvtoken.py --version v3 --model <adapter> --max-steps 60 --probe-axes
```

`--probe-axes` records the measured per-token TCP delta into `calibration.json`.

Calibration facts (load-bearing; full derivations in the yaml comments):

- `step_m: 0.026` x `sim_steps_per_decision: 2` is the commanded setting that
  achieves ~20.2 mm per decision (PD lag makes achieved < commanded) — see
  `configs/robot_maniskill.yaml`.
- `wrist_flip: both` and `agentview_square_size: 256` are training contracts;
  read the yaml comments before touching either, and regenerate data after any
  camera change.

Training-data generation lives in `scripts/trajectory/real2sim/`: a
simulator-agnostic core (`atomic_tokenizer.py` — token vocabulary, closed-loop
2 cm execution, Manhattan/RDP/chase planners, teleop-format writer) plus one
backend per simulator (`backends/maniskill.py`, `backends/robolab.py`). It
produces rollouts in exactly the real-teleop format, so sim and real data mix
without special cases. See `scripts/trajectory/real2sim/README.md`.

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

Training-data generation uses the same real2sim core. The RoboLab oracle parses
each task's own subtask declaration, so any single-object pick-and-place task
among the 120 generates without code changes (others raise `UnsupportedTask`).
Generate per task with `real2sim/robolab/record_demos.py` then
`follow_tokenize.py` (commands in
[real2sim/README.md](../scripts/trajectory/real2sim/README.md)), looping over the
task names `--list-tasks` prints to build a set. Quality gate before training:
`python scripts/robolab/check_dataset.py <dir>` (nonzero exit = do not train on
it). Prefer rotation-insensitive objects — the vocabulary has no wrist-rotation
token, so elongated objects are ungraspable.
