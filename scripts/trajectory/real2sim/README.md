# real2sim — atomic-token training data from simulation

`scripts/trajectory/` holds the real-robot collectors (Franka / Piper / web teleop); this
directory is their simulation-side sibling. Scripted demonstrations driven by privileged
sim state produce rollouts in the exact real-teleop layout (`agentview/NNNN.png` +
`wrist/NNNN.png` + `actions.jsonl` + `metadata.json`), so sim and real data mix directly
in training. The converter to training format lives in the external LlamaFactory checkout
(`train/data_preparation/rollouts_to_alpaca.py`) and consumes sim
rollouts unchanged.

## Layout: the discretiser is the reusable core, the simulator a swappable peripheral

```
real2sim/
├── atomic_tokenizer.py      Sim-agnostic core: token vocabulary, closed-loop 2 cm
│                            execution, planners (Manhattan, RDP, chase), continuous
│                            demo recorder, teleop-format writer
├── backends/                One adapter per simulator, implementing AtomicSimEnv
│   ├── __init__.py          make_backend("maniskill" | "robolab", ...)
│   ├── maniskill.py
│   └── robolab.py           RoboLab / Isaac Lab (relative IK, 7-dim action)
├── maniskill/
│   ├── tasks.py             Task table + layout sampling (shared with deployment eval)
│   ├── oracle.py            Scheme A: privileged oracle (blockpap / blockstack)
│   ├── record_demos.py      Scheme D step 1: record continuous demos, export TCP tracks
│   ├── follow_tokenize.py   Scheme D step 2: re-execute tracks as 2 cm single-axis atoms
│   ├── make_dataset.py      LlamaFactory conversion + stats (step-size/oscillation gates)
│   └── merge_shards.py      Merge parallel shards, renumber
└── robolab/
    ├── tasks.py             Plan parsing (task's own subtasks) + gripper geometry
    ├── oracle.py            Scheme A: privileged oracle, task-agnostic
    ├── record_demos.py      Scheme D step 1
    ├── follow_tokenize.py   Scheme D step 2
    ├── calibrate_fingertip.py      Behavioural flange-to-fingertip sweep (see gotchas)
    └── make_short_finger_asset.py  Real rig's fingertip USD (see docs/simulators.md)
```

Everything that only calls the backend interface lives in the core: `DemoRecorder` (the
continuous servo recorder) and `gripper_events` are shared by both simulators, so each
simulator adds only the scripted demonstration itself — which privileged poses to servo
to, in what order. `make_dataset.py` (conversion + quality gates) is likewise sim-agnostic.

**A new simulator = one `backends/<sim>.py`** implementing the six `AtomicSimEnv` methods
(`tcp_pos / tcp_pose7 / gripper_width / apply_delta / grab_frames / success`);
`atomic_tokenizer.py` is untouched — token semantics, closed-loop execution, planning and
the on-disk format all live in the core. The RoboLab integration validated the split: it
added only `backends/robolab.py` and `robolab/{tasks,oracle}.py`, and its three
differences were absorbed by the backend — a 7-dim relative-IK action (rotation dims held
at 0 = orientation locked), gripper polarity opposite to ManiSkill (RoboLab: 1.0 closes,
0.0 opens), and `delta_bound_m` taken from the action config's `scale` (0.5) rather than
a controller bound.

**The RoboLab oracle needs no task table.** The ManiSkill scenes are authored in this
repo, so `maniskill/tasks.py` hand-writes each task's actor names and heights. RoboLab's
120 tasks declare their own targets
(`subtasks = [pick_and_place(object=["banana"], container="bowl")]`), and
`core.sim.robolab_task.task_targets` reads object and container off that declaration —
any single-object pick-and-place task generates with no code change. Other shapes
(stacking, sorting, multi-object clutter, tool use) raise `UnsupportedTask` instead of
silently grasping the wrong thing; they need their own planners.

### Why tokens are executed, not labelled

The policy deploys one token per decision, so training frames must sit on the same
discrete lattice the deployment loop walks. Labelling a continuous demo offline breaks
that: frame-to-frame displacements on a smooth path are diagonal while the label claims a
single axis — measured on this project's demos, 65-72% of steps were off-axis and 40-65%
were more than 1 cm off-axis. Unusable. Every generator here therefore records the
current frame first, then executes that token, closed-loop, pinning each `MV_*` to
`step_m` (2 cm) regardless of PD lag.

### Why the scene definitions are not here

`core/sim/maniskill_task.py` and `core/sim/maniskill_scenes.py` stay in `core/sim/`: they
are the deployment contract — the eval runner constructs its env from those modules, and
a private copy for data generation would let training and deployment camera/robot
definitions drift silently. Likewise frames go through `core.record.images.rotate_and_flip`: the
runner serves inference the output of the same function that writes the dataset.

`maniskill/tasks.py` therefore describes only layout sampling (`carried` / `target` /
`layout` / `drop_z` / `snap_layout`); task text and robot uids come from
`core.sim.maniskill_scenes.SCENES` via `tasks.task_text()` / `tasks.task_robot_uids()`.
The sampler is shared too: `core/sim/mvtoken_maniskill_runner.py` calls it for
`layout: wide` evaluation, so eval layouts and training layouts come from the same
distribution.

## Two schemes

| Scheme | Scripts | Method |
|---|---|---|
| **A** | `oracle.py` | State machine reads privileged sim poses and plans/executes **directly** in 2 cm single-axis tokens |
| **D** | `record_demos.py` then `follow_tokenize.py` | Record a **continuous** servo demo (multi-axis, ~6 mm steps) and export its TCP track; re-walk the track's RDP corners with 2 cm single-axis atoms, validating success per episode |

In both, a frame is a state the discrete controller actually reached, and every
frame-to-frame move is strictly single-axis. Earlier offline-decomposition schemes
(accumulator / waypoint-Manhattan / RL-h5) were removed: they took frames from the
continuous demo, so frames and labels mismatched. Scheme D takes only the demo's **path
shape** — every frame comes from token re-execution.

In Scheme D the follower emits every token itself (`TokenEpisode.chase`: read the real
TCP, pick the largest-error axis, emit that token, execute the full 2 cm, read again).
The continuous track contributes only RDP waypoints and gripper-event positions, not a
single token — which is exactly why frame and label cannot disagree. The track comes from
a scripted proportional servo on privileged poses rather than a motion planner (mplib
0.1.1 segfaults constructing its planner in this environment); the follower only needs
the track's shape — continuous, multi-axis, smooth — which the servo produces naturally.

## Usage — ManiSkill

Run with the ManiSkill Python environment (setup: `../../../docs/simulators.md`):

```bash
PY=<maniskill-env>/bin/python
GEN=scripts/trajectory/real2sim/maniskill

# Scheme A
$PY $GEN/oracle.py --task blockpap --episodes 100 \
    --table-tex 006 --agentview-square 256 --out <dir>/blockpap_oracle

# Scheme D (recording does not render and is fast; the follower renders
# ~50 s/episode at 94-100% success)
$PY $GEN/record_demos.py    --task blockpap --episodes 100 --seed0 20000 --out <stage>
$PY $GEN/follow_tokenize.py --tracks <stage>/tracks/blockpap --out <stage> \
    --table-tex 006 --agentview-square 256

# Merge parallel shards, then convert to training format + stats
$PY $GEN/merge_shards.py --shards <d1> <d2> ... --out <final> --move
LLAMAFACTORY_ROOT=<llamafactory-checkout> $PY $GEN/make_dataset.py --root <root> --version v3
```

`--task`: `blockpap` / `blockstack` — the RLinf real2sim rigs replicating the real Franka
setup (require an RLinf checkout via `RLINF_ROOT`).

## Usage — RoboLab (Isaac Lab)

Use RoboLab's own venv interpreter (`ROBOLAB_ROOT` required; Isaac Sim EULA on first run):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH=$ROBOLAB_ROOT/.deps/lib:$LD_LIBRARY_PATH   # libGLU
PY=$ROBOLAB_ROOT/.venv/bin/python
STAGE=rollouts/robolab_data

# Scheme A
$PY scripts/trajectory/real2sim/robolab/oracle.py \
    --task RubiksCubeTask --episodes 20 --agentview-square 256 --out $STAGE/cube_oracle

# Scheme D
$PY scripts/trajectory/real2sim/robolab/record_demos.py \
    --task RubiksCubeTask --episodes 10 --out $STAGE
$PY scripts/trajectory/real2sim/robolab/follow_tokenize.py \
    --tracks $STAGE/tracks/RubiksCubeTask --out $STAGE --agentview-square 256
```

The camera transform contract (rotation, flip, crop) is read from
`configs/robot_robolab.yaml` (`--robot-config`) — the same file the deployment runner
reads; leave the `--wrist-flip` / `--crop-aspect` overrides unset. RoboLab renders 16:9
while the training rigs are 4:3, so the wrist view is centre-cropped before storing
(`core.record.images.center_crop_to_aspect`).

`--task` takes the Task class name, which cannot be derived from the filename
(`bagel_on_plate_task.py` declares `BagelsOnPlateTask`);
`scripts/run_robolab_mvtoken.py --list-tasks` prints all 120 real class names.

Task selection matters more than on ManiSkill: MVTOKEN locks rotation, so the gripper
keeps its reset yaw and grasps along one fixed axis. Measured, `BananaInBowlTask`'s
banana bounding box is `[0.109, 0.178, 0.037] m` with the long axis roughly parallel to
the closing direction — 17.8 cm against an 8.5 cm maximum opening cannot be grasped
without rotating the wrist. Prefer orientation-insensitive objects; `RubiksCubeTask`
(58 mm cube into a bowl) is this project's reference task.

Two RoboLab-specific gotchas; both silently produce plausible-looking results:

- **Resetting after stepping freezes the env.** `RobolabEnv._reset_idx` treats a mid-run
  reset as "this episode terminated": the env is held at its final state and all further
  actions are zeroed in `step()` — an axis probe reads exactly 0.0000 m per token, with
  no error anywhere. Clear the eval state first (`env.reset_eval_state()`), as
  `core/sim/robolab_task.py` does before every reset.
- **The flange-to-fingertip offset must be measured, never taken from a spec sheet.**
  RoboLab's USD is flattened — every finger body reports the base_link pose, so the
  offset cannot be read off the kinematics — and the Robotiq 2F-85 datasheet figure
  (0.1628 m) is ~45 mm above the measured 0.118 m: the gripper aims high and closes on
  air while every episode still reports a clean token sequence. The constants (with
  measurement history, including the current short-finger value) live in
  `robolab/tasks.py`; re-measure with `robolab/calibrate_fingertip.py`, a behavioural
  sweep that only trusts whether the object actually lifted.

`robolab/make_short_finger_asset.py` rebuilds the short yellow fingertip USD the sim
Franka wears to match the real rig; embodiment and camera details in
`../../../docs/simulators.md`.

## Checklist for new datasets

- **`--agentview-square 256`**: the real-robot datasets and the deployment runner use a
  256x256 letterbox; without it frames store as raw 640x480 and vLLM's dynamic tiling
  splits one frame into 13 tiles.
- **`--table-tex 006` (wood) beats `white`**: a plain white tabletop turns the wrist view
  into a featureless gray field as the gripper closes in, starving a wrist-dependent
  policy exactly during pre-GRASP fine alignment. Texture applies at render time only, so
  the same tracks can be re-rendered on any tabletop without re-recording.
- **Changing a camera or a flip means regenerating the data** — training images are
  deployment images.
- **Quality gates** (`make_dataset.py` writes `stats.json`): `per_token_disp_mm_mean`
  ~20 with sigma < 1 mm, and `adjacent_opposite_pairs` = 0. For RoboLab also run
  `scripts/robolab/check_dataset.py` (an episode must not end on RELEASE; the
  post-release retreat must actually move).
- **Watch the preview video.** Every defect that has cost real time passed all numeric
  gates and was only visible by eye: a tilted gripper, an object knocked out of the
  container, a rotated wrist camera.
- **Tasks with a success tolerance below half a step need lattice snapping**
  (`snap_layout` in `maniskill/tasks.py`: BlockStack's 1 cm tolerance needs it,
  BlockPAP's 4.3 cm coaster does not).

## Deployment alignment

`configs/robot_maniskill*.yaml` sets `step_m: 0.026` x `sim_steps_per_decision: 2`,
measured at 20.2 +/- 0.2 mm per decision — matching the data's 2 cm; `wrist_flip: both`
and `agentview_square_size: 256` match the transforms applied at generation time.
Evaluate with `bash scripts/maniskill/eval_batch.sh <config> <model> <N>` or
`bash scripts/robolab/eval_batch.sh <model> <N> [task ...]`. Scene, camera and simulator
setup details: `../../../docs/simulators.md`.
