# Plugins

A plugin mounts on one stage of the perceive-reason-act loop and is toggled by
one boolean in the `plugins:` block of the robot config. The framework contract
(also in `__init__.py`):

- **Self-contained.** `plugins/<name>/` bundles the implementation
  (`plugin.py`), a thin re-exporting `__init__.py`, and any prompt text as a
  co-located `<name>.txt` — a plugin can be read, reviewed, or removed as one
  unit. Nothing in `core/` (the VLM client and roles included) imports back from `plugins/`.
- **Disabled means byte-identical.** A disabled plugin still constructs, and
  every hook returns `""`/`[]`/`None`/identity, so the loop with a plugin off
  is exactly the loop without it. This is what makes single-plugin ablations
  meaningful.
- **Duck-typed VLM access.** Plugins that call the model take a `client` with
  `complete_json` / `complete_text` / `complete_token` — never a concrete
  client class.

## Hook surfaces

Runners construct plugin instances explicitly (no registry) and thread them
into three places. The plugins that are the same in every inference mode
(`dagger`, `video_ref`, `auto_release`) are built by `plugins/assembly.py`, so
their wiring lives in one place instead of being copied into each entry point;
it is a shared constructor, not a registry -- each runner still chooses which
plugins it builds.

1. **Build-time prompt transforms** — `apply(prompt_template)` rewrites the
   controller prompt in memory before the episode starts (`coords`,
   `wrist_frame`, `ego`, `action_ablation`; applied in that order). Source
   prompt files are never modified.
2. **Per-step prompt assembly** — context providers fill named `{placeholders}`
   in `prompts/controller*.txt` (`proprioception`, `mem_text`,
   `variable_step`, `action_chunk`, `rotation`, `affordance`); answer-protocol
   plugins own the whole output contract via `answer_tokens` /
   `output_contract()` / `fallback_answer()` / `map_response()`.
   Substitution is `str.replace`, never `str.format`, so fragment text may
   contain literal braces.
3. **Execution interceptors** — `recovery.before_decision/after_step`,
   `deepplan.is_pivot/resolve`, `dagger`'s preemption of in-flight decisions,
   `affordance`'s per-step annotation, and the interpreter-side hooks
   (`variable_step`, `smooth`, `rotation`) passed into the controller.

## Paper name ↔ code name

| Paper plugin | Code | Notes |
| --- | --- | --- |
| Multi-View Guidance | prompt scaffolding + `core/prompting/wrist_marker.py` | always-on view-role guidance; `view_select` extends it on the dual rig |
| Proprioception | `proprioception` | verbalized gripper height, contact, gripper state |
| Subtask Planning | `subgoal` | ordered plan with visually checkable completion criteria |
| Situated Planning | `deepplan` | deferred-branch `<REASON>` pivot for conditional tasks |
| Action Chunking | `action_chunk` | open-loop move plans while the target is far |
| Adaptive Step | `variable_step` | coarse/fine step from the shared `WRIST: YES/NO` signal |
| Visual Prompt | `affordance` | grounded contact-point dot with draw-and-verify |
| Action History | `mem_text` | recent-move line + anti-oscillation rules |
| Failure Recovery | `recovery`, `auto_release` | empty-grasp detection, reopen, plan rollback |

Not in the paper's table: `rotation` (offers the ROTATE units + eye-in-hand
yaw compensation), `smooth` (min-jerk setpoint ramp), `dagger` (live human
override), `video_ref` (in-context learning from a demo video), `ego` /
`wrist_frame` (per-rig direction-frame adapters), `coords` / `mcq`
(experimental prompt variants), `action_ablation` (the action-representation
ablation harness; keep `off`).

## Writing a plugin

Copy the layout of a small plugin (`mem_text` is a good template), honor the
disabled-inert rule, take your hyperparameters as constructor arguments read
from one config key each (see `core/launch.py`'s per-key helpers), and mount it
in `core/launch.py:make_runner`. If it contributes prompt text, put the exact
sentences in your `<name>.txt` under a `[section]` header so operators can
inspect and edit them without reading Python (`plugins/prompt_text.py`).

Two coupling rules to respect: `action_ablation`'s blind mode rewrites other
plugins' fragment sentences by regex, and `ego`/`wrist_frame` anchor on exact
phrases in the controller prompt — if you edit those sentences, update the
matchers with them.
