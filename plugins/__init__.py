"""Harness plugins, each fully self-contained.

Every plugin lives in its own subpackage under ``plugins/`` and bundles BOTH its
code and its prompt text file(s), so a plugin can be read, reviewed, or moved as
one unit. A plugin depends only on shared infrastructure (``core`` domain types, the
duck-typed VLM ``client``, ``core.vlm`` response types); nothing in ``core``
(``core.vlm`` included) imports back from ``plugins``.

Canonical layout for a plugin ``plugins/<name>/`` (follow this exactly):
  * ``plugin.py``   -- the implementation; the public plugin class lives here. A
    richer plugin may add focused sibling modules (e.g. subgoal's ``agent.py`` VLM
    sub-role), but the primary class is always ``plugin.py``.
  * ``__init__.py`` -- a THIN re-export of the public API only (no logic), with a
    docstring listing that API. Importers use ``from plugins.<name> import X``.
  * ``<name>.txt``  -- any prompt file lives beside the code and is loaded by the
    plugin itself; prompts are never scattered into the top-level ``prompts/``.
  * VLM-calling plugins take a duck-typed ``client`` (``complete_json`` / ``complete_text``
    / ``complete_token``) rather than importing a concrete client class.

Enable/disable: every plugin is toggled from the ``plugins:`` config block, resolved by
:class:`plugins.config.PluginsConfig`. A disabled plugin must still let the build run (the
active runner supplies the fallback), must leave the loop byte-identical, and must not
read another plugin's state.

Plugins by harness stage (paper terminology in parentheses where it differs):
  * ``plugins.subgoal``        -- expand a task + image into an ordered plan of subgoals
    (Subtask Planning).
  * ``plugins.proprioception`` -- add measured proprio context to the controller prompt
    (Proprioception).
  * ``plugins.recovery``       -- reopen and rewind when measured width shows empty grasp
    (Failure Recovery).
  * ``plugins.auto_release``   -- planner-free empty-grasp auto-reopen for the fine-tuned
    runners (Failure Recovery).
  * ``plugins.mem_text``       -- the move-history line + oscillation / empty-grasp rules
    (Action History).
  * ``plugins.variable_step``  -- coarse step when high/lifting (MV_UP) or when the TARGET
    is not yet in the wrist view (shared `WRIST: YES/NO` signal); fine step once close
    (Adaptive Step).
  * ``plugins.action_chunk``   -- while the TARGET is far, the model PLANS its next step_num
    distinct moves in one VLM call (`PLAN: ...`) and the runner runs them open-loop; one move
    per call once the TARGET is in the wrist view (Action Chunking).
  * ``plugins.deepplan``       -- a deferred-branch ``<REASON>`` checkpoint for conditional
    (IF-THEN) tasks: the planner emits an info-gathering prefix + one REASON pivot, and the
    runner resolves that pivot into concrete subgoals on the live scene when it is reached
    (Situated Planning).
  * ``plugins.affordance``     -- on stage entry, a dedicated pointing role grounds the
    stage's (target, affordance) into an exact front-view contact point (with a
    draw-and-verify correction pass); the runner premarks it as a colored dot (single/LEFT
    red, RIGHT blue) and REWRITES the controller's AFFORD field to "<part> = RED dot in
    <view>", so every DIRECTION/GRASP rule (all keyed on AFFORD) steers by the dot with
    no extra prompt lines (Visual Prompt). Offline check:
    ``python -m plugins.affordance.test_affordance`` (``--plan`` runs the real planner +
    per-stage grounding sweep).
  * ``plugins.rotation``       -- offers ROTATE_CW/CCW (a grasp-alignment gripper yaw) and,
    because the wrist camera is eye-in-hand, rotates a wrist-judged MV_* by the accumulated
    yaw so the VLM keeps reasoning in the wrist frame after the gripper has turned.
  * ``plugins.view_select``    -- dual mode B: the controller reports each arm's guiding view
    (WRIST rule A / FRONT rule B) and that view picks the move's motion frame per step
    (wrist / base), so both views' direction rules execute exactly (Multi-View Guidance).
  * ``plugins.video_ref``      -- distill a reference demo video into an ordered
    operation brief (arm, grasp part, destination) that the subgoal planner replicates on
    the live scene (in-context learning from video).
  * ``plugins.dagger``         -- real-time human keyboard override during the rollout (the
    teleop bindings, via the live-view window); an in-flight VLM decision superseded by
    human input is dropped instead of executed.
  * ``plugins.smooth``         -- ramp each move's setpoint along a min-jerk profile
    (interpreter-side motion realization).
  * ``plugins.coords``         -- opt-in axis wording for controller prompts (experimental).
  * ``plugins.mcq``            -- a multiple-choice answer protocol for the controller
    (experimental).
  * ``plugins.ego``            -- egocentric-view direction-rule rewrite for rigs whose
    front camera faces the same way as the arm.
  * ``plugins.wrist_frame``    -- execute MV_* in the wrist heading frame (hardware key
    ``motion_frame: wrist``), rewriting the direction rules to match.
  * ``plugins.action_ablation``-- the action-representation ablation harness
    (``action_ablation_mode: off | bare | letters | letters_blind``); research
    instrumentation, off by default.
"""
