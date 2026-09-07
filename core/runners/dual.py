"""Dual-arm VLM closed-loop rollout runner (unified control, "Mode B").

The dual counterpart of :class:`core.runners.real.RealEpisodeRunner`. One
:class:`~plugins.subgoal.dual_plugin.DualSubgoalPlanner` call expands the task into TWO
concurrent subgoal tracks (one per arm), and each step ONE
:class:`~core.vlm.dual_roles.DualControllerAgent` call sees three images (front + both
wrists) and returns one atomic token PER ARM (``MV_*`` / ``GRASP`` / ``RELEASE`` /
``DONE`` / ``STILL``). Both tokens are executed **simultaneously** (one thread per
arm -- the arms are independent ROS nodes on separate CAN buses), so the arms move in
parallel instead of taking turns.

Control contract per arm is unchanged from the single-arm runner: ``DONE`` advances
that arm's track; ``DONE`` on an arm's final stage finishes that arm; BOTH arms
finished completes the task. ``STILL`` is the explicit no-op -- an arm waits for the
other (handover, a WAIT stage) or has finished its track. Waiting is expressed through
stage descriptions/completions the VLM reasons over, never through runner-side gates.

Recovery (measured-width grasp verification), move memory, per-stage step caps, and
logging all run PER ARM, reusing the same plugins as the single-arm path. Logging
reuses :class:`~core.record.episode_logger.EpisodeLogger` with the two wrist views side by
side in the wrist slot, so the on-disk layout and the analysis video stay familiar.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import core.ui.console as console
from core.action_units import MOVE_ATOMS
from core.runners.preemption import InterruptibleDecider
from core.record.episode_logger import EpisodeLogger, status_flags
from core.record.images import to_uint8_hwc
from core.runners.real import EMPTY_GRASP_LABEL, descend_travel, video_fps
from core.v0_types import EpisodeResult, Subgoal, V0Config
from core.vlm.dual_roles import SIDES, STILL_TOKEN

DONE_TOKEN = "DONE"
GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
NO_DIRECTION = "NONE"


@dataclass
class _ArmState:
    """Per-arm rollout bookkeeping (the single-arm runner's loop locals, per side)."""

    controller: Any
    recovery_plugin: Any
    subgoals: list[Subgoal] = field(default_factory=list)
    index: int = 0
    stage_start_step: int = 0
    recent_moves: list[str] = field(default_factory=list)
    current_direction: Optional[str] = None
    recovery_note: str = ""
    finished: bool = False
    failed_reason: str = ""
    # True once a finished arm has been parked back at its BEGIN pose (see _retire_arm).
    homed: bool = False
    # Last MV_DOWN's (actually travelled, commanded) height in meters, or None when the
    # arm's last motion was not a descent. Fed to the proprioception tool, which reports
    # a descent that did not happen (see _note_descend).
    descend: Optional[tuple[float, float]] = None

    @property
    def subgoal(self) -> Optional[Subgoal]:
        if self.finished or self.index >= len(self.subgoals):
            return None
        return self.subgoals[self.index]


class DualEpisodeRunner:
    """Closed-loop unified dual-arm VLM rollout on the physical Piper rig."""

    def __init__(
        self,
        *,
        session: Any,
        controllers: dict[str, Any],
        planner: Any,
        controller_agent: Any,
        logger: EpisodeLogger,
        config: V0Config,
        task: str,
        max_steps: int,
        loop_period_s: float,
        debug: bool,
        recovery_tools: Optional[dict[str, Any]] = None,
        recent_moves_max: int = 3,
        viewer: Any = None,
        home_arm: Optional[Callable[[str], None]] = None,
        view_select_plugin: Any = None,
        affordance_plugin: Any = None,
        dagger_plugin: Any = None,
    ) -> None:
        self.session = session
        self.planner = planner
        self.agent = controller_agent
        self.logger = logger
        self.config = config
        self.task = str(task)
        self.max_steps = int(max_steps)
        self.loop_period_s = float(loop_period_s)
        self.debug = bool(debug)
        self.recent_moves_max = max(1, int(recent_moves_max))
        self.viewer = viewer
        # Multi-view action selection: maps each arm's reported guiding view to the
        # motion frame its move executes in (WRIST -> wrist, FRONT -> base). None or
        # disabled -> no overrides; every move runs in the controller's configured frame.
        self.view_select_plugin = view_select_plugin
        # Affordance dots: grounds each arm's stage contact point on stage entry and
        # premarks it on the front image every consumer sees (controller, live view,
        # video); once an arm is wrist-guided it also tracks the point on that arm's
        # wrist frame per step. None/disabled -> the frames pass through untouched.
        self.affordance_plugin = affordance_plugin
        # Each arm's guiding view from the PREVIOUS step ("WRIST"/"FRONT"), used to
        # gate wrist-dot tracking (the report for step N is only known after step N's
        # decision, so the premark uses N-1's). Empty until the first decision.
        self._prev_views: dict[str, Optional[str]] = {}
        # DAGGER: real-time human keyboard override (plugins.dagger). Keys arrive on
        # the live view's stream thread; the runner consumes them at step
        # boundaries and preempts/drops in-flight VLM decisions (see
        # _decide_interruptible). None/disabled -> the loop is byte-identical.
        self.dagger_plugin = dagger_plugin
        # The single in-flight preemptible decision (DAGGER only).
        self._decider: Optional[InterruptibleDecider] = None
        # Sends ONE arm back to its BEGIN pose (the caller owns the poses/hardware).
        # None -> a finished arm just holds where it stopped.
        self.home_arm = home_arm
        self._replans = 0
        recovery_tools = recovery_tools or {}
        self.arms: dict[str, _ArmState] = {
            side: _ArmState(
                controller=controllers[side],
                recovery_plugin=recovery_tools.get(side),
            )
            for side in SIDES
        }

    # -- main loop ---------------------------------------------------------
    def run(self) -> EpisodeResult:
        success = False
        end_reason = "max_steps_exceeded"
        steps = 0
        raw_plan = ""
        self._replans = 0
        self._t0 = time.monotonic()
        self._last_vlm_ms: Optional[int] = None
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"

        # Start from a known, open-gripper state on BOTH arms.
        self._execute({side: RELEASE_TOKEN for side in SIDES})

        try:
            obs = self.session.get_observation()
            agentview, wrist_left, wrist_right = self._images(obs)
            tracks, raw_plan = self.planner.plan(
                self.task, agentview, wrist_left=wrist_left, wrist_right=wrist_right,
                debug=self.debug,
            )
            for side in SIDES:
                self.arms[side].subgoals = tracks.get(side, [])
                # An empty track is a valid plan: that arm has no work.
                self.arms[side].finished = not self.arms[side].subgoals
            self.logger.write_plan(
                {
                    "task": self.task,
                    "raw_plan": raw_plan,
                    "subgoals": {
                        side: [sg.to_prompt_dict() for sg in self.arms[side].subgoals]
                        for side in SIDES
                    },
                }
            )
            self._print_plan()

            for step_idx in range(self.max_steps):
                loop_started = time.monotonic()

                # Per-arm stage step cap: abandon the stage and move on (per side).
                for side in SIDES:
                    self._apply_stage_cap(self.arms[side], step_idx)

                # Both tracks done (a per-arm DONE last step, or a cap-finish above):
                # final visual check of the WHOLE task, then success / replan / fail.
                # (Not counted in `steps`: no robot motion happens on this iteration.)
                if all(arm.finished for arm in self.arms.values()):
                    outcome, reason = self._completion_outcome(step_idx)
                    if outcome == "success":
                        success = True
                        end_reason = "plan_complete"
                        break
                    if outcome == "failed":
                        end_reason = reason
                        break
                    continue  # replanned: fresh tracks, keep stepping

                # One arm done while the other still works: park it at BEGIN so it stops
                # blocking the workspace / occluding the front view. Done BEFORE the
                # observation, so the still-working arm's next decision sees a clear scene.
                for side in SIDES:
                    self._retire_arm(side)

                steps = step_idx + 1

                obs = self.session.get_observation()
                agentview, wrist_left, wrist_right = self._images(obs)
                agentview, wrist_left, wrist_right = self._premark_affordance(
                    agentview, wrist_left, wrist_right
                )
                self._show_live(
                    step_idx,
                    (agentview, wrist_left, wrist_right),
                    "deciding ...",
                    steps_done=step_idx,
                )

                # Recovery pre-decision (per arm): a measured-width intervention
                # overrides that arm's token for this step.
                overrides: dict[str, Optional[Any]] = {}
                for side in SIDES:
                    overrides[side] = self._recovery_before_decision(side, obs)
                    if overrides[side] is not None and self.affordance_plugin is not None:
                        # The grasp measurably failed, so the scene was likely
                        # disturbed: re-ground this arm's dot on the next observation.
                        self.affordance_plugin.clear(side)

                decision = None
                human: dict[str, tuple[str, str]] = {}
                tokens: dict[str, str] = {}
                for side in SIDES:
                    arm = self.arms[side]
                    if arm.finished:
                        tokens[side] = STILL_TOKEN
                    elif overrides[side] is not None and overrides[side].token:
                        tokens[side] = overrides[side].token
                # Sides whose token did NOT come from this step's decision (finished /
                # recovery override): the model's view report does not apply to them,
                # and a human DAGGER intent does not either (a measured-width recovery
                # keeps priority -- it corrects a physical fact, not a preference).
                overridden = set(tokens)
                if len(tokens) < len(SIDES):
                    decide_kwargs = dict(
                        task=self.task,
                        subgoals={
                            side: (
                                arm.subgoal.to_prompt_dict()
                                if arm.subgoal is not None
                                else None
                            )
                            for side, arm in self.arms.items()
                        },
                        gripper_states={
                            side: self._observed_gripper_state(side, obs)
                            for side in SIDES
                        },
                        recent_moves={
                            side: self._recent_moves_text(arm.recent_moves)
                            for side, arm in self.arms.items()
                        },
                        previous_directions={
                            side: arm.current_direction or NO_DIRECTION
                            for side, arm in self.arms.items()
                        },
                        recovery_contexts={
                            side: self._recovery_prompt(side)
                            for side in SIDES
                        },
                        proprios={side: self._proprio(side, obs) for side in SIDES},
                        agentview_image=agentview,
                        wrist_left_image=wrist_left,
                        wrist_right_image=wrist_right,
                        debug=self.debug,
                    )
                    if getattr(self.dagger_plugin, "enabled", False):
                        # DAGGER: the VLM call is preemptible -- the moment human
                        # keys arrive it is abandoned (and its result later dropped);
                        # None means the human owns this step.
                        decision = self._decide_interruptible(decide_kwargs)
                    else:
                        decision = self.agent.decide(**decide_kwargs)
                    if decision is not None:
                        for side in SIDES:
                            tokens.setdefault(side, decision.tokens.get(side, STILL_TOKEN))
                        latency_s = (decision.payload or {}).get("latency_s")
                        if latency_s is not None:
                            self._last_vlm_ms = int(round(float(latency_s) * 1000.0))
                    else:
                        # Human step: consume the intents; an arm without one holds
                        # STILL (never execute a model token beside a human one --
                        # the joint decision it came from is stale by definition).
                        human = self.dagger_plugin.drain()
                        for side in SIDES:
                            if side in tokens:
                                continue
                            token, kind = human.get(side, (STILL_TOKEN, "still"))
                            if kind == "gripper":
                                # Teleop's flush-time contract: the toggle direction
                                # comes from the controller's ACTUAL state.
                                token = (
                                    RELEASE_TOKEN
                                    if self.arms[side].controller.gripper_closed
                                    else GRASP_TOKEN
                                )
                            tokens[side] = token

                every = int(getattr(self.config, "controller_prompt_log_every", 0) or 0)
                if decision is not None and every > 0 and step_idx % every == 0:
                    prompt_text = getattr(self.agent, "last_prompt", "")
                    if prompt_text:
                        self.logger.save_controller_prompt(step_idx, prompt_text)

                # Multi-view action selection: each decided arm's move executes in the
                # frame of the view the model reported as its guide (WRIST -> wrist,
                # FRONT -> base). Tool off / no decision / no view -> no override, the
                # controller's configured frame applies (today's behavior).
                views: dict[str, Optional[str]] = {}
                frames: dict[str, Optional[str]] = {}
                if getattr(self.view_select_plugin, "enabled", False) and decision is not None:
                    for side in SIDES:
                        if side in overridden:
                            continue
                        views[side] = decision.views.get(side)
                        frames[side] = self.view_select_plugin.frame_for(views[side])
                # Remember this step's guiding views so the NEXT step's affordance
                # premark knows which arms are wrist-guided (see _premark_affordance).
                self._prev_views = dict(views)

                # Execute BOTH arms simultaneously (Z floors enforced inside each
                # controller). STILL executes nothing: no re-command, the arm holds.
                results = self._execute(tokens, frames)

                # Per-arm bookkeeping (mirrors the single-arm runner, per side).
                records: dict[str, dict[str, Any]] = {}
                stage_done: dict[str, bool] = {}
                recoveries: dict[str, Any] = {}
                for side in SIDES:
                    arm = self.arms[side]
                    token = tokens[side]
                    result = results.get(side)
                    done = bool(getattr(result, "done", False)) or token == DONE_TOKEN
                    recovery = overrides[side]
                    post = self._recovery_after_step(side, token, result, obs, done)
                    if post is not None:
                        recovery = post
                    if recovery is not None and recovery.release:
                        arm.controller.step(RELEASE_TOKEN)
                    if recovery is not None and recovery.block_done:
                        done = False
                    # A close that measurably holds expires any lingering recovery
                    # note BEFORE a fresh this-step note (if any) is applied below.
                    self._expire_grasp_note(arm, token, result)
                    if recovery is not None and recovery.prompt_note:
                        arm.recovery_note = recovery.prompt_note
                    self._note_blocked_reach(arm, result)
                    self._note_descend(arm, token, result)
                    stage_done[side] = done
                    recoveries[side] = recovery
                    self._update_move_memory(arm, token, recovery, done, views.get(side))
                    records[side] = self._arm_record(side, token, result, obs, done, recovery)
                    if views.get(side):
                        records[side]["view"] = views[side]
                    if side in human and side not in overridden:
                        records[side]["src"] = "human"
                    if self.affordance_plugin is not None:
                        dot = self.affordance_plugin.point_of(side)
                        if dot:
                            records[side]["afford_dot"] = dot
                        wrist_dot = self.affordance_plugin.wrist_point_of(side)
                        if wrist_dot:
                            records[side]["afford_wrist"] = wrist_dot

                self._show_live(
                    step_idx,
                    (agentview, wrist_left, wrist_right),
                    "executed",
                    steps_done=step_idx + 1,
                    tokens=tokens,
                    records=records,
                    reason=(
                        decision.reasoning
                        if decision is not None
                        else (
                            "human override (DAGGER)"
                            if human
                            else "recovery override -- no VLM call this step"
                        )
                    ),
                )
                self._print_step(step_idx, tokens, decision, recoveries, records)
                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=[wrist_left, wrist_right],
                    record=self._record(step_idx, tokens, records, decision, obs),
                )

                for side in SIDES:
                    self._advance_arm(side, stage_done[side], recoveries[side], step_idx)

                # All-finished handling happens at the TOP of the next iteration (the
                # final visual check needs a fresh, retreated observation anyway).

                elapsed = time.monotonic() - loop_started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
            else:
                # Step budget exhausted. If the tracks DID finish on the very last
                # step, still run the final check (no budget for a replan) so a
                # genuinely complete episode is not reported as max_steps_exceeded.
                if all(arm.finished for arm in self.arms.values()):
                    outcome, reason = self._completion_outcome(
                        self.max_steps, allow_replan=False
                    )
                    if outcome == "success":
                        success = True
                        end_reason = "plan_complete"
                    else:
                        end_reason = reason
        except KeyboardInterrupt:
            end_reason = "interrupted"
            print(
                "\n[run-dual] Ctrl+C received -- stopping rollout and compiling the "
                "visualization video..."
            )
        finally:
            video_path = self.logger.close(
                success=success, fps=video_fps(self.config.video_fps)
            )
            self.logger.write_summary(
                {
                    "success": success,
                    "steps": steps,
                    "max_steps": self.max_steps,
                    "end_reason": end_reason,
                    "video_path": str(video_path),
                    "run_dir": str(self.logger.run_dir),
                    "control_mode": "real_dual",
                    "task": self.task,
                    "z_floor_m": {
                        side: self.arms[side].controller.z_floor_m for side in SIDES
                    },
                    "raw_plan": raw_plan,
                    "subgoals": {
                        side: [sg.to_prompt_dict() for sg in self.arms[side].subgoals]
                        for side in SIDES
                    },
                    # Every affordance grounding of the episode (metadata.json is
                    # written before the run, so its copy is always empty).
                    **(
                        {"affordance": self.affordance_plugin.metadata()}
                        if getattr(self.affordance_plugin, "enabled", False)
                        else {}
                    ),
                }
            )

        return EpisodeResult(
            success=success,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    # -- execution -----------------------------------------------------------
    def _execute(
        self, tokens: dict[str, str], frames: Optional[dict[str, Optional[str]]] = None
    ) -> dict[str, Any]:
        """Run both arms' tokens SIMULTANEOUSLY (one thread per acting arm).

        STILL -> no controller call at all (the arm holds its setpoint). ``frames``
        carries the view-select per-arm motion-frame override (absent/None -> the
        controller's configured frame). Exceptions are joined-then-reraised so one
        arm's fault never silently strands the other mid-motion (same policy as
        :func:`core.piper.poses.go_begin_dual`).
        """
        results: dict[str, Any] = {}
        errors: dict[str, BaseException] = {}
        frames = frames or {}

        def _run(side: str, token: str) -> None:
            try:
                results[side] = self.arms[side].controller.step(
                    token, motion_frame=frames.get(side)
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised after join
                errors[side] = exc

        acting = {
            side: token
            for side, token in tokens.items()
            if token and token != STILL_TOKEN
        }
        if len(acting) == 1:
            side, token = next(iter(acting.items()))
            results[side] = self.arms[side].controller.step(
                token, motion_frame=frames.get(side)
            )
            return results
        threads = [
            threading.Thread(target=_run, args=(side, token), name=f"arm-{side}")
            for side, token in acting.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            side, exc = next(iter(errors.items()))
            raise RuntimeError(f"{side} arm step failed: {exc}") from exc
        return results

    # -- DAGGER: preemptible VLM decisions -------------------------------------------
    def _decide_interruptible(self, decide_kwargs: dict[str, Any]) -> Optional[Any]:
        """One controller ``decide``, preempted by human DAGGER keys.

        Delegates to :class:`core.runners.preemption.InterruptibleDecider`; ``None`` means
        the human owns this step (the runner executes the human intent instead)."""
        if self._decider is None:
            self._decider = InterruptibleDecider(
                self.dagger_plugin, lambda **kw: self.agent.decide(**kw)
            )
        return self._decider.decide(decide_kwargs)

    # -- retire a finished arm -----------------------------------------------------
    def _retire_arm(self, side: str) -> None:
        """Return a just-finished arm to its BEGIN pose, clearing the shared workspace.

        An arm that reached the end of its track otherwise freezes wherever it stopped --
        typically right over the destination it just used -- where it blocks the other
        arm's approach and occludes the front view both arms reason from. Homing it
        parks it at the known start pose, out of the way.

        No-op when BOTH arms are done (the episode is ending; the caller homes both
        together) or when no ``home_arm`` callback was provided. The gripper is left as
        it is: a finished arm has already released, and force-opening one that (against
        expectation) still holds something would drop it mid-air.
        """
        arm = self.arms[side]
        if self.home_arm is None or not arm.finished or arm.homed:
            return
        if all(a.finished for a in self.arms.values()):
            return
        print(
            _c(_DIM, f"  {side.upper()} finished -- returning to BEGIN to clear the workspace")
        )
        try:
            self.home_arm(side)
        except Exception as exc:  # noqa: BLE001 - homing must not abort the other arm's run
            print(_c(_YELLOW, f"  {side.upper()} go-begin failed: {exc}"))
            return
        arm.homed = True
        # The joint move invalidates the controller's cached Cartesian setpoint. Re-sync
        # so that if a replan hands this arm new work, it resumes from where it actually
        # is instead of lunging back to the stale pre-homing setpoint.
        arm.controller.sync_from_robot()

    # -- completion: final visual check + bounded replan --------------------------
    def _completion_outcome(
        self, step_idx: int, allow_replan: bool = True
    ) -> tuple[str, str]:
        """Both tracks are finished: decide ``success`` / ``replanned`` / ``failed``.

        Per-stage DONE is judged while a gripper often occludes the drop point, so
        "plan complete" can hide a missed placement (seen on hardware: banana released
        at the plate rim, ended beside it). This re-judges the WHOLE task from a fresh
        observation; if incomplete and the ``v0.max_replans`` budget allows, the dual
        planner replans the REMAINING work from the live scene and stepping continues.
        Reasoning stays in the VLM -- the runner only routes the outcome.
        """
        failed = {
            side: arm.failed_reason
            for side, arm in self.arms.items()
            if arm.failed_reason
        }
        if failed:
            return "failed", "; ".join(f"{s}: {r}" for s, r in failed.items())
        obs = self.session.get_observation()
        agentview, wrist_left, wrist_right = self._images(obs)
        complete, reason = self.agent.verify_task(
            self.task, agentview, wrist_left, wrist_right, debug=self.debug
        )
        verdict = _c(_GREEN, "COMPLETE") if complete else _c(_YELLOW, "NOT complete")
        print(f"\n{_c(_DIM, '─── final check ' + '─' * 47)}")
        print(f"  {verdict}")
        for line in _wrap_reason(reason):
            print(f"  {_c(_DIM, '·')} {line}")
        if complete:
            return "success", ""
        if not allow_replan or self._replans >= self.config.max_replans:
            return "failed", f"final_check_failed: {reason}"
        self._replans += 1
        print(
            _c(
                _DIM,
                f"  replanning the remaining work from the live scene "
                f"(replan {self._replans}/{self.config.max_replans}) ...",
            )
        )
        try:
            tracks, raw_plan = self.planner.plan(
                self.task, agentview, wrist_left=wrist_left, wrist_right=wrist_right,
                debug=self.debug,
            )
        except RuntimeError as exc:
            return "failed", f"final_check_failed: {reason} (replan failed: {exc})"
        for side in SIDES:
            arm = self.arms[side]
            arm.subgoals = tracks.get(side, [])
            arm.index = 0
            arm.stage_start_step = step_idx + 1
            arm.recent_moves.clear()
            arm.current_direction = None
            arm.recovery_note = ""
            arm.finished = not arm.subgoals
            arm.failed_reason = ""
            # Fresh work -> this arm can be retired again once it finishes the new track.
            arm.homed = False
        self.logger.write_plan(
            {
                "task": self.task,
                "replan": self._replans,
                "replan_reason": reason,
                "raw_plan": raw_plan,
                "subgoals": {
                    side: [sg.to_prompt_dict() for sg in self.arms[side].subgoals]
                    for side in SIDES
                },
            }
        )
        self._print_plan()
        return "replanned", ""

    # -- per-arm bookkeeping ---------------------------------------------------
    def _apply_stage_cap(self, arm: _ArmState, step_idx: int) -> None:
        if arm.finished or arm.subgoal is None:
            return
        if step_idx - arm.stage_start_step < self.config.max_subgoal_steps:
            return
        arm.index += 1
        arm.stage_start_step = step_idx
        arm.current_direction = None
        arm.recent_moves.clear()
        if arm.index >= len(arm.subgoals):
            arm.finished = True
            arm.failed_reason = "subgoal_step_cap_exceeded"

    def _advance_arm(
        self, side: str, stage_done: bool, recovery: Any, step_idx: int
    ) -> None:
        arm = self.arms[side]
        if arm.finished:
            return
        if recovery is not None and getattr(recovery, "rollback_index", None) is not None:
            arm.index = int(recovery.rollback_index)
            arm.stage_start_step = step_idx + 1
            arm.current_direction = None
            arm.recent_moves.clear()
            if getattr(recovery, "grasp_empty", False):
                arm.recent_moves.insert(0, EMPTY_GRASP_LABEL)
            return
        if stage_done:
            if self._is_grasp_stage(arm.subgoal):
                arm.recovery_note = ""
            arm.index += 1
            arm.stage_start_step = step_idx + 1
            arm.current_direction = None
            arm.recent_moves.clear()
            if arm.index >= len(arm.subgoals):
                arm.finished = True

    def _update_move_memory(
        self,
        arm: _ArmState,
        token: str,
        recovery: Any,
        stage_done: bool,
        view: Optional[str] = None,
    ) -> None:
        if recovery is not None and getattr(recovery, "reset_history", False):
            arm.current_direction = None
            arm.recent_moves.clear()
        elif token in MOVE_ATOMS:
            # Under view select the remembered move carries its guiding view
            # ("MV_FWD@WRIST"): the stateless per-step controller can only detect a
            # futile same-view pattern (kept pushing on a view whose goal is not
            # visible) from this line. current_direction stays the BARE token -- it
            # doubles as the degraded-output fallback and must remain a valid action.
            arm.recent_moves.insert(0, f"{token}@{view}" if view else token)
            del arm.recent_moves[self.recent_moves_max:]
            arm.current_direction = token
        elif token == GRASP_TOKEN:
            arm.recent_moves.insert(0, GRASP_TOKEN)
            del arm.recent_moves[self.recent_moves_max:]
            arm.current_direction = None
        elif token in (RELEASE_TOKEN, DONE_TOKEN) or stage_done:
            arm.current_direction = None
        # STILL: no memory update -- waiting is not a move.

    # -- recovery ---------------------------------------------------------------
    def _recovery_before_decision(self, side: str, obs: dict[str, Any]):
        arm = self.arms[side]
        if arm.recovery_plugin is None or arm.finished:
            return None
        return arm.recovery_plugin.before_decision(
            current_index=arm.index,
            subgoals=arm.subgoals,
            measured_width_m=self._measured_width(side, obs),
            gripper_closed=bool(arm.controller.gripper_closed),
        )

    def _recovery_after_step(
        self, side: str, token: str, result: Any, obs: dict[str, Any], stage_done: bool
    ):
        arm = self.arms[side]
        if arm.recovery_plugin is None or result is None:
            return None
        return arm.recovery_plugin.after_step(
            token=token,
            result=result,
            current_index=arm.index,
            subgoals=arm.subgoals,
            measured_width_m=self._measured_width(side, obs),
            subgoal_done=stage_done,
            gripper_closed=bool(result.gripper_closed),
        )

    def _recovery_prompt(self, side: str) -> str:
        arm = self.arms[side]
        if arm.recovery_plugin is None:
            return ""
        return arm.recovery_plugin.render_prompt_context(arm.recovery_note)

    # The prompt hint injected after a move died at the arm's reach boundary. Without
    # it the VLM has no way to know its command moved nothing -- the recent-moves line
    # claims the move happened -- and it repeats the dead direction forever (observed
    # on hardware: 3 pinned MV_FWDs chasing a plate that was already behind the arm).
    _REACH_NOTE = (
        "The last move hit this arm's REACH LIMIT and moved little or not at all. That "
        "direction is exhausted: the goal cannot be further that way -- re-judge, it is "
        "likely beside or behind the gripper."
    )

    def _expire_grasp_note(self, arm: _ArmState, token: str, result: Any) -> None:
        """Drop a lingering recovery note once a GRASP measurably HOLDS.

        The note describes a PAST event, but it stays until the stage completes --
        observed on hardware (rollout 20-16-46): a step-18 "Empty close" note was
        still in the prompt at step 28, where it primed the model to call a real
        15 mm hold "closed empty" and RELEASE the object. A non-empty close (the
        controller verifies settled width against the empty band) makes any prior
        grasp/reach note stale; a fresh note from THIS step is applied after this.
        """
        if (
            token == GRASP_TOKEN
            and result is not None
            and bool(getattr(result, "gripper_closed", False))
            and not bool(getattr(result, "grasp_empty", False))
        ):
            arm.recovery_note = ""

    @staticmethod
    def _note_descend(arm: _ArmState, token: str, result: Any) -> None:
        """Record how much of a commanded MV_DOWN this arm actually travelled, for the
        NEXT prompt (the proprioception tool reports a descent that stalled against
        something). Same measurement as the single-arm runner, kept per arm."""
        arm.descend = descend_travel(token, result, arm.descend)

    def _note_blocked_reach(self, arm: _ArmState, result: Any) -> None:
        """Maintain the reach-limit prompt hint from the controller's step note.

        Set when a move ends in a reach clamp (partial motion at the workspace
        boundary) or the 0-streamed fallback (arm already pinned); cleared by the next
        move that executes normally, so a stale warning never lingers after the arm
        works its way free. Recovery-tool notes (grasp events) take precedence and are
        not overwritten."""
        if result is None:
            return
        note = str(getattr(result, "note", "") or "")
        if "reach clamp" in note or "reach fallback" in note:
            arm.recovery_note = self._REACH_NOTE
        elif arm.recovery_note == self._REACH_NOTE and getattr(result, "kind", "") == "move":
            arm.recovery_note = ""

    # -- observation helpers ------------------------------------------------------
    def _images(self, obs: dict[str, Any]):
        return (
            to_uint8_hwc(obs["agentview"]),
            to_uint8_hwc(obs["wrist_left"]),
            to_uint8_hwc(obs["wrist_right"]),
        )

    def _premark_affordance(self, agentview, wrist_left, wrist_right):
        """Affordance dots on ALL THREE images. Per arm: ground the front dot once on
        stage entry (the 'which part' anchor), then -- while that arm is wrist-guided
        (its PREVIOUS step's guiding view was WRIST) -- re-locate that part on THIS
        step's wrist frame and draw it there too. Every consumer (controller, live
        view, video) sees the annotated frames. Tool off/absent -> frames pass through.

        Wrist tracking keys on the view-select report, so it engages only when
        plugins.view_select is on (both are mode-B plugins); without it the tool stays
        front-only, exactly as before."""
        tool = self.affordance_plugin
        if tool is None or not getattr(tool, "enabled", False):
            return agentview, wrist_left, wrist_right
        wrists = {"left": wrist_left, "right": wrist_right}
        for side in SIDES:
            arm = self.arms[side]
            subgoal = arm.subgoal
            tool.ensure(
                slot=side,
                stage_key=_affordance_stage_key(arm),
                task=self.task,
                subgoal=subgoal.to_prompt_dict() if subgoal is not None else None,
                agentview=agentview,
                debug=self.debug,
            )
            tool.update_wrist(
                slot=side,
                wrist_image=wrists[side],
                active=self._prev_views.get(side) == "WRIST",
                task=self.task,
                debug=self.debug,
            )
        return (
            tool.annotate(agentview),
            tool.annotate_wrist("left", wrist_left),
            tool.annotate_wrist("right", wrist_right),
        )

    def _proprio(self, side: str, obs: dict[str, Any]) -> dict[str, Any]:
        ee_pose = np.asarray(obs[side].get("ee_pose", []), dtype=float).reshape(-1)
        arm = self.arms[side]
        proprio: dict[str, Any] = {
            "eef_pos": ee_pose[:3].tolist() if ee_pose.size >= 3 else [],
            "gripper_width": self._measured_width(side, obs),
            "gripper_command_name": "CLOSED" if arm.controller.gripper_closed else "OPEN",
        }
        if arm.descend is not None:
            proprio["descend_moved_m"], proprio["descend_commanded_m"] = arm.descend
        return proprio

    @staticmethod
    def _measured_width(side: str, obs: dict[str, Any]) -> float:
        return float(obs[side].get("gripper_width", 0.0))

    def _observed_gripper_state(self, side: str, obs: dict[str, Any]) -> str:
        """OPEN/CLOSED as the CAMERA sees it, from the measured width (see the
        single-arm runner's rationale: the commanded state leads the fingers)."""
        thr = float(
            getattr(self.arms[side].controller, "gripper_close_threshold_m", 0.07)
        )
        return "CLOSED" if self._measured_width(side, obs) < thr else "OPEN"

    @staticmethod
    def _recent_moves_text(recent_moves: list[str]) -> str:
        return ", ".join(recent_moves) if recent_moves else "none"

    @staticmethod
    def _is_grasp_stage(subgoal: Optional[Subgoal]) -> bool:
        return str(getattr(subgoal, "motion", "") or "").upper() == GRASP_TOKEN

    def _viewer_arm(
        self, side: str, token: Optional[str] = None, rec: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """The live window's per-arm status payload (STAGE label, token, grip, flags)."""
        arm = self.arms[side]
        if arm.subgoal is None:
            stage = "FINISHED"
        else:
            stage = f"STAGE {arm.index + 1}/{len(arm.subgoals)} · {arm.subgoal.motion}"
        payload: dict[str, Any] = {
            "stage": stage,
            "grip": "CLOSED" if arm.controller.gripper_closed else "OPEN",
        }
        if token is not None:
            payload["token"] = token
        if rec is not None:
            payload["flags"] = status_flags(rec)
        return payload

    def _telemetry_text(self, steps_done: int) -> str:
        elapsed = time.monotonic() - self._t0
        parts = [f"total {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"]
        if steps_done > 0:
            parts.append(f"avg {elapsed / steps_done:.1f} s/step")
        if self._last_vlm_ms is not None:
            parts.append(f"VLM {self._last_vlm_ms} ms")
        return "  ·  ".join(parts)

    def _show_live(
        self,
        step_idx: int,
        images: tuple[np.ndarray, np.ndarray, np.ndarray],
        phase: str,
        steps_done: int,
        tokens: Optional[dict[str, str]] = None,
        records: Optional[dict[str, dict[str, Any]]] = None,
        reason: str = "",
    ) -> None:
        if self.viewer is None:
            return
        agentview, wrist_left, wrist_right = images
        self.viewer.show_dual(
            step=step_idx,
            agentview=agentview,
            wrist_left=wrist_left,
            wrist_right=wrist_right,
            task=self.task,
            phase=phase,
            telemetry=self._telemetry_text(steps_done),
            arms={
                side: self._viewer_arm(
                    side,
                    tokens.get(side) if tokens else None,
                    records.get(side) if records else None,
                )
                for side in SIDES
            },
            reason=reason,
        )

    def _print_plan(self) -> None:
        """Both arms' stage tracks, side by side (also printed on a replan)."""
        print(f"\n{_c(_DIM, '  plan')}")
        for side in SIDES:
            arm = self.arms[side]
            head = _c(_SIDE_COLOR[side], f"  {side.upper():<5}")
            if not arm.subgoals:
                print(f"{head} {_c(_DIM, '(no work for this arm)')}")
                continue
            n = len(arm.subgoals)
            print(f"{head} {_c(_DIM, f'{n} stage' + ('s' if n != 1 else ''))}")
            for i, sg in enumerate(arm.subgoals):
                print(
                    f"        STAGE {i + 1}  {sg.motion:<8} "
                    f"{_c(_DIM, _short(sg.completion, 68))}"
                )
        print()

    def _print_step(
        self,
        step_idx: int,
        tokens: dict[str, str],
        decision: Any,
        recoveries: dict[str, Any],
        records: dict[str, dict[str, Any]],
    ) -> None:
        """One readable block per step: a rule, one line per arm, then the reasoning.

        Deliberately NOT a coordinate dump -- the operator watches the arms and the
        camera window; what the terminal is for is WHY the model did what it did, and
        whether anything blocked it. Per-token controller chatter is silenced in this
        mode (see run_real_dual), so this is the whole per-step story.
        """
        print(_c(_DIM, f"\n─── step {step_idx:03d} " + "─" * 52))
        for side in SIDES:
            arm = self.arms[side]
            rec = records[side]
            stage = (
                f"STAGE {arm.index + 1}/{len(arm.subgoals)} {rec['stage']}"
                if arm.subgoal is not None
                else "done"
            )
            token = tokens[side]
            tag = _c(_SIDE_COLOR[side], f"  {side[0].upper()}")
            act = token if token == STILL_TOKEN else _c(_BOLD, token)
            flags = []
            if rec.get("done"):
                flags.append(_c(_GREEN, "stage done"))
            if rec.get("grasp_fail"):
                flags.append(_c(_RED, "empty close -> reopened"))
            blocked = rec.get("blocked")
            if blocked == "z_floor":
                flags.append(_c(_YELLOW, "blocked: z-floor"))
            elif blocked == "reach":
                flags.append(_c(_YELLOW, "blocked: reach limit"))
            event = getattr(recoveries.get(side), "event", None)
            if event:
                flags.append(_c(_YELLOW, f"recovery: {event}"))
            # DAGGER: this arm's token came from the operator, not the model.
            if rec.get("src") == "human":
                flags.append(_c(_YELLOW, "human"))
            # View-select: which view guided this arm's move (and so its motion frame).
            view = rec.get("view")
            if view and token in MOVE_ATOMS:
                flags.append(_c(_DIM, f"{str(view).lower()} guided"))
            suffix = ("   " + " · ".join(flags)) if flags else ""
            print(f"{tag}  {stage:<20} {act:<20} {_c(_DIM, 'grip ' + rec['grip'])}{suffix}")
        if decision is None:
            if any(records[s].get("src") == "human" for s in SIDES):
                print(_c(_DIM, "  · human override (DAGGER) -- VLM decision skipped or dropped"))
            else:
                print(_c(_DIM, "  · recovery override -- no VLM call this step"))
            return
        for line in _wrap_reason(decision.reasoning):
            print(f"  {_c(_DIM, '·')} {line}")

    # -- logging -------------------------------------------------------------------
    def _arm_record(
        self,
        side: str,
        token: str,
        result: Any,
        obs: dict[str, Any],
        stage_done: bool,
        recovery: Any,
    ) -> dict[str, Any]:
        arm = self.arms[side]
        if result is not None and getattr(result, "post_pose", None) is not None:
            eef = np.asarray(result.post_pose, dtype=float).reshape(-1)[:3]
        else:
            eef = np.asarray(obs[side].get("ee_pose", []), dtype=float).reshape(-1)[:3]
        record: dict[str, Any] = {
            "sg": int(arm.index),
            "n_sg": len(arm.subgoals),
            "stage": arm.subgoal.motion if arm.subgoal is not None else "FINISHED",
            "sid": arm.subgoal.id if arm.subgoal is not None else "-",
            "act": token,
            "eef": [round(float(x), 3) for x in eef],
            "w": round(self._measured_width(side, obs), 5),
            "grip": "CLOSED" if arm.controller.gripper_closed else "OPEN",
        }
        note = getattr(result, "note", None) if result is not None else None
        if note and "z-floor" in note:
            record["blocked"] = "z_floor"
        elif note and ("reach clamp" in note or "reach fallback" in note):
            record["blocked"] = "reach"
        if bool(getattr(result, "grasp_empty", False)) or bool(
            getattr(recovery, "grasp_empty", False)
        ):
            record["grasp_fail"] = note or True
        if recovery is not None:
            record["recover"] = {
                key: value
                for key in ("event", "reason", "rollback_index", "token", "release")
                if (value := getattr(recovery, key, None)) not in (None, "", False)
            }
        if stage_done:
            record["done"] = True
        return record

    def _record(
        self,
        step_idx: int,
        tokens: dict[str, str],
        arm_records: dict[str, dict[str, Any]],
        decision: Any,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """One combined step record: analysis-frame-compatible top-level fields plus
        the full per-arm sub-records."""
        record: dict[str, Any] = {
            "i": int(step_idx),
            "sg": "|".join(f"{s[0].upper()}{arm_records[s]['sg']}" for s in SIDES),
            "stage": "|".join(
                f"{s[0].upper()}:{arm_records[s]['stage']}" for s in SIDES
            ),
            "act": "|".join(f"{s[0].upper()}:{tokens[s]}" for s in SIDES),
            "grip": "|".join(f"{s[0].upper()}:{arm_records[s]['grip']}" for s in SIDES),
            "w": self._measured_width("left", obs),
            "left": arm_records["left"],
            "right": arm_records["right"],
        }
        elapsed = time.monotonic() - self._t0
        record["t_s"] = round(elapsed, 1)
        record["avg_s"] = round(elapsed / max(1, step_idx + 1), 2)
        if decision is not None:
            latency_s = (decision.payload or {}).get("latency_s")
            c_entry: dict[str, Any] = {}
            if latency_s is not None:
                c_entry["ms"] = int(round(float(latency_s) * 1000.0))
                record["vlm_ms"] = c_entry["ms"]
            if decision.reasoning:
                c_entry["why"] = decision.reasoning
            if c_entry:
                record["vlm"] = {"c": c_entry}
        if self._replans:
            record["replan"] = self._replans
        human_sides = [s for s in SIDES if arm_records[s].get("src") == "human"]
        if human_sides:
            record["dagger"] = "|".join(s[0].upper() for s in human_sides)
        if all(arm_records[s].get("done") for s in SIDES):
            record["done"] = True
        return record


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _affordance_stage_key(arm: _ArmState) -> str:
    """Identity of the arm's CURRENT stage for affordance grounding. Content-based
    (index + the fields the pointer consumes), so a replaced plan whose same-index
    stage differs re-grounds while an unchanged stage never re-pays the VLM call."""
    subgoal = arm.subgoal
    if subgoal is None:
        return ""
    return f"{arm.index}:{subgoal.id}:{subgoal.target}:{subgoal.affordance}"


# -- console formatting ------------------------------------------------------------
# Shared ANSI styling (core.ui.console): same palette/behavior as the single-arm path.
_DIM, _BOLD = console.DIM, console.BOLD
_GREEN, _YELLOW, _RED = console.GREEN, console.YELLOW, console.RED
_SIDE_COLOR = console.SIDE_COLOR
_c = console.c

# The trailing machine-readable ``FINAL: LEFT=... RIGHT=...`` line is stripped: the
# tokens are already shown above the reasoning.
_FINAL_LINE = r"\s*FINAL:\s*LEFT\s*=\s*\S+\s+RIGHT\s*=\s*\S+\s*$"


def _wrap_reason(reasoning: Any) -> list[str]:
    return console.wrap_reason(reasoning, strip=_FINAL_LINE)
