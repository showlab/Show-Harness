"""Planner-free / stage-free closed-loop rollout for the MVTOKEN_v0 Qwen LoRA.

Real-robot counterpart of :class:`core.runners.real.RealEpisodeRunner`, stripped down for
the MVTOKEN_v0 policy: there is NO subgoal planner, NO stage, and NO ``DONE`` token. Each
step feeds the model the task, the current gripper state, and the recent ``MV_*`` moves
(NO stage line), executes the single atomic token it returns on the physical Franka (with
the controller's Z-floor safety still enforced), logs the frame, and repeats. Because there
is no completion signal, the episode always ends on ``max_steps`` (or Ctrl+C); the video /
logs use the same on-disk layout as the subgoal-pipeline runners.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

import core.ui.console as console
from core.action_units import MOVE_ATOMS
from interpreters.franka_atomic_controller import AtomicStepResult, FrankaAtomicController
from core.runners.preemption import InterruptibleDecider
from core.record.episode_logger import EpisodeLogger, status_flags
from core.franka.franka_session import FrankaSession
from core.record.images import to_uint8_hwc
from core.v0_types import EpisodeResult
from plugins.auto_release import AutoReleasePlugin

GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
DONE_TOKEN = "DONE"
FALLBACK_TOKEN = "MV_DOWN"
# STILL is NOT in this policy's vocabulary (the single-arm LoRAs were trained on 9 tokens:
# MV_* / GRASP / RELEASE / DONE). It exists here only as the DAGGER hold -- the operator
# pressing "keep this arm where it is" -- and executes as no controller call at all.
STILL_TOKEN = "STILL"
# The DAGGER plugin's single-arm intent slot (plugins.dagger.SINGLE_SIDE).
DAGGER_SIDE = "arm"
# Match the MVTOKEN training window (rollout_to_llamafactory.py RECENT_WINDOW = 5): the
# input lists up to the 5 most recent MV_* moves, newest first; GRASP/RELEASE are excluded.
RECENT_MOVES_MAX = 5


class MvTokenRunner:
    """Closed-loop MVTOKEN_v0 rollout on the physical Franka (no planner, no stage)."""

    def __init__(
        self,
        *,
        session: FrankaSession,
        controller: FrankaAtomicController,
        agent: Any,
        logger: EpisodeLogger,
        task: str,
        gripper_color: str,
        max_steps: int,
        loop_period_s: float,
        use_wrist_image: bool,
        video_fps: float,
        debug: bool,
        viewer: Any = None,
        auto_release: Optional[AutoReleasePlugin] = None,
        prompt_log_every: int = 20,
        dagger_plugin: Any = None,
    ) -> None:
        self.session = session
        self.controller = controller
        # DAGGER: real-time human keyboard override (plugins.dagger). Keys arrive on the live
        # view's stream thread, so they land even while the VLM call is in flight (see
        # _decide_interruptible). None/disabled -> the loop is byte-identical to before.
        self.dagger_plugin = dagger_plugin
        # The single in-flight decide() call, if one was abandoned mid-step.
        self._decider: Optional[InterruptibleDecider] = None
        self.agent = agent
        self.logger = logger
        self.task = str(task)
        self.gripper_color = str(gripper_color)
        self.max_steps = int(max_steps)
        self.loop_period_s = float(loop_period_s)
        self.use_wrist_image = bool(use_wrist_image)
        self.video_fps = float(video_fps)
        self.debug = bool(debug)
        self.viewer = viewer
        # Auto-release safety rule (plugins.auto_release): consulted after each step to
        # reopen a closed gripper that has collapsed below the empty width. None or a
        # disabled tool -> no-op, so the loop is byte-identical to having no rule.
        self.auto_release = auto_release
        # How many PAST frames ride along with the current one (0 = classic single-frame
        # request). 2 -> the model sees t-2, t-1, t = 6 images (agentview + wrist each).
        # The served LoRA must have been trained on the same shape; see MvTokenController.
        # Dump the exact prompt sent to the VLM every N steps into controller_prompts/
        # (0 disables).
        self.prompt_log_every = int(prompt_log_every)
        # Wall-clock + last VLM latency for the terminal blocks / live-window telemetry.
        self._t0 = time.monotonic()
        self._last_vlm_ms: Optional[int] = None

    # -- main loop ---------------------------------------------------------
    def run(self) -> EpisodeResult:
        steps = 0
        end_reason = "max_steps_reached"
        recent_moves: list[str] = []
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"
        self._t0 = time.monotonic()
        self._last_vlm_ms = None

        # Start from a known, open-gripper state. The controller is already synced (and the
        # Z floor locked) by the caller before the rollout begins.
        self.controller.step(RELEASE_TOKEN)

        try:
            for step_idx in range(self.max_steps):
                steps = step_idx + 1
                loop_started = time.monotonic()
                # Wall-clock step start: ts[i+1]-ts[i] in steps.jsonl is the true
                # action-to-action period (scripts/trajectory/step_timing.py).
                step_ts = time.time()

                t_mark = time.monotonic()
                obs = self.session.get_observation()
                t_obs_ms = (time.monotonic() - t_mark) * 1000.0
                agentview, wrist = self._images(obs)
                self._show_live(step_idx, agentview, wrist, "deciding ...", steps_done=step_idx)

                gripper_state = self._gripper_state()
                fallback = False
                human = False
                decide_kwargs = dict(
                    task=self.task,
                    gripper_state=gripper_state,
                    recent_moves=self._recent_moves_text(recent_moves),
                    agentview_image=agentview,
                    wrist_image=wrist,
                    debug=self.debug,
                )
                t_mark = time.monotonic()
                try:
                    if self._dagger_enabled():
                        # DAGGER: the VLM call is preemptible -- the moment human keys arrive
                        # it is abandoned (and its result later dropped); None means the human
                        # owns this step.
                        response: Any = self._decide_interruptible(decide_kwargs)
                    else:
                        response = self.agent.decide(**decide_kwargs)
                    if response is not None:
                        token = response.token
                    else:
                        token = self._human_token()
                        human = True
                except RuntimeError as exc:
                    # Degraded-output safeguard: never crash a live rollout on a bad VLM
                    # reply; hold by descending one step (Z-floor still protects the table).
                    print(
                        console.c(
                            console.YELLOW,
                            f"  [mvtoken] step {step_idx}: VLM token parse failed ({exc}); "
                            f"falling back to {FALLBACK_TOKEN}",
                        )
                    )
                    response = None
                    token = FALLBACK_TOKEN
                    fallback = True
                t_decide_ms = (time.monotonic() - t_mark) * 1000.0

                # Faithfully record the exact rendered prompt sent to the VLM every N steps,
                # beside the step images (the returned token is already in steps.jsonl). A
                # human step reuses the previous decision's prompt text, so it is skipped.
                if response is not None and self.prompt_log_every > 0 and step_idx % self.prompt_log_every == 0:
                    prompt_text = getattr(self.agent, "last_prompt", "")
                    if prompt_text:
                        self.logger.save_controller_prompt(
                            step_idx,
                            prompt_text,
                            media=getattr(self.agent, "last_media", None),
                        )

                # Terminal token: the model reports the task complete -> end the rollout. (On
                # the real robot there is no env success check, so we record end_reason="done"
                # without claiming success.) Printed as its own step block; nothing executes.
                if token == DONE_TOKEN:
                    record = {
                        "grip": "CLOSED" if self.controller.gripper_closed else "OPEN",
                        "ts": round(step_ts, 3),
                        "t_obs_ms": int(round(t_obs_ms)),
                        "t_decide_ms": int(round(t_decide_ms)),
                    }
                    self._capture_latency(response)
                    self._print_step(step_idx, token, record, fallback=fallback, done=True)
                    self._show_live(
                        step_idx, agentview, wrist, "executed",
                        steps_done=step_idx + 1, token=token, record=record,
                    )
                    end_reason = "done"
                    break

                # Execute the token on the real robot (Z floor enforced inside). STILL is a
                # DAGGER-only hold (it is NOT in this policy's vocabulary): it issues no
                # controller call at all, so the arm simply keeps its setpoint -- the same
                # policy core.runners.real and the dual runners apply.
                t_mark = time.monotonic()
                result = None if token == STILL_TOKEN else self.controller.step(token)
                t_exec_ms = (time.monotonic() - t_mark) * 1000.0

                if token in MOVE_ATOMS:
                    recent_moves.insert(0, token)
                    del recent_moves[RECENT_MOVES_MAX:]

                # Auto-release safety rule: a closed gripper whose measured width has
                # collapsed below the empty threshold is holding nothing -- reopen it
                # immediately so the next decision starts from a clean, open gripper.
                auto_released = self._maybe_auto_release(step_idx, result)
                if auto_released is not None:
                    result = auto_released

                record = self._record(
                    step_idx, token, result, response, obs,
                    auto_released=auto_released is not None,
                    human=human,
                )
                # Per-step latency decomposition, for offline analysis of real
                # rollouts (camera read / decision incl. VLM / arm motion).
                record["ts"] = round(step_ts, 3)
                record["t_obs_ms"] = int(round(t_obs_ms))
                record["t_decide_ms"] = int(round(t_decide_ms))
                record["t_exec_ms"] = int(round(t_exec_ms))
                self._print_step(step_idx, token, record, fallback=fallback)
                self._show_live(
                    step_idx, agentview, wrist, "executed",
                    steps_done=step_idx + 1, token=token, record=record,
                )
                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=wrist,
                    record=record,
                )

                elapsed = time.monotonic() - loop_started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
        except KeyboardInterrupt:
            end_reason = "interrupted"
            print(
                "\n[run-real-mvtoken] Ctrl+C received -- stopping rollout and compiling "
                "the visualization video..."
            )
        finally:
            video_path = self.logger.close(success=False, fps=video_fps(self.video_fps))
            self.logger.write_summary(
                {
                    "success": False,
                    "steps": steps,
                    "max_steps": self.max_steps,
                    "end_reason": end_reason,
                    "video_path": str(video_path),
                    "run_dir": str(self.logger.run_dir),
                    "control_mode": "real_mvtoken",
                    "task": self.task,
                    "gripper_color": self.gripper_color,
                    "z_floor_m": self.controller.z_floor_m,
                }
            )

        return EpisodeResult(
            success=False,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    # -- helpers -----------------------------------------------------------
    def _images(self, obs: dict[str, Any]):
        agentview = to_uint8_hwc(obs["agentview"])
        wrist = None
        if self.use_wrist_image and obs.get("wrist") is not None:
            wrist = to_uint8_hwc(obs["wrist"])
        return agentview, wrist

    def _gripper_state(self) -> str:
        """The COMMANDED gripper state, lower-cased (``open``/``closed``).

        This matches the MVTOKEN training label, which used the recorded ``gripper_closed``
        intent. (real_runner reports the MEASURED width instead, to stop the controller
        calling a premature DONE while the async fingers still lag -- but this loop has no
        DONE, and using the commanded state both matches training and stops the model from
        re-grasping while the width sensor catches up to a just-issued close.)
        """
        return "closed" if self.controller.gripper_closed else "open"

    @staticmethod
    def _recent_moves_text(recent_moves: list[str]) -> str:
        return ", ".join(recent_moves) if recent_moves else "none"

    def _maybe_auto_release(
        self, step_idx: int, result: Optional[AtomicStepResult]
    ) -> Optional[AtomicStepResult]:
        """Reopen the gripper if the auto-release rule fires; return the RELEASE result.

        Returns ``None`` when the rule is absent/disabled or the gripper is not empty,
        leaving the just-executed step untouched. Otherwise issues a ``RELEASE`` on the
        robot and returns its result so the caller can log the reopened state.
        """
        if self.auto_release is None or not self.auto_release.enabled:
            return None
        if result is None:
            # A STILL hold (DAGGER) executed nothing, so the gripper state is exactly what
            # the previous step already checked -- nothing new to react to.
            return None
        if not bool(result.gripper_closed):
            return None
        width_m = self.controller.measured_gripper_width()
        if not self.auto_release.should_release(width_m, bool(result.gripper_closed)):
            return None
        print(
            console.c(
                console.YELLOW,
                f"  [mvtoken] step {step_idx}: auto-release -- gripper width "
                f"{width_m:.4f}m < {self.auto_release.empty_width_m:.4f}m; opening gripper",
            )
        )
        return self.controller.step(RELEASE_TOKEN)

    def _record(
        self,
        step_idx: int,
        token: str,
        result: Optional[AtomicStepResult],
        response: Any,
        obs: dict[str, Any],
        auto_released: bool = False,
        human: bool = False,
    ) -> dict[str, Any]:
        """Compact step record compatible with EpisodeLogger's analysis frame.

        ``result`` is None for a STILL hold (DAGGER), which issues no controller call: the
        pose and gripper state then come from the observation / the live controller.
        """
        if result is not None and result.post_pose is not None:
            eef = np.asarray(result.post_pose, dtype=float).reshape(-1)[:3]
        else:
            eef = np.asarray(obs.get("ee_pose", []), dtype=float).reshape(-1)[:3]
        closed = (
            result.gripper_closed if result is not None else self.controller.gripper_closed
        )
        record: dict[str, Any] = {
            "i": int(step_idx),
            "stage": "-",
            "act": token,
            "eef": [round(float(x), 3) for x in eef],
            "w": round(float(obs.get("gripper_width", 0.0)), 5),
            "grip": "CLOSED" if closed else "OPEN",
        }
        # This step was driven by the OPERATOR (DAGGER), not the model. Same field name as
        # the dual runners, so human-driven steps can be filtered out of any analysis.
        if human:
            record["dagger"] = True
        latency_ms = self._capture_latency(response)
        if latency_ms is not None:
            record["vlm_ms"] = latency_ms
        # Which step magnitude the MOVE used ("up" for the dedicated MV_UP distance;
        # "coarse"/"fine" when the variable-step plugin is on); absent for a fixed step.
        step_kind = str(getattr(result, "step_kind", "") or "")
        if step_kind:
            record["step_kind"] = step_kind
            record["step_cm"] = round(float(getattr(result, "step_m", 0.0)) * 100.0, 1)
        if getattr(result, "note", None) and "z-floor" in result.note:
            record["blocked"] = "z_floor"
        if getattr(result, "grasp_empty", False):
            record["grasp_fail"] = result.note or True
        if auto_released:
            record["auto_release"] = True
        return record

    # -- DAGGER: preemptible VLM decisions ----------------------------------------
    def _dagger_enabled(self) -> bool:
        return bool(getattr(self.dagger_plugin, "enabled", False))

    def _human_token(self) -> str:
        """Consume the pending DAGGER intent and resolve it into one atomic token.

        The gripper placeholder resolves into GRASP/RELEASE from the controller's ACTUAL
        state (teleop's flush-time contract), so a toggle always does the opposite of what
        the gripper is currently doing. No pending intent (a race with drain) degrades to a
        STILL hold, which executes nothing.
        """
        intents = self.dagger_plugin.drain()
        token, kind = intents.get(DAGGER_SIDE, (STILL_TOKEN, "still"))
        if kind == "gripper":
            token = RELEASE_TOKEN if self.controller.gripper_closed else GRASP_TOKEN
        return token

    def _decide_interruptible(self, decide_kwargs: dict[str, Any]) -> Optional[Any]:
        """One controller ``decide``, preempted by human DAGGER keys.

        Delegates to :class:`core.runners.preemption.InterruptibleDecider`; ``None`` means
        the human owns this step (the runner executes the human intent instead)."""
        if self._decider is None:
            self._decider = InterruptibleDecider(
                self.dagger_plugin, lambda **kw: self.agent.decide(**kw)
            )
        return self._decider.decide(decide_kwargs)

    def _capture_latency(self, response: Any) -> Optional[int]:
        """Remember (and return) the last VLM latency in ms for telemetry/records."""
        latency_s = (getattr(response, "payload", None) or {}).get("latency_s")
        if latency_s is None:
            return None
        self._last_vlm_ms = int(round(float(latency_s) * 1000.0))
        return self._last_vlm_ms

    # -- console + live-view status ----------------------------------------------
    def _print_step(
        self,
        step_idx: int,
        token: str,
        record: dict[str, Any],
        fallback: bool = False,
        done: bool = False,
    ) -> None:
        """One readable block per step (the subgoal runners' convention). The MVTOKEN
        policy emits a bare token -- no reasoning -- so the block is a single line;
        the flags carry everything that qualified the action."""
        print(console.dim(f"\n─── step {step_idx:03d} " + "─" * 52))
        act = console.c(console.BOLD, token)
        flags = []
        # Step precision first: it qualifies the action itself ("MV_FWD coarse 5 cm").
        if record.get("step_kind"):
            flags.append(console.dim(f"{record['step_kind']} {record['step_cm']:g} cm"))
        if done:
            flags.append(console.c(console.GREEN, "model reports task complete"))
        if record.get("grasp_fail"):
            flags.append(console.c(console.RED, "empty close"))
        if record.get("auto_release"):
            flags.append(console.c(console.YELLOW, "auto-release -> reopened"))
        if record.get("blocked") == "z_floor":
            flags.append(console.c(console.YELLOW, "blocked: z-floor"))
        if fallback:
            flags.append(console.c(console.YELLOW, "fallback (bad VLM reply)"))
        if record.get("vlm_ms") is not None:
            flags.append(console.dim(f"VLM {record['vlm_ms']} ms"))
        suffix = ("   " + " · ".join(flags)) if flags else ""
        tag = console.c(console.ARM_COLOR, "  ARM")
        grip = console.dim("grip " + record.get("grip", "-"))
        print(f"{tag}  {'MVTOKEN':<10} {act:<20} {grip}{suffix}")

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
        agentview,
        wrist,
        phase: str,
        steps_done: int,
        token: Optional[str] = None,
        record: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.viewer is None:
            return
        arm: dict[str, Any] = {
            "stage": "MVTOKEN (stage-free)",
            "grip": "CLOSED" if self.controller.gripper_closed else "OPEN",
        }
        if token is not None:
            arm["token"] = token
        if record is not None:
            arm["flags"] = status_flags(record)
        self.viewer.show_single(
            step=step_idx,
            agentview=agentview,
            wrist=wrist,
            arm=arm,
            task=self.task,
            phase=phase,
            telemetry=self._telemetry_text(steps_done),
        )


def video_fps(video_fps: float) -> float:
    return min(30.0, max(0.5, float(video_fps)))
