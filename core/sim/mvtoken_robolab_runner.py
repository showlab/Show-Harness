"""Planner-free / stage-free RoboLab rollout for the MVTOKEN LoRA.

Simulator sibling of
:class:`core.sim.mvtoken_maniskill_runner.MvTokenManiskillRunner`, driving a RoboLab
(Isaac Lab) Franka + Robotiq 2F-85 through the relative differential-IK action space
instead of MuJoCo OSC / ManiSkill ``pd_ee_delta_pos``. The flat MVTOKEN loop is identical:
each step feeds task + gripper state + recent MV_* moves with the agentview
(``over_shoulder_left_camera``) + wrist (``wrist_cam``) images, maps the returned atomic
token to a 7-dim ``[dx, dy, dz, 0, 0, 0, gripper]`` action via
:class:`~interpreters.robolab_atomic_controller.RobolabAtomicController`, steps the env, and
checks RoboLab's own termination predicate. The episode ends on success, ``DONE``, env
truncation (RoboLab's ``episode_length_s`` time-out) or max_steps.

CROSS-DOMAIN experiment, same caveat as the ManiSkill runs: the LoRA was
trained on REAL Franka/Piper rollouts, so its behaviour in RoboLab's photoreal Isaac Sim
scenes is exploratory. Two things differ more here than they did on ManiSkill and are worth
watching in the video before reading anything into a number:

* **The embodiment is a Robotiq 2F-85, not a Panda hand.** The gripper looks different in
  both views, and it is a binary open/close command rather than a mimic joint.
* **The cameras are 16:9 (1280x720)** where the real rigs are 4:3. See
  ``agentview_crop_aspect`` in ``configs/robot_robolab.yaml``.

Movement tokens are applied as BASE-FRAME atomics (MV_DOWN = lower the gripper), matching
the teleop labels; the axis mapping lives in ``configs/robot_robolab.yaml`` and is
verifiable with ``scripts/run_robolab_mvtoken.py --probe-axes``.
"""
from __future__ import annotations

import time
from typing import Any, Optional


from core.action_units import MOVE_ATOMS
from interpreters.robolab_atomic_controller import RobolabAtomicController
from core.record.episode_logger import EpisodeLogger
from core.record.images import prepare_view
from core.sim.robolab_task import (
    reset_robolab,
    rl_ee_quat,
    rl_gripper_width,
    rl_rgb,
    rl_success,
    rl_tcp,
    step_robolab,
)
from core.v0_types import EpisodeResult, V0Config
from plugins.auto_release import AutoReleasePlugin

GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
DONE_TOKEN = "DONE"
FALLBACK_TOKEN = "MV_DOWN"
RECENT_MOVES_MAX = 5  # match the MVTOKEN training window (MV_* only, newest first)


# The full camera transform contract lives in core.record.images.prepare_view -- shared with the
# ManiSkill runner and with the real2sim generators, so agentview and wrist go through one
# code path and stored frames stay byte-identical to sent frames.


class MvTokenRobolabRunner:
    """Closed-loop MVTOKEN rollout in the RoboLab simulator (no planner, no stage)."""

    def __init__(
        self,
        *,
        env: Any,
        task_description: str,
        controller: RobolabAtomicController,
        agent: Any,
        logger: EpisodeLogger,
        config: V0Config,
        max_steps: int,
        num_steps_wait: int,
        loop_period_s: float,
        sim_steps_per_decision: int,
        settle_steps_per_decision: int,
        agentview_camera: str,
        wrist_camera: str,
        agentview_rotation_degrees: int,
        wrist_rotation_degrees: int,
        agentview_flip: str,
        wrist_flip: str,
        use_wrist_image: bool,
        auto_release: Optional[AutoReleasePlugin] = None,
        debug: bool,
        prompt_log_every: int = 20,
        agentview_square_size: Optional[int] = None,
        agentview_crop_aspect: Optional[float] = None,
        wrist_crop_aspect: Optional[float] = None,
        wrist_square_size: Optional[int] = None,
        gripper_hold_steps: int = 0,
        close_env: bool = False,
    ) -> None:
        self.env = env
        self.task_description = str(task_description)
        self.controller = controller
        self.agent = agent
        self.logger = logger
        self.config = config
        self.max_steps = int(max_steps)
        self.num_steps_wait = int(num_steps_wait)
        self.loop_period_s = float(loop_period_s)
        self.sim_steps_per_decision = max(1, int(sim_steps_per_decision))
        # Extra ZERO-delta steps after a decision's motion steps. The relative-IK
        # controller re-targets "current pose + delta" every control step, so it lags the
        # command; holding lets the arm actually arrive before the next frame is captured.
        # Without it the frame the policy sees is mid-flight and the achieved
        # displacement per decision comes out short of step_m.
        self.settle_steps_per_decision = max(0, int(settle_steps_per_decision))
        # The Robotiq is a BINARY joint command, not a mimic force: a GRASP token has to
        # be held for a few control steps for the fingers to actually travel and load up
        # on the object. 0 -> use sim_steps_per_decision.
        self.gripper_hold_steps = int(gripper_hold_steps)
        self.agentview_camera = str(agentview_camera)
        self.wrist_camera = str(wrist_camera)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_flip = str(agentview_flip or "none")
        self.wrist_flip = str(wrist_flip or "none")
        self.use_wrist_image = bool(use_wrist_image)
        # Auto-release safety rule (plugins.auto_release): consulted after each step to
        # reopen a closed gripper that is holding nothing. None -> rule absent (no-op).
        self.auto_release = auto_release
        self.debug = bool(debug)
        self.prompt_log_every = int(prompt_log_every)
        # Letterbox the agentview to this square size before it is sent (None = raw).
        # TRAINING CONTRACT: the real-robot sets are 256x256 resize_with_pad.
        self.agentview_square_size = (
            int(agentview_square_size) if agentview_square_size else None
        )
        # Centre-crop to this width/height ratio BEFORE the letterbox. RoboLab renders
        # 16:9; the training rigs are 4:3 (1.3333). See core.record.images.center_crop_to_aspect.
        self.agentview_crop_aspect = (
            float(agentview_crop_aspect) if agentview_crop_aspect else None
        )
        self.wrist_crop_aspect = float(wrist_crop_aspect) if wrist_crop_aspect else None
        self.wrist_square_size = int(wrist_square_size) if wrist_square_size else None
        # RoboLab keeps ONE Isaac Sim app per process and constructing an env is slow
        # (tens of seconds), so a multi-episode driver reuses it: closing is the caller's
        # call, not the runner's. (ManiSkill's runner closes because there env
        # construction is cheap and one process == one episode.)
        self.close_env = bool(close_env)

    def run(self) -> EpisodeResult:
        # Frame history (if the agent keeps any) must not leak across episodes.
        reset_history = getattr(self.agent, "reset", None)
        if callable(reset_history):
            reset_history()
        # Cleared before the reset so the hold action does not correct toward the
        # PREVIOUS episode's reference while the arm is being teleported home.
        self.controller.set_orientation_reference(None)
        obs, terminated, truncated = reset_robolab(
            self.env,
            hold_action=self.controller.open_gripper(),
            settle_steps=self.num_steps_wait,
        )
        # The home pose IS the contract's top-down orientation; latch it as the reference
        # the whole episode is held to. Identical to what the generator does at reset, so
        # deployment reproduces the geometry the data was recorded under.
        self.controller.set_orientation_reference(rl_ee_quat(self.env))
        success = bool(terminated) or rl_success(self.env)
        end_reason = "max_steps_exceeded"
        steps = 0
        recent_moves: list[str] = []
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"

        try:
            for step_idx in range(self.max_steps):
                steps = step_idx + 1
                loop_started = time.monotonic()

                agentview, wrist = self._images(obs)
                gripper_state = (
                    "closed" if self.controller.state.gripper_name == "CLOSE" else "open"
                )
                try:
                    response: Any = self.agent.decide(
                        task=self.task_description,
                        gripper_state=gripper_state,
                        recent_moves=self._recent_moves_text(recent_moves),
                        agentview_image=agentview,
                        wrist_image=wrist,
                        debug=self.debug,
                    )
                    token = response.token
                except RuntimeError as exc:
                    print(
                        f"[mvtoken-robolab] step {step_idx}: VLM token parse failed ({exc}); "
                        f"falling back to {FALLBACK_TOKEN}"
                    )
                    response = None
                    token = FALLBACK_TOKEN

                if self.prompt_log_every > 0 and step_idx % self.prompt_log_every == 0:
                    prompt_text = getattr(self.agent, "last_prompt", "")
                    if prompt_text:
                        self.logger.save_controller_prompt(
                            step_idx,
                            prompt_text,
                            media=getattr(self.agent, "last_media", None),
                        )

                if token == DONE_TOKEN:
                    print(f"[mvtoken-robolab] step {step_idx}: DONE emitted -- ending rollout.")
                    end_reason = "done"
                    break

                obs, terminated, truncated = self._execute(token, obs)
                success = bool(terminated) or rl_success(self.env)

                # Reflex, not policy: if that step left the gripper closed on nothing,
                # reopen it now so the next decision sees an open gripper.
                released = self._maybe_auto_release(step_idx)
                if released is not None:
                    obs, term2, trunc2 = released
                    terminated = terminated or term2
                    truncated = truncated or trunc2
                    success = bool(success or terminated or rl_success(self.env))

                if token in MOVE_ATOMS:
                    recent_moves.insert(0, token)
                    del recent_moves[RECENT_MOVES_MAX:]

                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=wrist,
                    record=self._record(
                        step_idx,
                        token,
                        response,
                        success,
                        terminated or truncated,
                        auto_released=released is not None,
                    ),
                )

                if success:
                    end_reason = "success"
                    break
                if truncated:
                    # RoboLab's own time_out DoneTerm (episode_length_s) fired: the env is
                    # frozen from here on, so further decisions would be no-ops.
                    print(f"[mvtoken-robolab] step {step_idx}: env truncated (episode time-out).")
                    end_reason = "env_truncated"
                    break

                elapsed = time.monotonic() - loop_started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
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
                    "control_mode": "robolab_mvtoken",
                    "task": self.task_description,
                }
            )
            if self.close_env:
                try:
                    self.env.close()
                except Exception:
                    pass

        return EpisodeResult(
            success=success,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    # -- helpers -----------------------------------------------------------
    def _execute(self, token: str, obs: dict) -> tuple[dict, bool, bool]:
        """Apply one token: N motion steps + M settle steps, gripper tokens held.

        Returns the observation AFTER the whole decision, plus the accumulated
        terminated/truncated flags. Motion stops early once the env terminates or
        truncates -- RoboLab freezes an env at that point, so extra steps are wasted.
        """
        terminated = truncated = False
        if token in MOVE_ATOMS:
            action = self.controller.action_for_atomic(token)
            motion_steps = self.sim_steps_per_decision
            settle_steps = self.settle_steps_per_decision
        elif token in (GRASP_TOKEN, RELEASE_TOKEN):
            action = (
                self.controller.close_gripper()
                if token == GRASP_TOKEN
                else self.controller.open_gripper()
            )
            # A gripper token commands zero displacement; all its steps are "hold".
            motion_steps = self.gripper_hold_steps or self.sim_steps_per_decision
            settle_steps = 0
        else:
            action = self.controller.hold_action()
            motion_steps = 1
            settle_steps = 0

        # The orientation correction is recomputed EVERY control step: it is a function
        # of the current orientation, so re-sending one decision's action unchanged would
        # keep commanding a rotation the arm has already made. (This is also why the
        # rotation slots cannot simply be baked into `action` above.)
        for _ in range(motion_steps):
            obs, terminated, truncated, _info = step_robolab(
                self.env,
                self.controller.with_orientation_hold(action, rl_ee_quat(self.env)),
            )
            if terminated or truncated:
                return obs, terminated, truncated
        if settle_steps:
            hold = self.controller.hold_action()
            for _ in range(settle_steps):
                obs, terminated, truncated, _info = step_robolab(
                    self.env,
                    self.controller.with_orientation_hold(hold, rl_ee_quat(self.env)),
                )
                if terminated or truncated:
                    break
        return obs, terminated, truncated

    def _maybe_auto_release(self, step_idx: int) -> Optional[tuple[dict, bool, bool]]:
        """Reopen the gripper if the auto-release rule fires; return the post-RELEASE obs.

        Returns ``None`` when the rule is absent/disabled, the gripper is open, or it is
        holding something. Mirrors the ManiSkill runner, but reads the width from the
        Robotiq ``finger_joint`` (``rl_gripper_width``) and has to step the env for the
        RELEASE to take physical effect.
        """
        if self.auto_release is None or not self.auto_release.enabled:
            return None
        gripper_closed = self.controller.state.gripper_name == "CLOSE"
        if not gripper_closed:
            return None
        width_m = rl_gripper_width(self.env)
        if not self.auto_release.should_release(width_m, gripper_closed):
            return None
        print(
            f"[mvtoken-robolab] step {step_idx}: auto-release -- gripper width "
            f"{width_m:.4f}m < {self.auto_release.empty_width_m:.4f}m; opening gripper"
        )
        action = self.controller.open_gripper()
        obs = None
        terminated = truncated = False
        for _ in range(self.gripper_hold_steps or self.sim_steps_per_decision):
            obs, terminated, truncated, _info = step_robolab(self.env, action)
            if terminated or truncated:
                break
        return (obs, terminated, truncated) if obs is not None else None

    def _images(self, obs: dict[str, Any]):
        agentview = prepare_view(
            rl_rgb(obs, self.agentview_camera),
            rotation_degrees=self.agentview_rotation_degrees,
            flip=self.agentview_flip,
            crop_aspect=self.agentview_crop_aspect,
            square_size=self.agentview_square_size,
        )
        wrist = (
            prepare_view(
                rl_rgb(obs, self.wrist_camera),
                rotation_degrees=self.wrist_rotation_degrees,
                flip=self.wrist_flip,
                crop_aspect=self.wrist_crop_aspect,
                square_size=self.wrist_square_size,
            )
            if self.use_wrist_image
            else None
        )
        return agentview, wrist

    @staticmethod
    def _recent_moves_text(recent_moves: list[str]) -> str:
        return ", ".join(recent_moves) if recent_moves else "none"

    def _record(
        self,
        step_idx: int,
        token: str,
        response: Any,
        success: bool,
        env_done: bool,
        auto_released: bool = False,
    ) -> dict[str, Any]:
        eef = rl_tcp(self.env)
        record: dict[str, Any] = {
            "i": int(step_idx),
            "stage": "-",
            "act": token,
            "eef": [round(float(x), 3) for x in eef],
            "w": round(rl_gripper_width(self.env), 5),
            "grip": self.controller.state.gripper_name,
        }
        if auto_released:
            record["auto_release"] = True
        latency_s = (getattr(response, "payload", None) or {}).get("latency_s")
        if latency_s is not None:
            record["vlm_ms"] = int(round(float(latency_s) * 1000.0))
        if success:
            record["ok"] = True
        if env_done:
            record["env_done"] = True
        return record


def video_fps(video_fps: float) -> float:
    return min(30.0, max(0.5, float(video_fps)))
