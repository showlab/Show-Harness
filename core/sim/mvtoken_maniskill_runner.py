"""Planner-free / stage-free ManiSkill rollout for the MVTOKEN Qwen LoRA.

Flat MVTOKEN simulator runner driving a
ManiSkill Panda in the translation-only ``pd_ee_delta_pos`` control mode instead of the
ManiSkill env. Each step feeds task + gripper
state + recent MV_* moves with the agentview (``base_camera``) + wrist (``hand_camera``)
images, maps the returned atomic token to a 4-dim ``[dx, dy, dz, gripper]`` action via
:class:`~interpreters.maniskill_atomic_controller.ManiskillAtomicController`, steps the env,
and checks ``info["success"]``. The episode ends on success, ``DONE``, or max_steps.

CROSS-DOMAIN experiment: the LoRA was trained on REAL Franka/Piper rollouts, so its
behaviour in the visually-different ManiSkill simulator is exploratory, not expected to
match real quality (a cross-domain caveat). Movement tokens are applied as
BASE-FRAME atomics (MV_DOWN = lower the gripper), matching the teleop labels -- the
base-frame axis mapping lives in ``configs/robot_maniskill.yaml`` and is tunable per camera.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.action_units import MOVE_ATOMS
from interpreters.maniskill_atomic_controller import ManiskillAtomicController
from core.record.episode_logger import EpisodeLogger
from core.record.images import prepare_view
from core.sim.maniskill_task import (
    ms_gripper_width,
    ms_is_grasped,
    ms_rgb,
    ms_success,
    ms_tcp,
    reset_maniskill,
    step_maniskill,
)
from core.v0_types import EpisodeResult, V0Config
from plugins.auto_release import AutoReleasePlugin

GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
DONE_TOKEN = "DONE"
FALLBACK_TOKEN = "MV_DOWN"
RECENT_MOVES_MAX = 5  # match the MVTOKEN training window (MV_* only, newest first)


# The full camera transform contract (rotate/flip -> crop -> letterbox) lives in
# core.record.images.prepare_view so the real2sim data generators apply the byte-identical
# transform to the frames they store, and so BOTH views go through one code path.


class MvTokenManiskillRunner:
    """Closed-loop MVTOKEN rollout in the ManiSkill simulator (no planner, no stage)."""

    def __init__(
        self,
        *,
        env: Any,
        task_description: str,
        controller: ManiskillAtomicController,
        agent: Any,
        logger: EpisodeLogger,
        config: V0Config,
        max_steps: int,
        num_steps_wait: int,
        loop_period_s: float,
        sim_steps_per_decision: int,
        seed: int,
        agentview_camera: str,
        wrist_camera: str,
        agentview_rotation_degrees: int,
        wrist_rotation_degrees: int,
        agentview_flip: str,
        wrist_flip: str,
        use_wrist_image: bool,
        reset_qpos: Any = None,
        layout: Optional[str] = None,
        auto_release: Optional[AutoReleasePlugin] = None,
        debug: bool,
        prompt_log_every: int = 20,
        agentview_square_size: Optional[int] = None,
        agentview_crop_aspect: Optional[float] = None,
        wrist_square_size: Optional[int] = None,
        wrist_crop_aspect: Optional[float] = None,
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
        self.seed = int(seed)
        self.agentview_camera = str(agentview_camera)
        self.wrist_camera = str(wrist_camera)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_flip = str(agentview_flip or "none")
        self.wrist_flip = str(wrist_flip or "none")
        self.use_wrist_image = bool(use_wrist_image)
        self.reset_qpos = reset_qpos
        # "wide" re-randomises the manipulated + target object exactly like the data
        # generators do (scripts/trajectory/real2sim/maniskill/tasks.py). The RLinf rigs
        # barely move their target by default (BlockPAP's coaster spans 4x2 cm), so without
        # this an eval episode sits OFF the training distribution and the numbers mean
        # nothing. Envs that already randomise both objects (official StackCube) pass None.
        self.layout = layout
        # Auto-release safety rule (plugins.auto_release): consulted after each step to
        # reopen a closed gripper that is holding nothing, so the next decision starts
        # clean instead of dragging an empty fist around. None -> rule absent (no-op).
        self.auto_release = auto_release
        self.debug = bool(debug)
        self.prompt_log_every = int(prompt_log_every)
        # Per-view geometry, SAME knobs for both views (see core.record.images.prepare_view):
        # rotate/flip -> centre-crop to crop_aspect -> letterbox into square_size.
        #
        # MUST match how the training set stores it: the ms_0717 sets were converted to
        # 256x256 via resize_with_pad to line up with the real-robot data, so an eval
        # against those LoRAs needs agentview_square_size=256. The wrist is already square
        # at 256 (scene.wrist_resolution), so its letterbox is a no-op -- but it is applied
        # through the same path, so a wrist camera reconfigured to a non-square resolution
        # gets the correct treatment instead of silently being sent raw.
        self.agentview_square_size = (
            int(agentview_square_size) if agentview_square_size else None
        )
        self.agentview_crop_aspect = (
            float(agentview_crop_aspect) if agentview_crop_aspect else None
        )
        self.wrist_square_size = int(wrist_square_size) if wrist_square_size else None
        self.wrist_crop_aspect = (
            float(wrist_crop_aspect) if wrist_crop_aspect else None
        )

    def run(self) -> EpisodeResult:
        # Frame history (if the agent keeps any) must not leak across episodes.
        reset_history = getattr(self.agent, "reset", None)
        if callable(reset_history):
            reset_history()
        obs, info = reset_maniskill(
            self.env,
            seed=self.seed,
            settle_steps=self.num_steps_wait,
            hold_action=self.controller.open_gripper(),
            reset_qpos=self.reset_qpos,
        )
        if self.layout == "wide":
            # Same sampler the training data was generated with, anchored at the settled
            # gripper XY, then a short settle so the objects rest before the first frame.
            import numpy as _np

            from scripts.trajectory.real2sim.maniskill.tasks import randomize_layout

            task_key = "blockpap" if hasattr(self.env.unwrapped, "target") else "blockstack"
            self.layout_info = randomize_layout(
                self.env, task_key, _np.random.default_rng(self.seed),
                ms_tcp(self.env)[:2], self.controller.step_m, snap=False,
            )
            for _ in range(6):
                obs, _t, _tr, info = step_maniskill(
                    self.env, self.controller.hold_action()
                )
        success = ms_success(info)
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
                        f"[mvtoken-maniskill] step {step_idx}: VLM token parse failed ({exc}); "
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
                    print(f"[mvtoken-maniskill] step {step_idx}: DONE emitted -- ending rollout.")
                    end_reason = "done"
                    break

                action = self._action_for_token(token)
                done = False
                for _ in range(self.sim_steps_per_decision):
                    obs, terminated, truncated, info = step_maniskill(self.env, action)
                    if terminated or truncated or ms_success(info):
                        done = True
                        break
                success = bool(ms_success(info))

                # Reflex, not policy: if that step left the gripper closed on nothing,
                # reopen it now so the next decision sees an open gripper.
                released = self._maybe_auto_release(step_idx)
                if released is not None:
                    obs, info = released
                    success = bool(success or ms_success(info))

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
                        info,
                        success,
                        done,
                        auto_released=released is not None,
                    ),
                )

                if success:
                    end_reason = "success"
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
                    "control_mode": "maniskill_mvtoken",
                    "task": self.task_description,
                }
            )
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
    def _action_for_token(self, token: str) -> np.ndarray:
        if token in MOVE_ATOMS:
            return self.controller.action_for_atomic(token)
        if token == GRASP_TOKEN:
            return self.controller.close_gripper()
        if token == RELEASE_TOKEN:
            return self.controller.open_gripper()
        return self.controller.hold_action()

    def _maybe_auto_release(self, step_idx: int) -> Optional[tuple[dict, dict]]:
        """Reopen the gripper if the auto-release rule fires; return the post-RELEASE
        ``(obs, info)``.

        Returns ``None`` when the rule is absent/disabled, the gripper is open, or it is
        holding something -- leaving the just-executed step untouched. Mirrors
        ``core.runners.mvtoken.MvTokenRunner._maybe_auto_release``, but reads the width from
        the sim (``ms_gripper_width``) and has to step the env for the RELEASE to take
        physical effect (the controller only sets the gripper command).
        """
        if self.auto_release is None or not self.auto_release.enabled:
            return None
        gripper_closed = self.controller.state.gripper_name == "CLOSE"
        if not gripper_closed:
            return None
        width_m = ms_gripper_width(self.env)
        if not self.auto_release.should_release(width_m, gripper_closed):
            return None
        print(
            f"[mvtoken-maniskill] step {step_idx}: auto-release -- gripper width "
            f"{width_m:.4f}m < {self.auto_release.empty_width_m:.4f}m; opening gripper"
        )
        action = self.controller.open_gripper()
        obs = info = None
        for _ in range(self.sim_steps_per_decision):
            obs, _terminated, _truncated, info = step_maniskill(self.env, action)
        return (obs, info) if obs is not None else None

    def _images(self, obs: dict[str, Any]):
        """Both views through the SAME transform chain -- see core.record.images.prepare_view.

        agentview and wrist differ only in their argument VALUES (the wrist is already
        square at 256, so its letterbox is a no-op), never in which steps run. They used
        to take two different code paths, which hid the fact that they are one operation.
        """
        agentview = prepare_view(
            ms_rgb(obs, self.agentview_camera),
            rotation_degrees=self.agentview_rotation_degrees,
            flip=self.agentview_flip,
            crop_aspect=self.agentview_crop_aspect,
            square_size=self.agentview_square_size,
        )
        wrist = (
            prepare_view(
                ms_rgb(obs, self.wrist_camera),
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
        info: dict[str, Any],
        success: bool,
        env_done: bool,
        auto_released: bool = False,
    ) -> dict[str, Any]:
        eef = ms_tcp(self.env)
        record: dict[str, Any] = {
            "i": int(step_idx),
            "stage": "-",
            "act": token,
            "eef": [round(float(x), 3) for x in eef],
            "w": round(ms_gripper_width(self.env), 5),
            "grip": self.controller.state.gripper_name,
        }
        if ms_is_grasped(info):
            record["grasped"] = True
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
