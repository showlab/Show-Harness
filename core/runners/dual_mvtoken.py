"""Planner-free / stage-free closed-loop dual-arm rollout for the dual_cloth MVTOKEN LoRAs.

The dual-arm counterpart of :class:`core.runners.mvtoken.MvTokenRunner`, and the stripped-down
sibling of :class:`core.runners.dual.DualEpisodeRunner`: no subgoal planner, no stage, no
recovery plugins, no task verifier. Each step captures the three views, asks the policy for one
atomic token per arm, executes BOTH simultaneously on the physical Pipers (each controller's Z
floor still enforced), logs the frame, and repeats.

Two conventions differ from the single-arm loop, and both come straight from the training data
(LlamaFactory ``rollout_to_llamafactory.py --dual``):

  STILL — a first-class token, executed as NOTHING (the arm holds its setpoint), exactly as
          :meth:`core.runners.dual.DualEpisodeRunner._execute` does. The arms were teleoperated
          independently, so STILL is what the data recorded whenever one arm waited for the
          other. It IS kept in that arm's recent-move history: "STILL, STILL, MV_FWD" is
          precisely how the model learns to see that an arm is waiting.

  DONE  — the training data synthesizes DONE for BOTH arms on the same terminal frame, so the
          rollout ends only when BOTH arms emit it. A lone DONE means one arm believes the task
          is over while the other disagrees; that arm HOLDS (treated as STILL) and the rollout
          continues, rather than cutting the other arm off mid-motion.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

import core.ui.console as console
from core.runners.preemption import InterruptibleDecider
from core.record.episode_logger import EpisodeLogger, status_flags
from core.record.images import to_uint8_hwc
from core.v0_types import EpisodeResult
from plugins.auto_release import AutoReleasePlugin
from core.vlm.dual_mvtoken_roles import SIDES

GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
DONE_TOKEN = "DONE"
STILL_TOKEN = "STILL"
FALLBACK_TOKEN = "STILL"
# Match the training window (rollout_to_llamafactory.py RECENT_WINDOW = 5): each arm's prompt
# lists up to its 5 most recent MV_*/STILL tokens, newest first. GRASP/RELEASE are excluded,
# matching the converter's DUAL_HISTORY_TOKENS.
RECENT_MOVES_MAX = 5
HISTORY_TOKENS = frozenset(
    {"MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN", STILL_TOKEN}
)


class DualMvTokenRunner:
    """Closed-loop dual-arm MVTOKEN rollout on the physical Pipers (no planner, no stage)."""

    def __init__(
        self,
        *,
        session: Any,
        controllers: dict[str, Any],
        agent: Any,
        logger: EpisodeLogger,
        task: str,
        max_steps: int,
        loop_period_s: float,
        video_fps: float,
        debug: bool,
        viewer: Any = None,
        prompt_log_every: int = 20,
        auto_release: Optional[dict[str, AutoReleasePlugin]] = None,
        dagger_plugin: Any = None,
    ) -> None:
        self.session = session
        self.controllers = controllers
        # DAGGER: real-time human keyboard override (plugins.dagger). Keys arrive on the live
        # view's stream thread, so they land even while the VLM call is in flight (see
        # _decide_interruptible). None/disabled -> the loop is byte-identical to before.
        self.dagger_plugin = dagger_plugin
        # The single in-flight decide() call, if one was abandoned mid-step.
        self._decider: Optional[InterruptibleDecider] = None
        # Auto-release safety rule (plugins.auto_release), one plugin per arm: consulted after
        # each step to reopen a closed gripper whose measured width has collapsed below that
        # arm's empty_width_m. None (or a disabled tool) leaves the loop untouched.
        self.auto_release = auto_release or {}
        self.agent = agent
        self.logger = logger
        self.task = str(task)
        self.max_steps = int(max_steps)
        self.loop_period_s = float(loop_period_s)
        self.video_fps = float(video_fps)
        self.debug = bool(debug)
        self.viewer = viewer
        self.prompt_log_every = int(prompt_log_every)
        # Wall-clock + last VLM latency for the live-window telemetry line.
        self._t0 = time.monotonic()
        self._last_vlm_ms: Optional[int] = None

    # -- main loop ---------------------------------------------------------
    def run(self) -> EpisodeResult:
        steps = 0
        end_reason = "max_steps_reached"
        recent: dict[str, list[str]] = {side: [] for side in SIDES}
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"
        self._t0 = time.monotonic()
        self._last_vlm_ms = None

        # Start both arms from a known, open-gripper state. The controllers are already
        # synced (and their Z floors locked) by the caller.
        for side in SIDES:
            self.controllers[side].step(RELEASE_TOKEN)

        try:
            for step_idx in range(self.max_steps):
                steps = step_idx + 1
                loop_started = time.monotonic()

                obs = self.session.get_observation()
                agentview = to_uint8_hwc(obs["agentview"])
                wrist_left = to_uint8_hwc(obs["wrist_left"])
                wrist_right = to_uint8_hwc(obs["wrist_right"])
                images = (agentview, wrist_left, wrist_right)
                self._show_live(step_idx, images, "deciding ...", steps_done=step_idx)

                decide_kwargs = dict(
                    task=self.task,
                    recent_left=self._recent_text(recent["left"]),
                    recent_right=self._recent_text(recent["right"]),
                    agentview_image=agentview,
                    wrist_left_image=wrist_left,
                    wrist_right_image=wrist_right,
                    debug=self.debug,
                )
                human_sides: set[str] = set()
                try:
                    if self._dagger_enabled():
                        # DAGGER: the VLM call is preemptible -- the moment human keys arrive
                        # it is abandoned (and its result later dropped); None means the human
                        # owns this step.
                        decision = self._decide_interruptible(decide_kwargs)
                    else:
                        decision = self.agent.decide(**decide_kwargs)
                    if decision is not None:
                        tokens = {side: decision.tokens[side] for side in SIDES}
                    else:
                        # Human step: consume the intents; an arm without one holds STILL
                        # (never execute a model token beside a human one -- the joint
                        # decision it came from is stale by definition).
                        tokens, human_sides = self._human_tokens()
                except (RuntimeError, KeyError, ValueError) as exc:
                    # Degraded-output safeguard: never crash a live rollout on a bad VLM
                    # reply. Both arms HOLD -- unlike the single-arm loop's MV_DOWN fallback,
                    # a wrong guess here can drive two arms into each other or into the cloth.
                    print(
                        f"[dual-mvtoken] step {step_idx}: VLM parse failed ({exc}); "
                        f"holding both arms ({FALLBACK_TOKEN})"
                    )
                    decision = None
                    tokens = {side: FALLBACK_TOKEN for side in SIDES}
                self._capture_latency(decision)

                # A human step reuses the previous decision's prompt text, so only log on a
                # real model decision (matching core.runners.dual).
                if decision is not None and self.prompt_log_every > 0 and step_idx % self.prompt_log_every == 0:
                    prompt_text = getattr(self.agent, "last_prompt", "")
                    if prompt_text:
                        self.logger.save_controller_prompt(
                            step_idx,
                            prompt_text,
                            media=getattr(self.agent, "last_media", None),
                        )

                # Terminal token: BOTH arms must agree the task is over (see module docstring).
                # A lone DONE is downgraded to STILL so the other arm can finish its motion.
                done = [side for side in SIDES if tokens[side] == DONE_TOKEN]
                if len(done) == len(SIDES):
                    print(f"[dual-mvtoken] step {step_idx}: DONE on both arms -- ending rollout.")
                    self._show_live(
                        step_idx, images, "executed", steps_done=step_idx + 1, tokens=tokens
                    )
                    end_reason = "done"
                    break
                for side in done:
                    print(
                        f"[dual-mvtoken] step {step_idx}: {side} emitted DONE but "
                        f"{[s for s in SIDES if s not in done]} has not -- holding {side}."
                    )
                    tokens[side] = STILL_TOKEN

                results = self._execute(tokens)

                # Auto-release safety rule: a closed gripper whose measured width has
                # collapsed below the empty threshold is holding nothing -- reopen it
                # immediately so the next decision starts from a clean, open gripper.
                released = self._maybe_auto_release(step_idx, results)

                self._print_step(
                    step_idx, tokens, decision, results, released, human_sides
                )
                self._show_live(
                    step_idx, images, "executed", steps_done=step_idx + 1,
                    tokens=tokens, flag_recs=self._flag_recs(results, released),
                )

                for side in SIDES:
                    if tokens[side] in HISTORY_TOKENS:
                        recent[side].insert(0, tokens[side])
                        del recent[side][RECENT_MOVES_MAX:]

                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=[wrist_left, wrist_right],
                    record=self._record(
                        step_idx, tokens, results, decision, obs, released, human_sides
                    ),
                )

                elapsed = time.monotonic() - loop_started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
        except KeyboardInterrupt:
            end_reason = "interrupted"
            print(
                "\n[run-real-dual-mvtoken] Ctrl+C received -- stopping rollout and compiling "
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
                    "control_mode": "real_dual_mvtoken",
                    "scheme": getattr(self.agent, "scheme", "?"),
                    "task": self.task,
                    "z_floor_m": {
                        side: self.controllers[side].z_floor_m for side in SIDES
                    },
                }
            )

        return EpisodeResult(
            success=False,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    # -- execution ---------------------------------------------------------
    def _execute(self, tokens: dict[str, str]) -> dict[str, Any]:
        """Run both arms' tokens SIMULTANEOUSLY (one thread per acting arm).

        Same policy as :meth:`core.runners.dual.DualEpisodeRunner._execute`: STILL issues no
        controller call at all (the arm holds its setpoint), and a fault on one arm is joined
        then re-raised so it can never silently strand the other mid-motion.
        """
        results: dict[str, Any] = {}
        errors: dict[str, BaseException] = {}

        def _run(side: str, token: str) -> None:
            try:
                results[side] = self.controllers[side].step(token)
            except BaseException as exc:  # noqa: BLE001 - re-raised after join
                errors[side] = exc

        acting = {
            side: token
            for side, token in tokens.items()
            if token and token != STILL_TOKEN
        }
        if not acting:
            return results
        if len(acting) == 1:
            side, token = next(iter(acting.items()))
            results[side] = self.controllers[side].step(token)
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
            raise RuntimeError(f"[{side}] arm failed executing its token") from exc
        return results

    # -- DAGGER: preemptible VLM decisions -------------------------------------------
    def _dagger_enabled(self) -> bool:
        return bool(getattr(self.dagger_plugin, "enabled", False))

    def _human_tokens(self) -> tuple[dict[str, str], set[str]]:
        """Consume the pending DAGGER intents -> ``(tokens, sides_the_human_drove)``.

        An arm with no intent holds STILL: a model token must never execute beside a human
        one, because the joint decision it came from is stale by definition. The gripper
        placeholder resolves into GRASP/RELEASE from that controller's ACTUAL state
        (teleop's flush-time contract), so a toggle always does the opposite of what the
        gripper is currently doing.
        """
        human = self.dagger_plugin.drain()
        tokens: dict[str, str] = {}
        driven: set[str] = set()
        for side in SIDES:
            token, kind = human.get(side, (STILL_TOKEN, "still"))
            if kind == "gripper":
                token = (
                    RELEASE_TOKEN
                    if self.controllers[side].gripper_closed
                    else GRASP_TOKEN
                )
            if side in human:
                driven.add(side)
            tokens[side] = token
        return tokens, driven

    def _decide_interruptible(self, decide_kwargs: dict[str, Any]) -> Optional[Any]:
        """One controller ``decide``, preempted by human DAGGER keys.

        Delegates to :class:`core.runners.preemption.InterruptibleDecider`; ``None`` means
        the human owns this step (the runner executes the human intent instead)."""
        if self._decider is None:
            self._decider = InterruptibleDecider(
                self.dagger_plugin, lambda **kw: self.agent.decide(**kw)
            )
        return self._decider.decide(decide_kwargs)

    def _maybe_auto_release(self, step_idx: int, results: dict[str, Any]) -> set[str]:
        """Reopen each arm's gripper when its auto-release rule fires; return those sides.

        The dual counterpart of :meth:`core.runners.mvtoken.MvTokenRunner._maybe_auto_release`,
        applied per arm AFTER both tokens executed. The closed state is read from the
        controller (commanded intent), not the step result -- an arm that held STILL has no
        result, yet a grasp that slipped while it waited must still be caught. Each fired
        side's RELEASE result replaces its step result so the log shows the reopened state.
        """
        released: set[str] = set()
        for side in SIDES:
            rule = self.auto_release.get(side)
            if rule is None or not rule.enabled:
                continue
            controller = self.controllers[side]
            if not bool(getattr(controller, "gripper_closed", False)):
                continue
            width_m = controller.measured_gripper_width()
            if not rule.should_release(width_m, True):
                continue
            print(
                f"[dual-mvtoken] step {step_idx}: [{side}] auto-release -- gripper width "
                f"{width_m:.4f}m < {rule.empty_width_m:.4f}m; opening gripper"
            )
            results[side] = controller.step(RELEASE_TOKEN)
            released.add(side)
        return released

    # -- console -----------------------------------------------------------
    def _print_step(
        self,
        step_idx: int,
        tokens: dict[str, str],
        decision: Any,
        results: dict[str, Any],
        released: set[str],
        human_sides: set[str],
    ) -> None:
        """One readable block per step (the dual_runner convention): a rule, then
        one line per arm. The MVTOKEN policy emits bare tokens -- no reasoning --
        so the flags carry everything that qualified the action. Per-token
        controller chatter is silenced by the entrypoint (controller.verbose=False),
        so this block is the whole per-step story."""
        print(console.dim(f"\n─── step {step_idx:03d} " + "─" * 52))
        for side in SIDES:
            result = results.get(side)
            token = tokens[side]
            tag = console.c(console.SIDE_COLOR[side], f"  {side[0].upper()}")
            act = token if token == STILL_TOKEN else console.c(console.BOLD, token)
            grip = "CLOSED" if self.controllers[side].gripper_closed else "OPEN"
            flags = []
            if getattr(result, "grasp_empty", False):
                flags.append(console.c(console.RED, "empty close"))
            if result is not None and result.note and "z-floor" in result.note:
                flags.append(console.c(console.YELLOW, "blocked: z-floor"))
            if side in released:
                flags.append(console.c(console.YELLOW, "auto-release -> reopened"))
            if side in human_sides:
                flags.append(console.c(console.YELLOW, "human"))
            suffix = ("   " + " · ".join(flags)) if flags else ""
            print(f"{tag}  {'MVTOKEN':<10} {act:<20} {console.dim('grip ' + grip)}{suffix}")
        if decision is None and not human_sides:
            print(console.dim("  · fallback (bad VLM reply) -- holding both arms"))
        elif decision is not None and self._last_vlm_ms is not None:
            print(console.dim(f"  · VLM {self._last_vlm_ms} ms"))

    # -- live-view status --------------------------------------------------
    def _capture_latency(self, decision: Any) -> None:
        """Remember the last VLM latency in ms for the telemetry line."""
        latency_s = (getattr(decision, "payload", None) or {}).get("latency_s")
        if latency_s is not None:
            self._last_vlm_ms = int(round(float(latency_s) * 1000.0))

    def _telemetry_text(self, steps_done: int) -> str:
        elapsed = time.monotonic() - self._t0
        parts = [f"total {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"]
        if steps_done > 0:
            parts.append(f"avg {elapsed / steps_done:.1f} s/step")
        if self._last_vlm_ms is not None:
            parts.append(f"VLM {self._last_vlm_ms} ms")
        return "  ·  ".join(parts)

    @staticmethod
    def _flag_recs(
        results: dict[str, Any], released: set[str]
    ) -> dict[str, dict[str, Any]]:
        """Per-arm status-flag records in the shape ``status_flags`` reads."""
        recs: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            rec: dict[str, Any] = {}
            result = results.get(side)
            if result is not None:
                if getattr(result, "grasp_empty", False):
                    rec["grasp_fail"] = True
                if result.note and "z-floor" in result.note:
                    rec["blocked"] = "z_floor"
            if side in released:
                rec["auto_release"] = True
            recs[side] = rec
        return recs

    def _show_live(
        self,
        step_idx: int,
        images: tuple[np.ndarray, np.ndarray, np.ndarray],
        phase: str,
        steps_done: int,
        tokens: Optional[dict[str, str]] = None,
        flag_recs: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        """Post the step's status to the live window (three-view dual dashboard).

        With the stream thread running (run_real_dual_mvtoken starts it), this only
        updates the status overlay -- the stream supplies fresh frames continuously.
        NEVER call the legacy ``viewer.show()`` here: it renders from this thread
        while the stream thread renders the dual layout, and the two fight over the
        one pygame window (resize churn + cross-thread pygame calls)."""
        if self.viewer is None:
            return
        agentview, wrist_left, wrist_right = images
        arms: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            info: dict[str, Any] = {
                "stage": "MVTOKEN (stage-free)",
                "grip": "CLOSED" if self.controllers[side].gripper_closed else "OPEN",
            }
            if tokens is not None:
                info["token"] = tokens[side]
            if flag_recs is not None:
                info["flags"] = status_flags(flag_recs.get(side) or {})
            arms[side] = info
        self.viewer.show_dual(
            step=step_idx,
            agentview=agentview,
            wrist_left=wrist_left,
            wrist_right=wrist_right,
            arms=arms,
            task=self.task,
            phase=phase,
            telemetry=self._telemetry_text(steps_done),
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _recent_text(moves: list[str]) -> str:
        return ", ".join(moves) if moves else "none"

    def _record(
        self,
        step_idx: int,
        tokens: dict[str, str],
        results: dict[str, Any],
        decision: Any,
        obs: dict[str, Any],
        released: set[str] = frozenset(),
        human_sides: set[str] = frozenset(),
    ) -> dict[str, Any]:
        """Compact step record compatible with EpisodeLogger's analysis frame."""
        record: dict[str, Any] = {"i": int(step_idx), "stage": "-"}
        for side in SIDES:
            result = results.get(side)
            if result is not None and getattr(result, "post_pose", None) is not None:
                eef = np.asarray(result.post_pose, dtype=float).reshape(-1)[:3]
            else:
                eef = np.asarray(
                    (obs.get(side) or {}).get("ee_pose", []), dtype=float
                ).reshape(-1)[:3]
            tag = side[0].upper()  # L / R
            record[f"act_{tag}"] = tokens[side]
            record[f"eef_{tag}"] = [round(float(x), 3) for x in eef]
            record[f"w_{tag}"] = round(
                float((obs.get(side) or {}).get("gripper_width", 0.0)), 5
            )
            if result is not None:
                record[f"grip_{tag}"] = "CLOSED" if result.gripper_closed else "OPEN"
                if result.note and "z-floor" in result.note:
                    record[f"blocked_{tag}"] = "z_floor"
                if getattr(result, "grasp_empty", False):
                    record[f"grasp_fail_{tag}"] = result.note or True
            if side in released:
                record[f"auto_release_{tag}"] = True
        latency_s = (getattr(decision, "payload", None) or {}).get("latency_s")
        if latency_s is not None:
            record["vlm_ms"] = int(round(float(latency_s) * 1000.0))
        if decision is not None:
            record["scheme"] = (decision.payload or {}).get("scheme", "?")
        # Which arms the OPERATOR drove this step (DAGGER). Same field name/shape as
        # core.runners.dual, so the analysis frame and any downstream filtering of
        # human-driven steps work identically on both dual paths.
        if human_sides:
            record["dagger"] = "|".join(s[0].upper() for s in sorted(human_sides))
        return record


def video_fps(video_fps: float) -> float:
    return min(30.0, max(0.5, float(video_fps)))
